"""DP-5 phase 1: drafter STEP latency against locked graphics clocks on ONE RTX 3090 (x86host GPU-bc62df2a only).

The production pair idles ~90 % of each round (worker ~2.5 ms of a ~39 ms B1 round) with no clock lock, so the SM
clock may ramp down between STEPs and back up inside the next one. DW1's timer issued STEPs back to back and could not
see that. This tool loads the fb68c52 one-card worker once (dw1_stages.build_one_card), then for each clock setting
(`sudo -n nvidia-smi -i <UUID> -lgc MIN,MAX`) times the live shapes with a production-like idle gap before every STEP,
recording handle wall time, the two graph GPU times and the SM clock read through NVML just before the STEP.
The card's default (`-lgc 0,1950`, nvidia-power-limit.sh) is restored in a finally block. Power limit untouched.

Usage (from offload/dspark, CUDA_VISIBLE_DEVICES=GPU-bc62df2a-...):
  python -m worker.tools.dp5_clocks ROOT OUT.json ITERS SETTING [SETTING ...]
  SETTING = MIN,MAX in MHz (e.g. 0,1950 default; 0,2100 like the unlocked pair; 1800,1800 locked) or gap0:MIN,MAX for a
  back-to-back (no idle gap) run of that setting.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from .. import protocol as P
from .dw1_stages import ROWS, SHAPES, Timers, build_one_card
from .server_selftest import Client, Shadow1, taps_for

UUID = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"
DEFAULT_LGC = "0,1950"   # nvidia-power-limit.sh: -lgc 0,<hw max 2100 - 150> on the solos (NDM 2026-10-13)
# ESTIMATED production idle gap before a STEP, per request count: round minus worker time. B1 38.9 ms, B3 80.8 ms,
# B4 92.1 ms (control archive, DRAFTER-REVIEW §2.4); B2 interpolated.
GAP_MS = {1: 36.0, 2: 55.0, 3: 77.0, 4: 88.0}
NVML_CLOCK_SM = 1


def parse_setting(s: str) -> tuple[str, bool]:
    gap = True
    if s.startswith("gap0:"):
        gap, s = False, s[5:]
    lo, hi = s.split(",")
    if not (lo.isdigit() and hi.isdigit()) or int(hi) == 0 or int(lo) > int(hi) or int(hi) > 2100:
        raise SystemExit(f"bad clock setting {s!r}: MIN,MAX with 0 <= MIN <= MAX <= 2100")
    return f"{int(lo)},{int(hi)}", gap


def stats(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    q = np.percentile(a, [10, 50, 90, 95, 99])
    return {"p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1), "p90": round(float(q[2]), 1),
            "p95": round(float(q[3]), 1), "p99": round(float(q[4]), 1), "mean": round(float(a.mean()), 1),
            "sd": round(float(a.std()), 1), "max": round(float(a.max()), 1), "n": int(a.size)}


class Nvml:
    """Minimal NVML through ctypes (pynvml is not in the worker venv): SM clock of one device by UUID."""

    def __init__(self, uuid: str) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit_v2 failed")
        self.h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h)) != 0:
            raise RuntimeError(f"nvmlDeviceGetHandleByUUID({uuid}) failed")
        self.v = ctypes.c_uint()

    def sm_mhz(self) -> int:
        if self.lib.nvmlDeviceGetClockInfo(self.h, NVML_CLOCK_SM, ctypes.byref(self.v)) != 0:
            raise RuntimeError("nvmlDeviceGetClockInfo failed")
        return int(self.v.value)


def set_clocks(lgc: str) -> None:
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", UUID, "-lgc", lgc], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    root, out_path, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    settings = [parse_setting(s) for s in sys.argv[4:]]
    if not settings:
        raise SystemExit("no clock settings given")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-bc62df2a"):
        raise SystemExit("DP-5 runs only on GPU-bc62df2a (coordinator grant)")
    nv = Nvml(UUID)
    rep: dict = {"device": torch.cuda.get_device_name(0), "uuid": UUID, "iterations": iters, "gap_ms": GAP_MS,
                 "default_lgc": DEFAULT_LGC, "tree": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                 "settings": {}}
    try:
        set_clocks(DEFAULT_LGC)
        core, m = build_one_card(root)
        sh = Shadow1(m)
        exp = core.expected
        rb, N, W = exp.row_bytes, exp.num_spec, exp.window
        cl = Client(core=core)
        h, _b = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "client_name": "dp5"}).encode())
        assert h.status == P.OK, P.FRAME_STATUS[h.status]
        gen = torch.Generator(device=m.dev).manual_seed(11)
        L = 128 + 6 * (iters + 40) + 8
        toks = torch.randint(0, 120000, (L,), generator=gen, device=m.dev)
        taps = taps_for(sh, toks, gen)
        starts = [W + 1] * 4

        def warm(nreq: int) -> None:
            for s_ in range(nreq):
                ap = starts[s_]
                body = P.encode_step(1, [P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 700 + s_, ap - W, W,
                                                    int(toks[ap]), ap)], taps[ap - W:ap].tobytes(), rb, 0)
                h2, _ = cl.call(P.OP_STEP, body)
                assert h2.status == P.OK, P.FRAME_STATUS[h2.status]

        def body_for(nreq: int, a0: list[int], n_out: int) -> bytes:
            ents, rows = [], []
            for s_ in range(nreq):
                ap = a0[s_]
                ents.append(P.ReqEntry(s_, P.REQ_NEED_DRAFT, N, 700 + s_, ap - ROWS, ROWS, int(toks[ap]), ap,
                                       n_out=n_out))
                rows.append(taps[ap - ROWS:ap])
            return P.encode_step(9, ents, np.concatenate(rows).tobytes(), rb, 0)

        tm = Timers(core, m)
        for idx, (lgc, gap) in enumerate(settings):
            set_clocks(lgc)
            time.sleep(2.0)
            key = f"{idx:02d}_{lgc}_{'gap' if gap else 'gap0'}"
            shapes: dict = {}
            for nreq, n_out in SHAPES:
                warm(nreq)
                tm.take()
                a0 = list(starts)
                rec: dict[str, list[float]] = {}
                gap_s = GAP_MS[nreq] / 1e3 if gap else 0.0
                for it in range(iters + 20):
                    a0 = [x + ROWS for x in a0]
                    body = body_for(nreq, a0, n_out)
                    if gap_s:
                        time.sleep(gap_s)
                    clk0 = nv.sm_mhz()
                    t0 = time.perf_counter()
                    h3, b = cl.call(P.OP_STEP, body)
                    handle = (time.perf_counter() - t0) * 1e6
                    clk1 = nv.sm_mhz()
                    assert h3.status == P.OK
                    sts = P.decode_step_reply(b).statuses
                    assert all(x[0] == P.R_OK for x in sts), sts
                    t = tm.take()
                    if it < 20:
                        continue
                    t["handle"] = handle
                    t["host_extra"] = handle - t.get("insert_gpu", 0.0) - t.get("draft_gpu", 0.0)
                    t["sm_mhz_before"] = float(clk0)
                    t["sm_mhz_after"] = float(clk1)
                    for k_, v in t.items():
                        rec.setdefault(k_, []).append(v)
                shapes[f"{nreq}req_nout{n_out or N}"] = {k_: stats(v) for k_, v in rec.items()}
                s1 = shapes[f"{nreq}req_nout{n_out or N}"]
                print(key, f"{nreq}req", "handle p50/p95", s1["handle"]["p50"], s1["handle"]["p95"],
                      "draft_gpu p50", s1["draft_gpu"]["p50"], "clk before p50", s1["sm_mhz_before"]["p50"],
                      flush=True)
            rep["settings"][key] = {"lgc": lgc, "gap": gap, "shapes": shapes}
            with open(out_path, "w") as fh:          # incremental: a stop keeps every finished setting
                json.dump(rep, fh, indent=1)
    finally:
        set_clocks(DEFAULT_LGC)
        rep["restored_lgc"] = DEFAULT_LGC
        with open(out_path, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
