#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S11 P-2 on a CUDA device: every refusal path of the async hand-off releases the process proxy's shared region.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 gpu_refusal_probe.py --pkg <dir with remote_dspark> --tests <dir with sim.py>

Cases (each creates the region through speculator._make_stream_sync, i.e. a memfd registered Portable|DeviceMap with
CUDA, and checks by the memfd's inode that afterwards no fd and no mapping of it is left in this process):
  register_refused   cuMemHostRegister refuses (RegisteredRegion raises)
  hostflag_refused   the hostflag library cannot be loaded (HostFlagSync raises after the registration)
  selftest_refused   the hand-off self-test fails (after registration, page attach and the stream test)
  proxy_start        the self-test passes, then the proxy process exits at once: make_remote_round falls back to
                     synchronous rounds (watchdog stopped, region unregistered and unmapped) and a round runs
  accepted           control: nothing refused, the region stays mapped and registered
Prints one JSON line per case and RESULT {...}; exit 0 only if every case passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import torch

MEMFD = "memfd:dsv41-remote-dspark"


def held(ino: int) -> tuple[int, int]:
    fds = 0
    for f in os.listdir("/proc/self/fd"):
        try:
            if MEMFD in os.readlink(f"/proc/self/fd/{f}") and os.stat(f"/proc/self/fd/{f}").st_ino == ino:
                fds += 1
        except OSError:
            pass
    with open("/proc/self/maps") as m:
        maps = sum(1 for line in m if MEMFD in line and int(line.split()[4]) == ino)
    return fds, maps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--allow-device", default="GPU-7744b44b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.startswith(args.allow_device) or "," in vis:
        raise SystemExit(f"refusing: CUDA_VISIBLE_DEVICES must be exactly {args.allow_device}-... (is {vis!r})")
    sys.path[:0] = [args.pkg, args.tests]
    import vllm_stubs

    vllm_stubs.install()
    import sim
    S = importlib.import_module("remote_dspark.speculator")
    shm = importlib.import_module("remote_dspark.shmregion")
    ss = importlib.import_module("remote_dspark.streamsync")
    pp = importlib.import_module("remote_dspark.proxy_proc")
    R = importlib.import_module("remote_dspark.round")
    cfgmod = importlib.import_module("remote_dspark.config")
    wire = importlib.import_module("remote_dspark.wire")
    mw = importlib.import_module("remote_dspark.mock_worker")
    stats_mod = importlib.import_module("remote_dspark.stats")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    ident = bytes(range(32))
    hello = wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ident)
    srv = mw.TcpStubServer(mw.WorkerCore(hello), "127.0.0.1", 0).start()
    env = {"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{srv.port}", "VLLM_DSPARK_REMOTE_IDENTITY": ident.hex(),
           "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_ASYNC": "1",
           "VLLM_DSPARK_REMOTE_ASYNC_PROXY": "process", "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
           "VLLM_DSPARK_REMOTE_STATS_FILE": "", "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2",
           "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4"}
    created: list[int] = []
    real_create = shm.SharedRegion.create.__func__

    def create(cls, layout):
        reg = real_create(cls, layout)
        created.append(os.fstat(reg.fd).st_ino)
        return reg

    shm.SharedRegion.create = classmethod(create)
    results = []

    def case(name, cfg_over=None, patch=None, expect_sync=False, proxy_exit=False):
        created.clear()
        e = dict(env)
        e.update(cfg_over or {})
        cfg = cfgmod.RemoteConfig.from_env(e)
        st = stats_mod.Stats()
        undo = patch() if patch else None
        boot = pp.BOOT
        if proxy_exit:
            pp.BOOT = "import sys\nsys.exit(3)\n"
        out = {"case": name}
        try:
            sync = S._make_stream_sync(cfg, dev, st, hello)
            out["async_fallback_reason"] = st.snapshot().get("async_fallback_reason")
            if proxy_exit:
                rnd = R.make_remote_round(cfg, hello, device=dev, tp_rank=0, broadcast_fn=None, stats=st,
                                          background=True, stream_sync=sync)
                out["async_after_start"] = rnd.async_mode
                out["async_fallback_reason"] = st.snapshot().get("async_fallback_reason")
                out["watchdog_running"] = sync._wd is not None
                import time
                t_end = time.monotonic() + 5
                while not rnd.client.ready and time.monotonic() < t_end:
                    time.sleep(0.01)
                eng = sim.Engine(seed=4)
                eng.new_request(0, prompt_len=40)
                for _ in range(3):                    # synchronous rounds on the device
                    sd = eng.step()
                    sd.batch.positions = sd.batch.positions.to(dev)
                    sd.batch.idx_mapping = sd.batch.idx_mapping.to(dev)
                    res = rnd.run(sd.batch, [t.to(dev) for t in sd.taps], sd.num_sampled.to(dev),
                                  sd.num_rejected.to(dev), sd.last_sampled.to(dev), sd.next_prefill_tokens.to(dev))
                    out.setdefault("statuses", []).append(res.read_status())
                out["ok_rounds"] = out["statuses"].count(0)
                rnd.client.close()
                sync = None
            torch.cuda.synchronize(dev)
            ino = created[0] if created else None
            out["region_held"] = held(ino) if ino is not None else None
            if expect_sync:
                out["ok"] = sync is None and out["region_held"] == (0, 0) and bool(out["async_fallback_reason"])
                if proxy_exit:
                    out["ok"] = (out["ok"] and out["async_after_start"] is False and not out["watchdog_running"]
                                 and out["ok_rounds"] >= 1)
            else:
                out["ok"] = sync is not None and out["region_held"] == (2, 1)
        finally:
            pp.BOOT = boot
            if undo:
                undo()
        print(json.dumps(out), flush=True)
        results.append(out)

    def refuse_register():
        real = ss.RegisteredRegion.__init__

        def init(self, region, device):
            raise ss.StreamSyncError("cuMemHostRegister(region) failed: CUresult 1 (injected)")
        ss.RegisteredRegion.__init__ = init
        return lambda: setattr(ss.RegisteredRegion, "__init__", real)

    def refuse_selftest():
        real = ss.HostFlagSync.self_test

        def st(self, timeout_s=2.0):
            ok, why = real(self, timeout_s)          # the real test runs (stream parked and drained) ...
            return False, f"injected refusal after the real self-test ({ok}, {why})"
        ss.HostFlagSync.self_test = st
        return lambda: setattr(ss.HostFlagSync, "self_test", real)

    case("register_refused", patch=refuse_register, expect_sync=True)
    case("hostflag_refused", cfg_over={"VLLM_DSPARK_REMOTE_HOSTFLAG_LIB": "/nonexistent/libdsv41_hostflag.so"},
         expect_sync=True)
    case("selftest_refused", patch=refuse_selftest, expect_sync=True)
    case("proxy_start", expect_sync=True, proxy_exit=True)
    case("accepted")
    srv.stop()
    ok = all(r["ok"] for r in results)
    res = {"ok": ok, "cases": results, "device": torch.cuda.get_device_name(dev)}
    print("RESULT " + json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
