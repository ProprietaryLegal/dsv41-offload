"""Independent official-reference parity on all four unmodified N1 tap windows."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

from .model import load
from .replay import dump_sample, relative, v1_import
from .safety import assert_gpu_lease


def fixtures(a: argparse.Namespace) -> None:
    assert_gpu_lease()
    sys.path.append(a.n1_repo)
    source = importlib.import_module("offload.dspark.worker.tools.parity_ref")
    bench = importlib.import_module("offload.dspark.worker.tools.tp2_bench")
    m = load(a.ckpt, "cuda", reference_math=True).eval()
    toks = bench.setup_inputs(torch.device("cuda"), a.worker_root)
    wrapper = type("Embedding", (), {"embed": m.t["embed.weight"]})()
    taps = source.make_taps(wrapper, toks, torch.Generator(device="cuda").manual_seed(5)).cpu()
    toks = toks.cpu()
    a.out.mkdir(parents=True, mode=0o700, exist_ok=True)
    inputs, dumps = [], []
    for P in bench.LASTS:
        positions = torch.arange(P-127, P+1)[None]
        inp = {"taps": taps[positions], "engine_mx": torch.zeros(1, 128, 5120), "positions": positions,
               "has_taps": torch.ones(1, 128, dtype=torch.bool), "mask": torch.ones(1, 128, dtype=torch.bool),
               "P": torch.tensor([P]), "anchor": toks[P+1:P+2]}
        wm = {"n": 128, "taps_rows": 128, "engine_mx_rows": 0, "P_source": "taps", "first": P-127, "last": P}
        dump = dump_sample(m, inp, wm, {"arm": "D2-N1-synthetic", "source": "N1 benchmark make_taps/setup_inputs seed5"})
        torch.save({"inputs": inp, "dump": dump}, a.out/f"n1-{P}.pt")
        inputs.append(inp)
        dumps.append(dump)
        print(json.dumps({"window_P": P, "written": True}), flush=True)
    with torch.no_grad():
        batch = {name: torch.cat([inp[name] for inp in inputs], 0) for name in inputs[0]}
        hidden, head_x = m.hidden_from(batch)
        picks, _ = m.greedy(hidden, head_x, batch["anchor"])
        errors = [relative(hidden[i], d["head_hidden"]) for i, d in enumerate(dumps)]
        equal = [picks[i].tolist() == d["drafts"] for i, d in enumerate(dumps)]
        result = {"pass": all(equal) and max(errors) <= 1e-6, "batch": 4, "draft_blocks_equal": sum(equal),
                  "max_head_hidden_relerr": max(errors)}
        (a.out/"batch-parity.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        if not result["pass"]:
            raise SystemExit("batched N1 windows differ from their independent forwards")


def reference(a: argparse.Namespace) -> None:
    v1_import(a.v1_root)
    NativeRef = importlib.import_module("verify.native_ref").NativeRef
    paths = sorted(a.fixtures.glob("n1-*.pt"))
    if len(paths) != 4:
        raise ValueError("all four N1 benchmark windows required")
    ref = NativeRef(a.ckpt, a.official, mode="ref", max_seq_len=1024)
    rows = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        inp, arm = data["inputs"], data["dump"]
        positions = inp["positions"][0].tolist()
        mx = [ref.main_x(t) for t in inp["taps"][0]]
        kv = torch.stack([ref.stage_kv(p, x) for p, x in zip(positions, mx)])
        ref.reset_window()
        ref.fill_window(positions, kv)
        rr = ref.draft(int(inp["P"][0]), int(inp["anchor"][0]), mx[-1], keep_full_logits=True)
        error = relative(arm["base_logits"], rr["base_logits"])
        row = {"P": int(inp["P"][0]), "drafts_equal": arm["drafts"] == rr["drafts"], "logit_relerr": error,
               "pass": arm["drafts"] == rr["drafts"] and error <= 1e-6}
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps({"pass": all(r["pass"] for r in rows), "windows": rows}, indent=2)+"\n")
    if not all(r["pass"] for r in rows):
        raise SystemExit("N1 synthetic parity failed")


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    gpu = sub.add_parser("fixtures")
    for n in ("ckpt", "n1-repo", "worker-root"):
        gpu.add_argument("--"+n, required=True)
    gpu.add_argument("--out", required=True, type=Path)
    cpu = sub.add_parser("reference")
    for n in ("ckpt", "official", "v1-root"):
        cpu.add_argument("--"+n, required=True)
    cpu.add_argument("--fixtures", required=True, type=Path)
    cpu.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.set_num_threads(8)
    fixtures(a) if a.command == "fixtures" else reference(a)


if __name__ == "__main__":
    main()
