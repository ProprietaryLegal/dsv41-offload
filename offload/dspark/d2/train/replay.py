"""V1 same-input replay; private dumps, aggregate parity receipts only."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

from .model import Drafter, load
from .ops import frozen_linear
from .data import D2Blocks, sha256


def v1_import(root: str) -> None:
    if not (Path(root)/"verify/run_ref.py").is_file():
        raise FileNotFoundError(f"V1 verify modules missing under {root}")
    sys.path.insert(0, root)


def g1_drafts(path: Path, data: D2Blocks) -> dict[str, dict[int, list[int]]]:
    """Read D2-DATA g1.py's local seq_id/P output; require exact block coverage.

    G1's JSONL has no embedded manifest hash. Its file digest and the current
    manifest identity must therefore be kept together in the evaluation receipt.
    Use only the coordinator's G1 output for that exact dataset, never a guess.
    """
    result: dict[str, dict[int, list[int]]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        req = row["seq_id"]
        if req in result:
            raise ValueError("duplicate sequence in G1 K2 output")
        result[req] = {int(p): [int(t) for t in ids] for p, ids in row["drafts"].items()}
    expected_requests = {req for _, _, req in data.sequences}
    if set(result) != expected_requests:
        raise ValueError("G1 K2 request set differs from the selected dataset split")
    for _, index, req in data.sequences:
        length, gen_start = index[1:3]
        if set(result[req]) != set(range(gen_start-1, length-2)):
            raise ValueError("G1 K2 position coverage is incomplete or mismatched")
        if any(len(ids) != 5 or any(t < 0 or t >= 129280 for t in ids) for ids in result[req].values()):
            raise ValueError("G1 K2 draft shape or vocabulary invalid")
    return result


def load_capture_set(path: str, reader: str | None = None, split: str | None = None,
                     k2_mapping: str | None = None) -> Any:
    if reader is None:
        return importlib.import_module("verify.captures").CaptureSet(path)
    if split is None:
        raise ValueError("D2 shard replay requires --split")
    manifest = Path(path)
    data = D2Blocks(manifest, split)
    spec = importlib.util.spec_from_file_location("d2_data_shards", reader)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load D2-DATA reader {reader}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    drafts = None
    by_seq = None
    if k2_mapping:
        if Path(k2_mapping).suffix == ".jsonl":
            by_seq = g1_drafts(Path(k2_mapping), data)
        else:
            mapping = json.loads(Path(k2_mapping).read_text())
            if mapping["arm"] != "k2" or mapping["split"] != split or mapping["manifest_sha256"] != sha256(manifest):
                raise ValueError("K2 draft mapping identity/split mismatch")
            drafts = mapping["drafts"]
    paths = [str((manifest.parent/entry["file"]).resolve()) for entry in data.manifest["splits"][split]["files"]]
    cs = module.ShardCaptureSet(paths, drafts=drafts, drafts_by_seq=by_seq, verify_sha256=True)
    if k2_mapping and any(s.engine_drafts_placeholder for s in cs.samples):
        raise ValueError("K2 replay mapping has missing samples")
    return cs


def capture_inputs(cs: Any, s: Any, device: str) -> tuple[dict[str, Tensor], dict[str, object]]:
    positions = cs.window(s)
    status = cs.window_status(s)
    if status["missing"]:
        raise ValueError(f"incomplete V1 window for {s.key}")
    rows = [cs.rows[s.req][p] for p in positions]
    has = [r.taps is not None for r in rows]
    taps = torch.stack([r.taps.reshape(-1) if r.taps is not None else torch.zeros(15360, dtype=torch.bfloat16) for r in rows])
    mx = torch.stack([r.mx.reshape(-1) if r.mx is not None else torch.zeros(5120) for r in rows])
    inp = {"taps": taps[None], "engine_mx": mx[None], "has_taps": torch.tensor([has]),
           "positions": torch.tensor([positions]), "mask": torch.ones(1, len(rows), dtype=torch.bool),
           "anchor": torch.tensor([s.anchor]), "P": torch.tensor([s.P])}
    meta = {"n": len(rows), "taps_rows": sum(has), "engine_mx_rows": len(rows)-sum(has),
            "P_source": "taps" if has[-1] else "engine_mx", "first": positions[0], "last": positions[-1]}
    return {k: v.to(device) for k, v in inp.items()}, meta


def read_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if sha256(path) != digest:
        raise ValueError("checkpoint changed while being read; use an immutable archived step")
    return state, digest


def apply_trainables(model: Drafter, values: dict[str, Tensor]) -> None:
    named = dict(model.named_parameters())
    if set(values) != set(named):
        raise ValueError("checkpoint trainable names differ from Phase 2a model")
    with torch.no_grad():
        for name, p in named.items():
            if p.shape != values[name].shape or not torch.isfinite(values[name]).all():
                raise ValueError(f"bad adapter tensor {name}")
            p.copy_(values[name])


def apply_adapter(model: Drafter, path: str) -> str:
    state, digest = read_checkpoint(Path(path))
    if state.get("schema") != "dsv41.d2.train-checkpoint.v1" or state.get("frozen_identity") != model.frozen_identity:
        raise ValueError("adapter checkpoint frozen source differs from loaded model")
    apply_trainables(model, state["trainable"])
    return digest


@torch.no_grad()
def dump_sample(model: Drafter, inp: dict[str, Tensor], window: dict[str, object], meta: dict[str, object]) -> dict[str, Any]:
    tr: dict[str, Any] = {}
    hidden, hx = model.hidden(**inp, trace=tr)
    ids, conf = model.greedy(hidden, hx, inp["anchor"])
    base = torch.cat([frozen_linear(hidden.float(), model.t["head.weight"][lo:lo+4096])
                      for lo in range(0, model.c.vocab, 4096)], -1)
    provenance = {"git_sha": os.environ.get("D2_GIT_SHA", "uncommitted-snapshot"), "torch": torch.__version__,
                  "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "frozen_identity": model.frozen_identity, "reference_math": model.reference_math,
                  "dtype_policy": "BF16 frozen masters; packed FP4 forward FP64 sum; FP32 activations/router/Markov; STE QDQ", **meta}
    out: dict[str, Any] = {"drafts": ids[0].tolist(), "conf": conf[0].cpu(), "base_logits": base[0].cpu(),
                           "P": int(inp["P"][0]), "anchor": int(inp["anchor"][0]), "window": window, "meta": provenance,
                           "head_hidden": hidden[0].cpu(), "head_x": hx[0].cpu(),
                           "main_x_P": tr["main_x"][0, -1].cpu()}
    kvs = []
    for i in range(model.c.stages):
        st = tr[f"s{i}"]
        kvs.append(st["kv"][0].cpu())
        out[f"s{i}"] = {n: st[n][0].cpu() for n in ("attn_out", "ffn_in", "ffn_out", "h_out")}
        out[f"s{i}"]["route"] = {"idx": st["route"]["idx"][0].cpu(), "biased": st["route"]["biased"].cpu()}
    out["kv_window"] = torch.stack(kvs, 1)
    return out


def relative(a: Tensor, b: Tensor) -> float:
    return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--references", required=True, help="V1 dumps root with ref and mirror directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--reference-math", action="store_true", help="CPU unquantized reference math; GPU quantized GEMMs")
    ap.add_argument("--data-reader", help="D2-DATA shards.py; --captures then names MANIFEST.json")
    ap.add_argument("--split", choices=("dev", "heldout_legal", "heldout_general", "acap"))
    ap.add_argument("--k2-mapping")
    ap.add_argument("--native-manifest")
    ap.add_argument("--package-root", type=Path, help="verified merged native arm; owns native manifest and adapter provenance")
    ap.add_argument("--parity-gate", action="store_true", help="also require exact agreement for an adapted arm")
    a = ap.parse_args()
    if a.device != "cpu":
        from .safety import assert_gpu_lease
        assert_gpu_lease()
    v1_import(a.v1_root)
    cs = load_capture_set(a.captures, a.data_reader, a.split, a.k2_mapping)
    if cs.plumbing:
        raise ValueError(f"V1 plumbing errors: {len(cs.plumbing)}")
    samples = sorted(cs.samples, key=lambda s: (s.req, s.P, s.round))
    samples = [s for s in samples if (Path(a.references)/"ref"/f"{s.key}.pt").is_file()]
    # Spread 50 rounds across all four requests rather than the first request.
    if a.limit and len(samples) > a.limit:
        samples = [samples[j*len(samples)//a.limit] for j in range(a.limit)]
    if not samples:
        raise ValueError("no reference-covered samples")
    torch.set_num_threads(8)
    package = None
    if a.package_root:
        from .reference import verify_package
        if a.adapter or a.native_manifest or Path(a.ckpt).resolve() != a.package_root.resolve():
            raise ValueError("--package-root must equal --ckpt and cannot combine with adapter/native-manifest")
        package = verify_package(a.package_root)
        a.native_manifest = str(a.package_root/"NATIVE_MANIFEST.json")
    model = load(a.ckpt, a.device, oracle=a.oracle, reference_math=a.reference_math, native_manifest=a.native_manifest).eval()
    adapter_identity = apply_adapter(model, a.adapter) if a.adapter else (package["adapter_sha256"] if package else None)
    out = Path(a.out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []
    begin = time.monotonic()
    for s in samples:
        inp, window = capture_inputs(cs, s, model.activation_device)
        arm = dump_sample(model, inp, window, {"arm": "D2-TRAIN", "adapter": a.adapter, "native_root": a.ckpt,
                                              "adapter_sha256": adapter_identity, "package_identity": package["drafter_identity"] if package else None,
                                              "accumulation": str(model.acc), "uuid": os.environ.get("CUDA_VISIBLE_DEVICES")})
        arm["sample"] = {"key": s.key, "req": s.req, "P": s.P, "realized": s.realized,
                         "engine_drafts": s.engine_drafts, "engine_drafts_placeholder": getattr(s, "engine_drafts_placeholder", False)}
        if a.k2_mapping:
            arm["meta"]["k2_mapping_sha256"] = sha256(Path(a.k2_mapping))
            arm["meta"]["target_manifest_sha256"] = sha256(Path(a.captures))
        torch.save(arm, out/f"{s.key}.pt")
        ref = torch.load(Path(a.references)/"ref"/f"{s.key}.pt", weights_only=False, map_location="cpu")
        mir = torch.load(Path(a.references)/"mirror"/f"{s.key}.pt", weights_only=False, map_location="cpu")
        err = relative(arm["base_logits"], ref["base_logits"])
        band = relative(mir["base_logits"], ref["base_logits"])
        row = {"key": s.key, "drafts_equal": arm["drafts"] == ref["drafts"], "logit_relerr": err,
               "storage_band": band, "within_v1_3x_band": err <= 3*band,
               "seconds": time.monotonic()-begin}
        receipts.append(row)
        (out/"parity.json").write_text(json.dumps({"samples": len(receipts), "rows": receipts,
                "reference_math": model.reference_math, "frozen_identity": model.frozen_identity,
                "acc": str(model.acc), "pass": all(r["drafts_equal"] and r["within_v1_3x_band"] for r in receipts)}, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    if ((not a.adapter and not package) or a.parity_gate) and not all(r["drafts_equal"] and r["within_v1_3x_band"] for r in receipts):
        raise SystemExit("D2 unadapted parity gate failed; see parity.json")


if __name__ == "__main__":
    main()
