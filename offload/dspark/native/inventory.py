"""N1-NATIVE inventory: every official DeepSeek-V4.1-Flash-1017 draft tensor (all mtp.* keys plus the target embed /
head the draft shares), its format, bytes and sha256, value-domain checks for native serving, and what the current
worker (TENSORS.json, legal-v2a / HeNryous EXL3 delivery) holds in its place.

Runs on x86host against the verified official shards in /opt/dsv41/x86host-data/native-1017 (fetch.sh / fetch2.sh,
RECEIPT.txt). Read-only. Writes NATIVE_MANIFEST.json (path given on the command line).

  python -m offload.dspark.native.inventory --root /opt/dsv41/x86host-data/native-1017 \
      --tensors offload/dspark/TENSORS.json --out offload/dspark/native/NATIVE_MANIFEST.json

Classification per tensor:
  supply   "shared-target"  the draft uses the target's tensor (embed.weight, head.weight: DSparkBlock.embed / .head are
                            the Transformer's own; convert.py drops any mtp.*.embed/head copy)
           "draft-only"     an mtp.* tensor only the draft reads
  official the official storage format: mxfp4 (I8 packed E2M1, E8M0 per 32 inputs), fp8 (E4M3, E8M0 per 32x32 block),
           e8m0 (a scale), bf16, f32
  worker   what the deployed worker serves today: "official-bytes" (byte-identical to the official tensor, proven by
           sha256), "exl3-K<k>" (re-quantized by the HeNryous delivery), "decoded-at-load" (official bytes decoded
           exactly at load, e.g. wo_a FP8 -> fp32 scales), or "absent" (not served: bias_vl, unused)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from . import formats as F

OFFICIAL_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
OFFICIAL_REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SHARED = ("embed.weight", "head.weight")
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "I32": 4}


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def role_of(name: str) -> str:
    if name in SHARED:
        return name.split(".")[0]
    rest = name.split(".", 2)[2]
    if ".experts." in name and "shared" not in name:
        return "routed_expert"
    for key, role in (("shared_experts", "shared_expert"), ("ffn.gate", "router"), ("hc_", "mhc"),
                      ("attn_sink", "attn_sink"), ("markov_head", "markov"), ("confidence_head", "confidence"),
                      ("main_proj", "main_proj"), ("norm", "norm"), ("attn.w", "attention")):
        if key in rest:
            return role
    return "other"


def worker_index(tensors_json: str) -> dict:
    with open(tensors_json) as f:
        t = json.load(f)
    by = {}
    for e in t["tensors"]:
        by[e["name"]] = e
    return by


def worker_view(name: str, dtype: str, wk: dict) -> dict:
    """What the deployed worker holds for official tensor ``name``."""
    if name in wk:
        e = wk[name]
        rep = e["representation"]
        return {"worker": "bytes", "worker_representation": rep, "worker_sha256": e.get("source_byte_range_sha256"),
                "worker_provenance": (e.get("provenance") or {}).get("repo")}
    base = re.sub(r"\.(weight|scale)$", "", name)
    if base + ".trellis" in wk:
        e = wk[base + ".trellis"]
        return {"worker": f"exl3-K{e['exl3_k']}", "worker_representation": "EXL3", "worker_sha256": None,
                "worker_provenance": (e.get("provenance") or {}).get("repo")}
    return {"worker": "absent", "worker_representation": None, "worker_sha256": None, "worker_provenance": None}


# ----------------------------------------------------------------------------------------------- value checks


def _f(x) -> float | None:
    return None if x is None else float(x)


def stats_float(t: torch.Tensor) -> dict:
    v = t.to(torch.float32)
    fin = torch.isfinite(v)
    a = v.abs()
    nz = a[(a > 0) & fin]
    return {"n": v.numel(), "nonfinite": int((~fin).sum()), "min": _f(v.min()), "max": _f(v.max()),
            "absmax": _f(a.max()), "min_nonzero_abs": _f(nz.min()) if nz.numel() else None,
            "zeros": int((v == 0).sum()), "exact_fp16": F.exact_in(v, torch.float16) if bool(fin.all()) else False,
            "exact_bf16": F.exact_in(v, torch.bfloat16) if bool(fin.all()) else False}


def scale_stats(s_u8: torch.Tensor) -> dict:
    s = s_u8.view(torch.uint8)
    nan = int((s == F.E8M0_NAN).sum())
    e = s.to(torch.int32) - 127
    h = torch.bincount(s.flatten().long(), minlength=256)
    return {"n": s.numel(), "nan": nan, "exp_min": int(e.min()), "exp_max": int(e.max()),
            "exp_hist": {str(i - 127): int(c) for i, c in enumerate(h.tolist()) if c}}


def decoded_stats(v: torch.Tensor) -> dict:
    a = v.abs()
    nz = a[a > 0]
    return {"absmax": _f(a.max()), "min_nonzero_abs": _f(nz.min()) if nz.numel() else None,
            "zeros": int((v == 0).sum()), "exact_fp16": F.exact_in(v, torch.float16),
            "exact_bf16": F.exact_in(v, torch.bfloat16)}


def check_pair(w: torch.Tensor, s: torch.Tensor, fmt: str) -> dict:
    """Format / alignment / code-domain / decoded-range checks of one quantized weight with its scale."""
    out: dict = {}
    if fmt == "mxfp4":
        codes = F.fp4_codes(w)
        O, I = codes.shape
        out["shape_ok"] = tuple(s.shape) == (O, I // 32) and I % 32 == 0
        out["code_hist"] = torch.bincount(codes.flatten().long(), minlength=16).tolist()
        out["neg_zero_codes"] = out["code_hist"][8]
        v = F.decode_fp4(w, s)
    else:
        O, I = w.shape
        out["shape_ok"] = tuple(s.shape) == ((O + 31) // 32, (I + 31) // 32)
        wb = w.view(torch.uint8)
        out["nan_codes"] = int(((wb == 0x7F) | (wb == 0xFF)).sum())
        out["neg_zero_codes"] = int((wb == 0x80).sum())
        out["code_absmax"] = _f(F.e4m3_codes_to_f32(w).abs().max())
        v = F.decode_fp8_block(w, s)
    out["pad_rows"] = (-O) % 32
    out["pad_cols"] = (-I) % 32
    out["decoded"] = decoded_stats(v)
    return out


def tp2_cuts(name: str, shape: list) -> dict | None:
    """The planned TP2 cut of this tensor and whether it lands on 32-element scale groups."""
    base = name.rsplit(".", 1)[0]
    rules = {"attn.wq_b": ("rows", 2), "attn.wo_a": ("rows", 2), "attn.wo_b": ("cols", 2),
             "shared_experts.w1": ("rows", 2), "shared_experts.w3": ("rows", 2), "shared_experts.w2": ("cols", 2)}
    for k, (axis, tp) in rules.items():
        if base.endswith(k):
            n = shape[0] if axis == "rows" else shape[1]
            return {"axis": axis, "per_rank": n // tp, "aligned_32": (n // tp) % 32 == 0}
    if ".experts." in name and "shared" not in name:
        if base.endswith(("w1", "w3")):
            return {"axis": "rows", "per_rank": 2304 // 2, "aligned_32": True}
        return {"axis": "cols(logical)", "per_rank": 2304 // 2, "aligned_32": (2304 // 2) % 32 == 0}
    if name in ("head.weight", "mtp.2.markov_head.head.weight"):
        return {"axis": "rows(vocab)", "per_rank": shape[0] // 2, "aligned_32": (shape[0] // 2) % 32 == 0}
    return None


# ----------------------------------------------------------------------------------------------- per shard


def _load(mm: np.memmap, off: int, info: dict) -> torch.Tensor:
    a, b = info["data_offsets"]
    raw = np.asarray(mm[off + a:off + b])
    dt = info["dtype"]
    t = torch.from_numpy(raw.copy())
    if dt == "F32":
        return t.view(torch.float32).view(info["shape"])
    if dt == "BF16":
        return t.view(torch.bfloat16).view(info["shape"])
    return t.view(torch.uint8).view(info["shape"])


def inspect_shard(args: tuple[str, str, list[str]]) -> list[dict]:
    root, fname, keys = args
    torch.set_num_threads(4)
    path = os.path.join(root, fname)
    h, off = read_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    recs = []
    keyset = set(keys)
    for name in keys:
        info = h[name]
        a, b = info["data_offsets"]
        raw = mm[off + a:off + b]
        sha = hashlib.sha256(memoryview(np.ascontiguousarray(raw))).hexdigest()
        nbytes = b - a
        exp = int(np.prod(info["shape"])) * DTYPE_BYTES[info["dtype"]] if info["shape"] else DTYPE_BYTES[info["dtype"]]
        rec = {"name": name, "file": fname, "dtype": info["dtype"], "shape": info["shape"],
               "byte_range": [off + a, off + b], "nbytes": nbytes, "nbytes_ok": nbytes == exp, "sha256": sha,
               "role": role_of(name), "supply": "shared-target" if name in SHARED else "draft-only"}
        dt = info["dtype"]
        if dt == "I8":
            rec["official"] = "mxfp4"
        elif dt == "F8_E4M3":
            rec["official"] = "fp8"
        elif dt == "F8_E8M0":
            rec["official"] = "e8m0"
        else:
            rec["official"] = dt.lower()
        t = _load(mm, off, info)
        if dt in ("F32", "BF16"):
            rec["values"] = stats_float(t)
        elif dt == "F8_E8M0":
            rec["values"] = scale_stats(t)
        else:
            sname = name[:-len("weight")] + "scale"
            if sname not in keyset:
                raise RuntimeError(f"{name}: scale {sname} not in this shard")
            rec["scale"] = sname
            rec["values"] = check_pair(t, _load(mm, off, h[sname]), rec["official"])
        cut = tp2_cuts(name, info["shape"])
        if cut:
            rec["tp2"] = cut
        recs.append(rec)
    return recs


# ----------------------------------------------------------------------------------------------- main


def summarize(recs: list[dict]) -> dict:
    by_fmt = collections.Counter()
    bytes_fmt = collections.Counter()
    exp_by_group: dict = collections.defaultdict(lambda: [10**9, -10**9])
    fp4_hist = collections.defaultdict(lambda: [0] * 16)
    bad = []
    for r in recs:
        by_fmt[r["official"]] += 1
        bytes_fmt[r["official"]] += r["nbytes"]
        v = r.get("values", {})
        if not r["nbytes_ok"]:
            bad.append((r["name"], "nbytes"))
        if v.get("nonfinite"):
            bad.append((r["name"], "nonfinite"))
        if v.get("nan") or v.get("nan_codes"):
            bad.append((r["name"], "nan"))
        if "shape_ok" in v and not v["shape_ok"]:
            bad.append((r["name"], "scale shape"))
        if r["official"] == "e8m0":
            g = re.sub(r"experts\.\d+\.", "experts.N.", re.sub(r"^mtp\.\d+\.", "mtp.S.", r["name"]))
            lo, hi = exp_by_group[g]
            exp_by_group[g] = [min(lo, v["exp_min"]), max(hi, v["exp_max"])]
        if r["official"] == "mxfp4":
            g = re.sub(r"experts\.\d+\.", "experts.N.", r["name"])
            fp4_hist[g] = [a + b for a, b in zip(fp4_hist[g], v["code_hist"])]
    dec = [r for r in recs if r["official"] in ("mxfp4", "fp8")]
    return {
        "count_by_format": dict(by_fmt), "bytes_by_format": dict(bytes_fmt), "problems": bad,
        "scale_exponent_range_by_group": {k: v for k, v in sorted(exp_by_group.items())},
        "fp4_code_hist_by_group": dict(fp4_hist),
        "decoded_all_exact_bf16": all(r["values"]["decoded"]["exact_bf16"] for r in dec),
        "decoded_all_exact_fp16": all(r["values"]["decoded"]["exact_fp16"] for r in dec),
        "decoded_not_exact_fp16": [r["name"] for r in dec if not r["values"]["decoded"]["exact_fp16"]][:50],
        "decoded_absmax": max(r["values"]["decoded"]["absmax"] for r in dec),
        "decoded_min_nonzero_abs": min(r["values"]["decoded"]["min_nonzero_abs"] for r in dec
                                       if r["values"]["decoded"]["min_nonzero_abs"] is not None),
        "router_bias_f32": all(r["dtype"] == "F32" for r in recs if r["name"].endswith((".gate.bias", ".gate.bias_vl"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tensors", required=True, help="the worker's TENSORS.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(a.root, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    keys = sorted(k for k in wmap if k.startswith("mtp.")) + list(SHARED)
    by_file: dict = collections.defaultdict(list)
    for k in keys:
        by_file[wmap[k]].append(k)
    missing = [fn for fn in by_file if not os.path.exists(os.path.join(a.root, fn))]
    if missing:
        raise SystemExit(f"official shards missing under {a.root}: {missing}")
    # one job per (file, chunk of keys): expert-heavy shards split so the pool stays busy
    jobs = []
    for fn, ks in sorted(by_file.items()):
        step = max(1, len(ks) // 8)
        for i in range(0, len(ks), step):
            chunk = ks[i:i + step]
            # a quantized weight and its scale must land in the same job
            jobs.append(chunk)
    fixed = []
    for chunk in jobs:
        fn = wmap[chunk[0]]
        need = set(chunk)
        for k in chunk:
            if k.endswith(".weight") and wmap.get(k[:-6] + "scale") == fn:
                need.add(k[:-6] + "scale")
        fixed.append((a.root, fn, sorted(need)))
    recs: dict = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for part in ex.map(inspect_shard, fixed):
            for r in part:
                recs[r["name"]] = r
    wk = worker_index(a.tensors)
    for name, r in recs.items():
        wv = worker_view(name, r["dtype"], wk)
        if wv["worker"] == "bytes":
            same = wv["worker_sha256"] == r["sha256"]
            wv["worker"] = "official-bytes" if same else "different-bytes"
            if name.endswith("wo_a.scale") or name.endswith("wo_a.weight"):
                wv["worker"] += " (decoded-at-load: FP8 kept, E8M0 -> fp32 scale)" if same else ""
        r.update(wv)
    receipt = open(os.path.join(a.root, "RECEIPT.txt")).read()
    out = {
        "schema": "n1-native-manifest/1", "official_repo": OFFICIAL_REPO, "official_revision": OFFICIAL_REV,
        "root": a.root, "receipt": receipt, "worker_tensors_json": os.path.abspath(a.tensors),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
        "n_tensors": len(recs), "summary": summarize(list(recs.values())),
        "worker_summary": dict(collections.Counter(r["worker"].split(" ")[0] for r in recs.values())),
        "tensors": [recs[k] for k in keys],
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=None, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({"n": out["n_tensors"], "seconds": out["seconds"], "summary": {k: v for k, v in out["summary"].items()
                      if k not in ("fp4_code_hist_by_group",)}, "worker": out["worker_summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
