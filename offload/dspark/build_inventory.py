"""Build the DSpark transfer inventory from inventory_source.py's JSON output."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from copy_to_x86host import TensorRow


def build(source_path: Path, output: Path) -> None:
    source = json.loads(source_path.read_text())
    plan = source["metadata"]["exl3_moe_plan.json"]
    tensors: list[TensorRow] = []
    tables: list[TensorRow] = []
    for filename, file in source["files"].items():
        for name, value in file["header"].items():
            if name == "__metadata__":
                continue
            needed = name.startswith(("mtp.", "head.")) or name in ("embed.weight", "norm.weight")
            engram = ".engram." in name and "0004" in filename and "of-00048" in filename
            if not (needed or engram):
                continue
            start, end = value["data_offsets"]
            absolute = [8 + file["header_bytes"] + start, 8 + file["header_bytes"] + end]
            exl3 = name.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1")
            k = None
            if exl3:
                k = 6 if name.startswith("head.") else 2 if ".experts." in name else 5
            row: TensorRow = {
                "name": name, "shape": value["shape"], "dtype": value["dtype"],
                "representation": "EXL3" if exl3 else "FP8/ue8m0" if value["dtype"].startswith("F8") or name.endswith(".scale") else value["dtype"],
                "exl3_k": k, "source_file": filename, "source_byte_range": absolute,
                "resident_packed_bytes": end-start,
                "source_file_sha256": source["manifest"][filename]["sha256"],
                "baseline_identical": file["baseline_same_inode"],
                "identity_evidence": {key: file[key] for key in ("device", "inode", "nlink", "baseline_same_inode")},
                "provenance": source["manifest"][filename],
            }
            (tensors if needed else tables).append(row)
    result = {
        "schema_version": 1, "source_root": source["source_root"],
        "baseline_root": source["baseline_root"], "local_only": True,
        "codebook": plan["codebook"], "drafter_plan": plan["drafter"],
        "resident_packed_bytes": sum(t["resident_packed_bytes"] for t in tensors),
        "runtime_footprint_note": "Packed tensor payload only; conversion, padding, KV, scratch, graphs and communication pools must be budgeted separately by O2.",
        "tensors": tensors, "engram_tensors": tables,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "TENSORS.json").write_text(json.dumps(result, indent=2)+"\n")
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for t in tensors:
        counts[t["source_file"]][0] += 1
        counts[t["source_file"]][1] += t["resident_packed_bytes"]
    lines = ["# Exact DSpark tensor inventory", "",
        "Source: sparkhead `/srv/dsv41/legal-v2a`; every safetensors header inspected with O_DIRECT. No tensor execution or conversion.", "",
        "Baseline: HeNryous revision `86f852533ec94d1a537fea41fd9c01d6b0f6eb50`. All selected files share device+inode with `/srv/dsv41/model`, proving byte identity; file SHA256 expectations are from the deployment manifest. Engram originates at DeepSeek revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.", "",
        "`TENSORS.json` lists every selected tensor's name, shape, dtype, EXL3 K, source absolute half-open byte range, packed resident bytes, baseline file SHA256 and hardlink evidence. After transfer it also records SHA256 of every source byte range and destination file/range. These artifacts are local only.", "",
        "| Source file | Tensors | Packed bytes |", "|---|---:|---:|"]
    lines += [f"| {f} | {n} | {size} |" for f, (n, size) in counts.items()]
    lines += ["", f"Total: {len(tensors)} tensors, {result['resident_packed_bytes']:,} bytes ({result['resident_packed_bytes']/2**30:.3f} GiB).", "",
        "Dependencies: `mtp.0.main_proj.*` / `mtp.0.main_norm.weight`; all `mtp.{0,1,2}` attention, MoE router+experts, shared expert and mHC tensors; `mtp.2.norm.weight`, `mtp.2.markov_head.{embed,head}.weight`, `mtp.2.confidence_head.proj.weight`. `embed.weight` and EXL3 `head.*` are the target aliases. `norm.weight` is included as an explicit target dependency for loader inspection; the drafter final norm is `mtp.2.norm.weight`.", "",
        "EXL3 representations: routed drafter experts K2, dense/attention/shared/main projection K5, target LM head K6; each includes trellis I16, suh/svh F16 and mul1 I32. Codebook: `"+str(plan['codebook'])+"` (algorithmic runtime codebook, no separate checkpoint tensor). Remaining `attn.wo_a` FP8 and ue8m0 scales are preserved. K values follow the deployed config and drafter plan, not target expert K allocations.", "",
        "Packed bytes are not a GPU fit claim: O2 must add dequantized runtime representations, padding, caches, scratch, CUDA graphs and bounded communication pools. The live Markov token cache is copied separately and compared byte-for-byte between both Sparks.", "",
        "## Engram shard entries", "", "| Tensor | Shape | Dtype | File | Absolute byte range |", "|---|---|---|---|---|"]
    lines += [f"| {t['name']} | {t['shape']} | {t['dtype']} | {t['source_file']} | {t['source_byte_range']} |" for t in tables]
    (output / "TENSORS.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
