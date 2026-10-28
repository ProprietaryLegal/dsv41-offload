"""PP4 emitted-dyadic24-q heldout replay. X86host solo only; no serving traffic.

The production source is imported read-only. No server, network client, model
download, trained adapter or service control is present. Outputs are aggregates.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from offload.dspark.worker.sampled import SEMANTIC, Top32
from offload.dspark.worker.sampled_fused import FusedContext

from . import acceptance_frozen as frozen
from .acceptance import ARM, score_sparse

UUID = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module_digest(name: str) -> str:
    path = importlib.import_module(name).__file__
    if path is None:
        raise ValueError(f"module {name} lacks a file identity")
    return digest(Path(path))


def write(path: Path, value: object) -> None:
    temp = path.with_suffix(".new")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def guard() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != UUID or os.sched_getaffinity(0) != {12, 13, 14, 15, 28, 29, 30, 31}:
        raise RuntimeError("requires exact solo UUID and reserved affinity")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("nice19 required")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 48 * 1024**2:
        raise RuntimeError("48GiB resident-memory budget exceeded")


def require_lock() -> None:
    lock = Path.home() / "dsv41-offload/locks/gpu-bc62df2a.lock"
    fd = int(os.environ["KX_FP8DENSE_LOCK_FD"])
    inherited, actual = os.fstat(fd), lock.stat()
    if (inherited.st_ino, inherited.st_dev) != (actual.st_ino, actual.st_dev):
        raise RuntimeError("inherited fd is not the solo-card lock")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    raise RuntimeError("exclusive solo flock was not inherited")


def full_logits(m: Any, base: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for rk in m.ranks:
        prev = previous.contiguous()
        bias = rk.markov.bias(prev, rk.markov.embed(prev))
        # The deployed fused selector rounds base+bias to BF16 before argmax.
        parts.append((base[:, rk.v0:rk.v1] + bias).to(torch.bfloat16))
    return torch.cat(parts, -1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--frozen-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="smoke only; final requires zero")
    a = p.parse_args()
    if a.stride != 128:
        raise ValueError("frozen anchor stride is 128")
    guard()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU required")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.removeprefix("GPU-") != UUID.removeprefix("GPU-"):
        raise RuntimeError("CUDA device UUID differs")
    sys.path.insert(0, str(a.production_root / "offload/dspark"))
    sys.path.insert(1, str(a.production_root))
    checkpoint = importlib.import_module("worker.checkpoint")
    tp2 = importlib.import_module("worker.tp2")
    # Production's historical benchmark helper admits its old lane-E card only.
    # PP2's explicit solo allocation is enforced here without modifying it.
    require_lock()
    print("Verifying production tensor payloads", flush=True)
    ck = checkpoint.Checkpoint(str(a.weights), threads=8)
    identity = importlib.import_module("worker.identity").compute(str(a.weights), manifest=ck.manifest)
    if (identity["drafter_identity"] != "be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd"
            or identity["markov_identity"] != "0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a"):
        raise ValueError("production K2/Markov identities differ from frozen receipt")
    dev = torch.device("cuda:0")
    print("Loading two logical production ranks onto the solo UUID", flush=True)
    m = tp2.TPDrafter(ck, [dev, dev], slots=1, max_rows=32, max_model_len=1048576,
                      markov_cache_rows=8192, fuse_pairs=False)
    manifest = json.loads(a.manifest.read_text())
    print("Production model loaded; preparing frozen capture anchors", flush=True)
    if digest(a.manifest) != "450bb71653f3def68cd95ee9579a456c74c469d08a016940cceda24de48b4f30":
        raise ValueError("capture manifest identity changed")
    selection = frozen.value(a.frozen_dir, "dev-selection.json")
    if "k32-t0.7" not in selection["selected"]:
        raise ValueError("frozen top32 policy differs")
    numeric = frozen.module(a.frozen_dir, "numeric")
    summary_module = frozen.module(a.frozen_dir, "summarize")
    baseline = frozen.value(a.frozen_dir, "heldout-replay.json")
    sampler = Top32(1, ck.shape.vocab, dev)
    fused = FusedContext(m, 1, 5)
    keys = torch.tensor([[4102, 0, 0]], device=dev, dtype=torch.int64)
    temperatures = torch.tensor([.7], device=dev, dtype=torch.float32)
    q_hash = hashlib.sha256()
    reference_q_different_rows = 0
    groups: dict[str, Any] = {}
    parity_rounds = 0
    rounds = 0
    start = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "pp4.acceptance-replay.v1", "classification": "MEASURED offline K2 TP2 solo emulation",
        "split": "heldout_legal", "complete": False, "smoke_limit": a.limit, "stride": a.stride,
        "manifest_sha256": digest(a.manifest), "selection_sha256": frozen.HASHES["dev-selection.json"],
        "identity": identity, "weights_tensor_payload_verified": True,
        "production_root": str(a.production_root.resolve()),
        "source_sha256": {str(path.relative_to(a.production_root)): digest(path)
                          for path in sorted((a.production_root / "offload/dspark/worker").rglob("*"))
                          if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".json")},
        "extension_sha256": {name: module_digest(name)
                             for name in ("exllamav3_ext", "exllamav3_ext_facc")},
        "replay_sha256": digest(Path(__file__)), "numeric_sha256": frozen.HASHES["numeric.py"],
        "scoring_sha256": digest(Path(__file__).with_name("acceptance.py")),
        "producer_sha256": module_digest("offload.dspark.worker.sampled"),
        "fused_producer_sha256": module_digest("offload.dspark.worker.sampled_fused"),
        "emitted_q_source": "FusedContext.select on real production base logits and forced target-history previous IDs",
        "semantic": SEMANTIC, "frozen_commit": frozen.COMMIT, "frozen_sha256": frozen.HASHES,
        "torch": torch.__version__, "python": sys.version, "uuid": UUID, "threads": torch.get_num_threads(),
        "python_executable": sys.executable, "python_executable_sha256": digest(Path(sys.executable)),
        "affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "arms": [ARM, "greedy"], "metrics": numeric.METRICS,
        "anchor_population": "gen_start-1 + 128*j; P+6<L; no future-token or gain selection",
        "groups": groups, "verified_shards": [],
    }
    units: dict[str, list[tuple[str, str]]] = {}
    # Source-connected components use all split requests, not only scored rows.
    for entry in manifest["splits"]["heldout_legal"]["files"]:
        sidepath = a.manifest.parent / entry["sidecar"]
        if digest(sidepath) != entry["sidecar_sha256"]:
            raise ValueError("sidecar digest mismatch")
        for seq in json.loads(sidepath.read_text())["sequences"]:
            units[seq["seq_id"]] = ([("document", d) for d in seq["source_docs"]]
                                     or [("prompt", seq["prompt_sha256"])])
    components = summary_module.source_components("heldout_legal", units)
    with torch.inference_mode():
        for entry in manifest["splits"]["heldout_legal"]["files"]:
            path = a.manifest.parent / entry["file"]
            print("Verifying capture shard " + entry["sha256"], flush=True)
            if digest(path) != entry["sha256"]:
                raise ValueError("shard payload digest mismatch")
            receipt["verified_shards"].append(entry["sha256"])
            side = json.loads((a.manifest.parent / entry["sidecar"]).read_text())
            with safe_open(path, framework="pt", device="cpu") as f:
                index = f.get_tensor("seq_index")
                for i, seq in enumerate(side["sequences"]):
                    tok_off, length, gen, r0, row_off, nr, span_off, ns = index[i].tolist()
                    if (length, gen, nr, ns) != (seq["L"], seq["gen_start"], length-r0, length-gen):
                        raise ValueError("sequence metadata mismatch")
                    tokens = f.get_slice("tokens")[tok_off:tok_off+length].long()
                    token_hash = hashlib.sha256(tokens.to(torch.int32).numpy().tobytes()).hexdigest()
                    if token_hash != seq["tokens_sha256"]:
                        raise ValueError("token history mismatch")
                    positions = f.get_slice("row_pos")[row_off:row_off+nr]
                    if not torch.equal(positions, torch.arange(r0, length, dtype=torch.int32)):
                        raise ValueError("absolute positions differ")
                    for pos in range(gen-1, length-6, a.stride):
                        guard()
                        lo = max(r0, pos-127)
                        taps = f.get_slice("taps")[row_off+lo-r0:row_off+pos-r0+1].to(dev)
                        if not bool(torch.isfinite(taps).all()):
                            raise ValueError("nonfinite taps")
                        m.reset_slots([0])
                        for offset in range(0, pos-lo+1, 8):
                            stop = min(offset+8, pos-lo+1)
                            m.insert_eager(torch.zeros(stop-offset, dtype=torch.int64),
                                           torch.arange(lo+offset, lo+stop), taps[offset:stop])
                        inp = torch.tensor([[0], [int(tokens[pos+1])], [pos], [lo]])
                        trace: dict[str, Any] = {}
                        greedy, _, _, rank1 = m.draft(inp, 5, trace=trace)
                        if not torch.equal(greedy, rank1):
                            raise ValueError("emulated ranks disagree")
                        base = trace["base_logits"][0]
                        prev = tokens[pos+1:pos+6].to(dev)
                        logits = full_logits(m, base, prev)
                        # Validate materialized full logits against unchanged fused selector,
                        # including ALL five positions on its own greedy-conditioned path.
                        own_prev = torch.cat((tokens[pos+1:pos+2].to(dev), greedy[0, :-1]))
                        own_logits = full_logits(m, base, own_prev)
                        if not torch.equal(own_logits.argmax(-1), greedy[0]):
                            raise ValueError("full-logit vs fused production Markov argmax parity failure")
                        parity_rounds += 1
                        offset = span_off + pos+2-gen
                        target_ids = f.get_slice("top_ids")[offset:offset+5].to(dev)
                        top_lp = f.get_slice("top_lp")[offset:offset+5].to(dev)
                        residual = f.get_slice("resid")[offset:offset+5].to(dev)
                        tok_lp = f.get_slice("tok_lp")[offset:offset+5].to(dev)
                        keys[0, 1], keys[0, 2] = rounds, pos
                        fused.update(keys, torch.ones(1, dtype=torch.bool, device=dev))
                        shard_logits = {rank: base[None, :, rk.v0:rk.v1].contiguous()
                                        for rank, rk in enumerate(m.ranks)}
                        outputs = {rank: torch.empty((1, 5), dtype=torch.int64, device=dev) for rank in m.ALL}
                        reference_ids, reference_q = [], []
                        for position in range(5):
                            previous = {rank: prev[position:position+1].contiguous() for rank in m.ALL}
                            fused.select(m.ALL, shard_logits, previous, outputs, position)
                            reference_keys = keys.clone()
                            reference_keys[0, 2] += position
                            token, support, q = sampler(logits[position:position+1], reference_keys, temperatures)
                            if bool(sampler.error.any()):
                                raise ValueError("producer refused frozen logits")
                            if not bool(((support == token[:, None]) & (q > 0)).any()):
                                raise ValueError("draw outside emitted positive q")
                            reference_ids.append(support.clone())
                            reference_q.append(q.clone())
                        emitted_ids, emitted_q = fused.result()
                        sparse_ids, sparse_q = emitted_ids[0], emitted_q[0]
                        if not torch.equal(outputs[0], outputs[1]):
                            raise ValueError("fused emulated ranks disagree")
                        if not bool(((sparse_ids == outputs[0].flatten()[:, None]) & (sparse_q > 0)).any(-1).all()):
                            raise ValueError("fused draw outside emitted positive q")
                        identical = ((sparse_ids == torch.cat(reference_ids)).all(-1)
                                     & (sparse_q == torch.cat(reference_q)).all(-1))
                        reference_q_different_rows += int((~identical).sum().item())
                        # Only a hash leaves transient memory; no captured tensors are saved.
                        q_hash.update(sparse_ids.cpu().numpy().tobytes())
                        q_hash.update(sparse_q.cpu().numpy().tobytes())
                        values = score_sparse(sparse_ids, sparse_q, logits.argmax(-1), target_ids, top_lp, residual,
                                              tokens[pos+2:pos+7].to(dev), tok_lp,
                                              vocab=ck.shape.vocab, numeric=numeric).cpu()
                        context = "<8K" if pos < 8000 else "8-16K" if pos < 16000 else "16-32K" if pos < 32000 else ">=32K"
                        key = hashlib.sha256(seq["seq_id"].encode()).hexdigest() + ":" + context
                        group = groups.setdefault(key, {"request_sha256": key.split(":")[0],
                            "source_cluster_sha256": components[seq["seq_id"]], "family": seq["family"],
                            "context": context, "rounds": 0, "sum": torch.zeros_like(values).tolist()})
                        group["rounds"] += 1
                        group["sum"] = (torch.tensor(group["sum"], dtype=torch.float64) + values).tolist()
                        rounds += 1
                        if rounds % 25 == 0:
                            receipt.update(rounds=rounds, parity_rounds=parity_rounds, seconds=time.monotonic()-start,
                                           maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                           gpu_peak_bytes=torch.cuda.max_memory_allocated(), emitted_q_sha256=q_hash.hexdigest(),
                                           reference_q_different_rows=reference_q_different_rows)
                            write(a.out, receipt)
                            print(json.dumps({"rounds": rounds, "seconds": time.monotonic()-start}), flush=True)
                        if a.limit and rounds >= a.limit:
                            break
                    write(a.out, receipt)
                    if a.limit and rounds >= a.limit:
                        break
            if a.limit and rounds >= a.limit:
                break
    receipt.update(complete=True, rounds=rounds, parity_rounds=parity_rounds, seconds=time.monotonic()-start,
                   maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   gpu_peak_bytes=torch.cuda.max_memory_allocated(), emitted_q_sha256=q_hash.hexdigest(),
                   reference_q_different_rows=reference_q_different_rows)
    if not a.limit and (rounds != baseline["rounds"] or set(groups) != set(baseline["groups"])):
        raise ValueError("frozen heldout anchor population differs")
    write(a.out, receipt)
    print(json.dumps({k: receipt[k] for k in ("complete", "rounds", "seconds", "maxrss_kib", "gpu_peak_bytes")}), flush=True)


if __name__ == "__main__":
    main()
