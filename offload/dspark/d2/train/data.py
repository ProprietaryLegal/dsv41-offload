"""D2-DATA d2.manifest.v1 reader and small synthetic internal block fixtures."""
from __future__ import annotations

import hashlib
import json
import bisect
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from torch import Tensor
from safetensors import safe_open

from .model import Config
from .synthetic import inputs

INPUT_NAMES = ("taps", "engine_mx", "has_taps", "positions", "mask", "anchor", "P")
TARGET_NAMES = ("previous", "top_ids", "top_logp", "tail_mass", "labels", "present", "position_mask")


class Block(TypedDict):
    inputs: dict[str, Tensor]
    targets: dict[str, Tensor]
    request: str


class Blocks(Protocol):
    fingerprint: str
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Block: ...


def collate(blocks: list[Block]) -> Block:
    if not blocks:
        raise ValueError("empty block batch")
    width = max(b["inputs"]["positions"].shape[1] for b in blocks)
    batches: dict[str, list[Tensor]] = {n: [] for n in INPUT_NAMES}
    for block in blocks:
        inp = block["inputs"]
        pad = width-inp["positions"].shape[1]
        for name in INPUT_NAMES:
            t = inp[name]
            if pad and name not in ("anchor", "P"):
                if name == "positions":
                    before = torch.arange(int(t[0, 0])-pad, int(t[0, 0]), dtype=t.dtype)[None]
                else:
                    before = torch.zeros(1, pad, *t.shape[2:], dtype=t.dtype)
                t = torch.cat((before, t), 1)
            batches[name].append(t)
    return {"inputs": {n: torch.cat(v, 0) for n, v in batches.items()},
            "targets": {n: torch.cat([b["targets"][n] for b in blocks], 0) for n in TARGET_NAMES},
            "request": "batch"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_families(split: str, units: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    """Union requests sharing source documents (or a source-less prompt identity).

    Run after reading ALL shard sidecars of a split: a later request can bridge
    two earlier components. Hash sorted component units so traversal order and
    task labels cannot change the resampling key. Never expose source IDs.
    """
    parent = {request: request for request in units}

    def find(request: str) -> str:
        while parent[request] != request:
            parent[request] = parent[parent[request]]
            request = parent[request]
        return request

    owners: dict[tuple[str, str], str] = {}
    for request, sources in units.items():
        for unit in sources:
            other = owners.setdefault(unit, request)
            a, b = find(request), find(other)
            parent[max(a, b)] = min(a, b)
    components: dict[str, set[tuple[str, str]]] = {}
    for request, sources in units.items():
        components.setdefault(find(request), set()).update(sources)
    identities = {root: hashlib.sha256(json.dumps([split, sorted(sources)], separators=(",", ":")).encode()).hexdigest()
                  for root, sources in components.items()}
    return {request: identities[find(request)] for request in units}


class SyntheticBlocks:
    def __init__(self, c: Config, count: int = 8):
        self.c, self.count = c, count
        self.fingerprint = f"synthetic-v1-{c}-{count}"

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Block:
        inp = inputs(self.c, batch=1, P=12+index)
        g = torch.Generator().manual_seed(111+index)
        full = torch.randn(1, 5, self.c.vocab, generator=g).softmax(-1)
        p, ids = full.topk(20, -1)
        labels = torch.multinomial(full.flatten(0, 1), 1, generator=g).reshape(1, 5)
        previous = torch.cat((inp["anchor"][:, None], labels[:, :-1]), -1)
        target = {"previous": previous, "top_ids": ids, "top_logp": p.log(), "tail_mass": 1-p.sum(-1),
                  "labels": labels, "present": torch.ones(1, 5, dtype=torch.bool),
                  "position_mask": torch.ones(1, 5, dtype=torch.bool)}
        return {"inputs": inp, "targets": target, "request": f"synthetic-{index}"}


class D2Blocks:
    """Memory-map one bounded D2 shard at a time; hash all split metadata first.

    Blocks follow DATASET.md §4 exactly: query anchor P+1, targets P+2..P+6,
    and teacher-forced Markov input P+1..P+5. Tail blocks are position masked.
    """
    def __init__(self, manifest: Path, split: str):
        self.root = manifest.parent.resolve()
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get("schema") != "d2.manifest.v1":
            raise ValueError("expected D2-DATA d2.manifest.v1")
        checks = self.manifest.get("checks", {})
        if "plumbing_errors" in checks and checks["plumbing_errors"] != 0:
            raise ValueError("dataset records plumbing errors")
        self.sequences: list[tuple[Path, list[int], str]] = []
        self.request_families: dict[str, str] = {}
        self.request_tasks: dict[str, str] = {}
        self.sequence_b: list[int] = []  # V1 counts even sequences with zero eligible blocks
        self.ends: list[int] = []
        seen: dict[str, str] = {}
        docs: dict[str, str] = {}
        total = 0
        for split_name, spec in self.manifest["splits"].items():
            split_b = 0
            units: dict[str, list[tuple[str, str]]] = {}
            for entry in spec["files"]:
                path = self.verified(entry["file"], entry["sha256"])
                side = self.verified(entry["sidecar"], entry["sidecar_sha256"])
                meta = json.loads(side.read_text())
                constants = {"schema": "d2.shard.v1", "split": split_name, "W": 128, "K": 5, "topk": 20,
                             "tap_layers": [37, 38, 39], "hidden": 5120, "noise_token": 128799}
                if any(meta.get(k) != v for k, v in constants.items()) or meta["sha256"] != entry["sha256"]:
                    raise ValueError(f"D2 shard sidecar mismatch {side}")
                with safe_open(path, framework="pt", device="cpu") as f:
                    if f.metadata() != {"schema": "d2.shard.v1", "split": split_name}:
                        raise ValueError(f"D2 safetensors metadata mismatch {path}")
                    index = f.get_tensor("seq_index")
                    if index.dtype != torch.int64 or index.shape != (len(meta["sequences"]), 8):
                        raise ValueError(f"D2 sequence index mismatch {path}")
                    expected = (0, 0, 0)
                    for j, seq in enumerate(meta["sequences"]):
                        req = seq["seq_id"]
                        if req in seen:
                            raise ValueError("duplicate request across shards/splits")
                        seen[req] = split_name
                        if not isinstance(seq.get("family"), str) or not seq["family"]:
                            raise ValueError("D2 request missing task label")
                        self.request_tasks[req] = seq["family"]
                        source_docs = seq.get("source_docs")
                        if not isinstance(source_docs, list) or any(not isinstance(d, str) or not d for d in source_docs):
                            raise ValueError("D2 request invalid source documents")
                        if source_docs:
                            units[req] = [("document", doc) for doc in source_docs]
                        else:
                            prompt = seq.get("prompt_sha256")
                            if not isinstance(prompt, str) or not prompt:
                                raise ValueError("D2 source-less request missing prompt identity")
                            units[req] = [("prompt", prompt)]
                        for doc in source_docs:
                            if doc in docs and docs[doc] != split_name:
                                raise ValueError("source-document split leakage")
                            docs[doc] = split_name
                        if split_name in ("train", "dev") and seq["family"] in ("chronology", "motion_outline"):
                            raise ValueError("held-out family in train/dev")
                        row = index[j].tolist()
                        tok_off, L, gen, r0, row_off, nr, span_off, ns = row
                        if not 1 <= gen < L or (L, gen) != (seq["L"], seq["gen_start"]) or r0 != max(0, gen-128) or nr != L-r0 or ns != L-gen:
                            raise ValueError("D2 sequence offset/length mismatch")
                        if (tok_off, row_off, span_off) != expected:
                            raise ValueError("D2 sequence offsets are not cumulative")
                        tokens = f.get_slice("tokens")[tok_off:tok_off+L]
                        if tokens.dtype != torch.int32 or tokens.shape != (L,) or bool(((tokens < 0) | (tokens >= 129280)).any()):
                            raise ValueError("D2 token shape/dtype/vocabulary mismatch")
                        digest = hashlib.sha256(tokens.contiguous().numpy().astype("<i4", copy=False).tobytes()).hexdigest()
                        if digest != seq.get("tokens_sha256"):
                            raise ValueError("D2 per-sequence token hash differs")
                        rp = f.get_slice("row_pos")[row_off:row_off+nr]
                        if rp.dtype != torch.int32 or not torch.equal(rp, torch.arange(r0, L, dtype=torch.int32)):
                            raise ValueError("D2 row positions differ from sequence positions")
                        expected = tok_off+L, row_off+nr, span_off+ns
                        if split_name == split and L-gen > 1:
                            total += L-gen-1
                            self.ends.append(total)
                            self.sequences.append((path, row, req))
                            self.sequence_b.append(split_b)
                        split_b += 1
                    shapes = [f.get_slice(n).get_shape()[0] for n in ("tokens", "row_pos", "top_lp")]
                    if tuple(shapes) != expected or f.get_slice("taps").get_shape() != [expected[1], 15360]:
                        raise ValueError("D2 aggregate tensor lengths differ from sequence offsets")
            self.request_families.update(source_families(split_name, units))
        if not total:
            raise ValueError(f"empty D2 block split {split}")
        self.fingerprint = hashlib.sha256((sha256(manifest)+split).encode()).hexdigest()

    def verified(self, relative: str, expected: str) -> Path:
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root) or sha256(path) != expected:
            raise ValueError(f"D2 file hash/path mismatch: {relative}")
        return path

    def __len__(self) -> int:
        return self.ends[-1]

    def supervised_tokens(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        seq_i = bisect.bisect_right(self.ends, index)
        _, row, _ = self.sequences[seq_i]
        _, length, gen = row[:3]
        offset = index-(self.ends[seq_i-1] if seq_i else 0)
        return min(5, length-(gen-1+offset)-2)

    def __getitem__(self, index: int) -> Block:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        seq_i = bisect.bisect_right(self.ends, index)
        path, row, req = self.sequences[seq_i]
        tok_off, L, gen, r0, row_off, nr, span_off, ns = row
        offset = index-(self.ends[seq_i-1] if seq_i else 0)
        P = gen-1+offset
        first = max(0, P-127)
        positions = torch.arange(first, P+1)
        with safe_open(path, framework="pt", device="cpu") as f:
            tokens = f.get_slice("tokens")[tok_off:tok_off+L].long()
            lo, hi = row_off+first-r0, row_off+P-r0+1
            taps = f.get_slice("taps")[lo:hi]
            actual_pos = f.get_slice("row_pos")[lo:hi].long()
            if taps.dtype != torch.bfloat16 or taps.shape != (positions.numel(), 15360) or not torch.equal(positions, actual_pos):
                raise ValueError("tap shape/dtype/absolute positions differ")
            target_pos = torch.arange(P+2, P+7)
            mask = target_pos < L
            n = int(mask.sum())
            start = span_off+P+2-gen
            ids = torch.full((5, 20), -1, dtype=torch.long)
            lp = torch.full((5, 20), -torch.inf)
            resid = torch.ones(5)
            ids[:n] = f.get_slice("top_ids")[start:start+n].long()
            lp[:n] = f.get_slice("top_lp")[start:start+n]
            resid[:n] = f.get_slice("resid")[start:start+n]
        labels = torch.full((5,), -1, dtype=torch.long)
        previous = torch.zeros(5, dtype=torch.long)
        labels[:n] = tokens[P+2:P+2+n]
        previous[:n] = tokens[P+1:P+1+n]
        return {"inputs": {"taps": taps[None], "engine_mx": torch.zeros(1, positions.numel(), 5120),
                           "has_taps": torch.ones(1, positions.numel(), dtype=torch.bool), "positions": positions[None],
                           "mask": torch.ones(1, positions.numel(), dtype=torch.bool), "anchor": tokens[P+1:P+2],
                           "P": torch.tensor([P])},
                "targets": {"previous": previous[None], "top_ids": ids[None], "top_logp": lp[None],
                            "tail_mass": resid[None], "labels": labels[None], "present": mask[None],
                            "position_mask": mask[None]}, "request": req}
