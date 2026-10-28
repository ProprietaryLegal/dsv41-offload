"""Private synthetic D2 shards: N1-only training; 50 captured keys eval-only.

Labels/probabilities and extra sequence rows are synthetic. This is plumbing
rehearsal, never G1 data or evidence of adaptation quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from safetensors.torch import save_file

from .data import D2Blocks, sha256
from .pilot import write_json


class Window(TypedDict):
    key: str
    inputs: dict[str, Tensor]
    picks: list[int]


def shard(root: Path, split: str, windows: list[Window], eval_inputs: dict[str, dict[str, str]]) -> dict[str, str]:
    columns: dict[str, list[Tensor]] = {n: [] for n in ("seq_index", "tokens", "taps", "row_pos", "top_ids", "top_lp",
                                                       "resid", "tok_lp", "tok_rank")}
    sequences = []
    k2 = []
    tok_off, row_off, span_off = 0, 0, 0
    for i, window in enumerate(windows):
        inp = window["inputs"]
        if (split == "train" and not bool(inp["has_taps"].all())) or not bool(inp["mask"].all()):
            raise ValueError("synthetic shard requires complete original taps")
        p = int(inp["P"][0])
        gen = p+1
        length = gen+(7 if split == "train" else 6)
        r0 = max(0, gen-128)
        if inp["positions"][0].tolist() != list(range(r0, p+1)):
            raise ValueError("fixture does not cover normative D2 context window")
        generator = torch.Generator().manual_seed(20261023+i)
        tokens = torch.randint(0, 129280, (length,), generator=generator, dtype=torch.int32)
        tokens[gen] = int(inp["anchor"][0])
        tokens[gen+1:gen+6] = torch.tensor(window["picks"], dtype=torch.int32)
        taps = torch.cat((inp["taps"][0], inp["taps"][0, -1:].expand(length-p-1, -1)), 0).contiguous()
        rows, spans = length-r0, length-gen
        ids = torch.arange(20, dtype=torch.int32)[None].expand(spans, -1).contiguous()
        lp = torch.full((spans, 20), .04).log()
        values = {"seq_index": torch.tensor([[tok_off, length, gen, r0, row_off, rows, span_off, spans]]),
                  "tokens": tokens, "taps": taps, "row_pos": torch.arange(r0, length, dtype=torch.int32),
                  "top_ids": ids, "top_lp": lp, "resid": torch.full((spans,), .2),
                  "tok_lp": torch.full((spans,), -10.), "tok_rank": torch.full((spans,), 100, dtype=torch.int32)}
        for name, value in values.items():
            columns[name].append(value)
        seq_id = f"synthetic-{split}-{i:03d}"
        if split != "train":
            # D2 shards are taps-only. Preserve the ORIGINAL mixed taps/main_x
            # window separately for this explicitly synthetic rehearsal only.
            exact = root/(seq_id+".pt")
            torch.save(inp, exact)
            exact.chmod(0o600)
            eval_inputs[seq_id] = {"file": exact.name, "sha256": sha256(exact)}
        sequences.append({"seq_id": seq_id, "source_docs": [seq_id], "family": "argument_section", "L": length,
                          "gen_start": gen, "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                          "synthetic_window_key": window["key"]})
        k2.append({"seq_id": seq_id, "drafts": {str(pos): (tokens[pos+2:pos+7].tolist()+[0]*5)[:5]
                                               for pos in range(gen-1, length-2)}})
        tok_off, row_off, span_off = tok_off+length, row_off+rows, span_off+spans
    path = root/(split+".safetensors")
    save_file({name: torch.cat(parts) for name, parts in columns.items()}, path,
              metadata={"schema": "d2.shard.v1", "split": split})
    side = root/(split+".json")
    write_json(side, {"schema": "d2.shard.v1", "split": split, "synthetic": True, "W": 128, "K": 5,
                      "topk": 20, "tap_layers": [37, 38, 39], "hidden": 5120, "noise_token": 128799,
                      "sha256": sha256(path), "sequences": sequences})
    with (root/(split+".k2_drafts.jsonl")).open("x") as f:
        for record in k2:
            f.write(json.dumps(record)+"\n")
    return {"file": path.name, "sha256": sha256(path), "sidecar": side.name, "sidecar_sha256": sha256(side)}


def build(inputs: Path, reference: Path, out: Path) -> None:
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    source = json.loads((inputs/"inputs.json").read_text())
    windows: dict[str, list[Window]] = {"n1": [], "captures": []}
    for name, expected in source["files"].items():
        path = inputs/name
        if sha256(path) != expected:
            raise ValueError("synthetic source fixture hash mismatch")
        fixture = torch.load(path, map_location="cpu", weights_only=False)
        base = torch.load(reference/name, map_location="cpu", weights_only=False)
        if base["inputs_sha256"] != expected:
            raise ValueError("synthetic reference window identity mismatch")
        windows[fixture["population"]].append({"key": fixture["key"], "inputs": fixture["inputs"], "picks": base["drafts"][0].tolist()})
    if len(windows["n1"]) != 4 or len(windows["captures"]) != 50:
        raise ValueError("require four N1 windows and exactly 50 captured keys")
    splits = {"train": [windows["n1"][i % 4] for i in range(34)],
              "acap": windows["captures"][:25], "heldout_legal": windows["captures"][25:]}
    manifest = out/"MANIFEST.json"
    eval_inputs: dict[str, dict[str, str]] = {}
    entries = {split: {"files": [shard(out, split, members, eval_inputs)]} for split, members in splits.items()}
    write_json(manifest, {"schema": "d2.manifest.v1", "synthetic": True,
                         "provenance": "N1-only train; exact 50 mixed captured input windows restored from eval-only sidecars; synthetic labels/top20/extensions; shard fallback tap slots are synthetic placeholders",
                         "inputs_sha256": sha256(inputs/"inputs.json"), "checks": {"plumbing_errors": 0},
                         "synthetic_eval_windows": eval_inputs, "splits": entries})
    for split in splits:
        dataset = D2Blocks(manifest, split)
        print(json.dumps({"split": split, "blocks": len(dataset), "synthetic": True}), flush=True)
    write_json(out/"synthetic-g1.json", {"synthetic": True, "pass": False, "note": "not a real G1 receipt"})


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    for name in ("inputs", "reference", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(8)
    build(a.inputs, a.reference, a.out)


if __name__ == "__main__":
    main()
