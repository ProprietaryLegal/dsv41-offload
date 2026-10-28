#!/usr/bin/env python3
"""Synthetic Engram shards for server tests: same safetensors layout as model-00047/48 (an F8_E4M3 [rows, 256]
weight + F8_E8M0 [rows, 8] scale for one layer, then a trailing non-Engram tensor), random bytes, plus a test
manifest with their size and sha256.   make_synth.py OUTDIR [--rows 2000003 --rows14 1500007]"""
import argparse
import hashlib
import json
import os
import struct

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=2000003)
ap.add_argument("--rows14", type=int, default=1500007)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
manifest = {}
for layer, rows, name in ((1, a.rows, "model-00047-of-00048.safetensors"), (14, a.rows14, "model-00048-of-00048.safetensors")):
    w, s, tail = rows * 256, rows * 8, 40960
    hdr = {
        f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, 256], "data_offsets": [0, w]},
        f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, 8], "data_offsets": [w, w + s]},
        f"layers.{layer}.engram.q_weight": {"dtype": "BF16", "shape": [4, 5120], "data_offsets": [w + s, w + s + tail]},
    }
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    if layer == 14:
        hb += b" " * 8  # a different header length than layer 1: offsets that are not 4 KiB aligned either way
    path = os.path.join(a.out, name)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for part in (struct.pack("<Q", len(hb)), hb):
            f.write(part)
            h.update(part)
        left = w + s + tail
        while left:
            blk = os.urandom(min(left, 8 << 20))
            f.write(blk)
            h.update(blk)
            left -= len(blk)
    manifest[name] = {"repo": "synthetic", "rev": "test", "size": os.path.getsize(path), "sha256": h.hexdigest()}
    print(name, rows, manifest[name]["size"], manifest[name]["sha256"])
json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
