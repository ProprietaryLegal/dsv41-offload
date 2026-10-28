#!/usr/bin/env python3
"""Verify every file in the model dir against the HF tree (sha256 for LFS files, size for all) at the pinned revisions.
usage: verify_hashes.py <dir> [manifest-out]   (run on either node; exits 1 on any mismatch/missing)"""
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
REPOS = [("HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark", "86f852533ec94d1a537fea41fd9c01d6b0f6eb50", None),
         ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170",
          {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"})]
want = {}
for repo, rev, only in REPOS:
    t = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1", timeout=60))
    for e in t:
        if e["type"] != "file" or (only and e["path"] not in only):
            continue
        want[e["path"]] = {"repo": repo, "rev": rev, "size": e["size"], "sha256": (e.get("lfs") or {}).get("oid")}
def h(p):
    s = hashlib.sha256()
    with open(os.path.join(D, p), "rb") as f:
        while b := f.read(1 << 24): s.update(b)
    return s.hexdigest()
bad = []
def check(p):
    w = want[p]; fp = os.path.join(D, p)
    if not os.path.exists(fp): return p, "MISSING"
    if os.path.getsize(fp) != w["size"]: return p, f"SIZE {os.path.getsize(fp)} != {w['size']}"
    if w["sha256"] and h(p) != w["sha256"]: return p, "SHA256 MISMATCH"
    return p, "ok"
with ThreadPoolExecutor(6) as ex:
    for p, st in ex.map(check, sorted(want)):
        if st != "ok": bad.append((p, st)); print("BAD", p, st, flush=True)
extra = sorted(set(os.path.relpath(os.path.join(r, f), D) for r, _, fs in os.walk(D) if "/.cache" not in r + "/" for f in fs) - set(want))
print(f"{len(want)} expected files, {len(bad)} bad, extra files: {extra}")
if OUT: json.dump(want, open(OUT, "w"), indent=1)
sys.exit(1 if bad else 0)
