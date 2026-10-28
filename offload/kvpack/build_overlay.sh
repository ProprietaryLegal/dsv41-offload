#!/usr/bin/env bash
# K1: rebuild the KV-packing overlay from the image-v5 sources and the pinned donor patches, then compare with the
# committed overlay/ (or write it with --write). Reproducible; reads only:
#   SRC   = /opt/dsv41/data/src-image        (python tree of ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5)
#   DONOR = /opt/dsv41/data/upstream/henryous (github.com/HeNryous/deepseek-v41-1017-dgx-spark @ afe4d5e)
# Chain: 0051 -> 0053 -> 0057 -> 0059 (code only; VLLM_DSV41_KV_NVFP4 stays off) -> 0060 -> 0076. Every switch defaults
# to off. 0053's new vllm_exl3_moe/display_kv.py is not deployed (imported only with VLLM_DSV41_DISPLAY_KV_MIB set).
#   offload/kvpack/build_overlay.sh            # verify: rebuilt files == overlay/ and MANIFEST.json
#   offload/kvpack/build_overlay.sh --write    # (re)write overlay/ and MANIFEST.json
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-/opt/dsv41/data/src-image}
DONOR=${DONOR:-/opt/dsv41/data/upstream/henryous}
DONOR_SHA=afe4d5e41ec6fa4db17c5cc88758af439a106913
PATCHES=(0051-dsv41-indexer-defer 0053-dsv41-display-kv 0057-dsv41-kv-pair-blocks 0059-dsv41-kv-nvfp4
         0060-dsv41-kv-three-pages 0076-dsv41-head-memory)
declare -A PATCH_SHA=(
  [0051-dsv41-indexer-defer]=055677b5ad2667ce92aeff0af040f1929aedbffbbf1ed793a9c4e026f87a81c3
  [0053-dsv41-display-kv]=5a676d0ceddb705a90d595b431ebbc81881639372b3da7f6a23eb381abfbd8a9
  [0057-dsv41-kv-pair-blocks]=6058242fbd5909d50507baf4876082b4dedbff068795c97de3fa5f939fb9cb3b
  [0059-dsv41-kv-nvfp4]=9f7f8a87f013b0d862f2d7fa2c713908ca3790f93270ac879b34be2f263907d4
  [0060-dsv41-kv-three-pages]=52b5528f2aa3eb4060c3e427466e9859b980fcb8be4ab5dcb367332146f65772
  [0076-dsv41-head-memory]=bf3ab06aae624f3e9ed68bdd76b94a4369787b366014f2dcb2588351b7c2663f)
mode=${1:-verify}
[ "$(git -C "$DONOR" rev-parse HEAD)" = "$DONOR_SHA" ] || { echo "donor is not at $DONOR_SHA" >&2; exit 1; }
[ -z "$(git -C "$DONOR" status --porcelain)" ] || { echo "donor checkout is dirty" >&2; exit 1; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mkdir -p "$work/tree"
# copy only what the chain touches (plus the parents patch(1) needs)
while read -r f; do
  [ -n "$f" ] || continue
  if [ -e "$SRC/$f" ]; then mkdir -p "$work/tree/$(dirname "$f")"; cp "$SRC/$f" "$work/tree/$f"; fi
done < <(cat "$here/FILES"; echo vllm_exl3_moe/display_kv.py)
for p in "${PATCHES[@]}"; do
  pf="$DONOR/patches/$p.patch"
  got=$(sha256sum "$pf" | cut -d' ' -f1)
  [ "$got" = "${PATCH_SHA[$p]}" ] || { echo "$p: sha256 $got, expected ${PATCH_SHA[$p]}" >&2; exit 1; }
  (cd "$work/tree" && patch -p1 -N --no-backup-if-mismatch -s < "$pf") || { echo "$p does not apply" >&2; exit 1; }
done
[ -z "$(find "$work/tree" -name '*.rej' -o -name '*.orig')" ] || { echo "rejects left" >&2; exit 1; }
python3 - "$here" "$work/tree" "$SRC" "$mode" "$DONOR_SHA" "${PATCHES[@]}" <<'PY'
import hashlib, json, os, py_compile, shutil, sys
here, tree, src, mode, donor_sha, *patches = sys.argv[1:]
def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
files = [l.strip() for l in open(os.path.join(here, "FILES")) if l.strip()]
PREFIX = "/usr/local/lib/python3.12/dist-packages/"
entries = []
for f in files:
    built = os.path.join(tree, f)
    py_compile.compile(built, cfile=os.path.join(tree, "__compile_check.pyc"), doraise=True)
    base = os.path.join(src, f)
    entries.append({"path": f, "container_path": PREFIX + f,
                    "base_sha256": sha(base) if os.path.exists(base) else None,
                    "overlay_sha256": sha(built)})
manifest = {"donor": "github.com/HeNryous/deepseek-v41-1017-dgx-spark", "donor_sha": donor_sha,
            "patches": [{"name": p, "sha256": sha(os.path.join(os.environ.get("DONOR", "/opt/dsv41/data/upstream/henryous"), "patches", p + ".patch"))} for p in patches],
            "image": "ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5",
            "not_deployed": ["vllm_exl3_moe/display_kv.py (0053; only with VLLM_DSV41_DISPLAY_KV_MIB)"],
            "switches_default_off": {"VLLM_DSV41_KV_PAIR": "0 | 1 (0057, two pages) | 3 (0060, three pages)",
                                     "VLLM_DSV41_KV_SWA_FILL": "0 (no effect with FP8 records)",
                                     "VLLM_DSV41_KV_NVFP4": "0 (never enable: rejected at the KV-KLD gate)",
                                     "VLLM_DSV41_INDEXER_DEFER": "0 | 1 (0051)",
                                     "VLLM_DSV41_DROP_MTP_BUFFER": "0 | 1 (0076b)",
                                     "VLLM_EXL3_TF_PROMPT_TRANSIENT": "0 | 1 (0076d)",
                                     "VLLM_EXL3_TF_PROMPT_ALIAS": "0 | 1 (0076e, with 0076d)",
                                     "VLLM_DSV41_MEMPROBE": "'' | <prefix> (0076a, measurement windows only)"},
            "files": entries}
mpath = os.path.join(here, "MANIFEST.json")
if mode == "--write":
    for e in entries:
        dst = os.path.join(here, "overlay", e["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(tree, e["path"]), dst)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {len(entries)} overlay files and MANIFEST.json")
else:
    bad = [e["path"] for e in entries if sha(os.path.join(here, "overlay", e["path"])) != e["overlay_sha256"]]
    committed = json.load(open(mpath))
    if committed != manifest:
        bad.append("MANIFEST.json")
    if bad:
        raise SystemExit("rebuilt overlay differs from the committed one: " + ", ".join(bad))
    print(f"verified: {len(entries)} overlay files and MANIFEST.json match a fresh rebuild")
PY
