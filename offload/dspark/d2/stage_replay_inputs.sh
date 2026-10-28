#!/usr/bin/env bash
# CPU only. Writes on x86host stay inside the assigned private D2 tree.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
umask 077
repo=$(git rev-parse --show-toplevel)
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/D2-REPLAY-PREP
data=/opt/dsv41/data/d2
remote=/opt/dsv41/x86host-data/d2
rev=$(git rev-parse HEAD)
ssh -o BatchMode=yes x86host "test ! -e '$remote/source/$rev'" || {
  echo "Refusing to replace an existing source snapshot: $remote/source/$rev" >&2; exit 1;
}
mkdir -p "$scratch/inputs/"{gen,prompts,encoder,seqs,acap,tokenizer}
chmod 700 "$scratch" "$scratch/inputs"
# The source generation state is read-only. Do not copy the live process LOCK.
cp -a "$data/gen/seqs/." "$scratch/inputs/seqs/"
cp -a "$data/acap/." "$scratch/inputs/acap/"
cp -a "$data/tokenizer/." "$scratch/inputs/tokenizer/"
cp "$data/prompts/manifest.json" "$scratch/inputs/prompts/manifest.json"
cp /opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py "$scratch/inputs/encoder/"
for file in budget.json events.jsonl rate.jsonl; do cp "$data/gen/$file" "$scratch/inputs/gen/$file"; done
find "$scratch/inputs" -type d -exec chmod 700 {} +
(cd "$scratch/inputs"; find . -type f ! -name INPUTS.SHA256 -print0 | sort -z | xargs -0 sha256sum > INPUTS.SHA256)
git -C "$repo" archive --format=tar HEAD offload/dspark > "$scratch/source.tar"
sha256sum "$scratch/source.tar" > "$scratch/source.sha256"
archive_sha=$(sha256sum "$scratch/source.tar" | cut -d ' ' -f 1)
ssh -o BatchMode=yes x86host "umask 077; mkdir -p '$remote/source/$rev'; chmod 700 '$remote' '$remote/source' '$remote/source/$rev'"
rsync -a --chmod=D700,F600 "$scratch/inputs/" "x86host:$remote/"
rsync -a --chmod=F600 "$scratch/source.tar" "x86host:$remote/source/$rev/source.tar"
ssh -o BatchMode=yes x86host "set -eu; umask 077; cd '$remote'; sha256sum -c INPUTS.SHA256 >/dev/null; cd source/$rev; echo '$archive_sha  source.tar' | sha256sum -c -; test ! -e repo; mkdir -m 700 repo; tar -xf source.tar -C repo; find repo -type d -exec chmod 700 {} +; printf '%s\n' '$rev' > SOURCE_COMMIT; sha256sum source.tar"
# V1 is an external reference, not a subdirectory of the deployment repository.
ssh -o BatchMode=yes x86host "set -euo pipefail; umask 077; if test ! -e '$remote/v1'; then mkdir -m 700 '$remote/v1'; cp -a \$HOME/dsv41-offload/d2-data/v1/verify '$remote/v1/'; find '$remote/v1' -type d -exec chmod 700 {} +; cd '$remote/v1'; find verify -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > V1.SHA256; fi; cd '$remote/v1'; echo '17109d736c217ea9c459be1f7851a2b7cfe057b762ae6b60ee86a7dc7f063684  V1.SHA256' | sha256sum -c -; sha256sum -c V1.SHA256 >/dev/null"
ssh -o BatchMode=yes x86host "set -eu; umask 077; export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; cd '$remote/source/$rev/repo/offload/dspark'; \$HOME/dsv41-offload/o2/venv/bin/python -m d2.replay plan --seqs '$remote/seqs' --acap '$remote/acap' --out '$remote/replay' > '$remote/plan-summary.json'; \$HOME/dsv41-offload/o2/venv/bin/python -m d2.replay estimate --seqs '$remote/seqs' > '$remote/estimate-generation.json'; cat '$remote/plan-summary.json'; cat '$remote/estimate-generation.json'; wc -c '$remote/replay/plan.jsonl'"
echo "Source archive matches $scratch/source.sha256. Inputs verified by INPUTS.SHA256."
