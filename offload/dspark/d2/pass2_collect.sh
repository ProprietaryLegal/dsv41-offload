#!/usr/bin/env bash
# Coordinator-only metadata collection. Never run this from a lane.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
if [[ "${1:-}" != --execute ]]; then
    if [[ $# -ne 0 ]]; then echo 'usage: pass2_collect.sh [--execute NEW_PRIVATE_CAMPAIGN_DIRECTORY]' >&2; exit 2; fi
    cat <<'PLAN'
DRY: no network access or output writes (local paths configuration is sourced).
Coordinator only: --execute NEW_PRIVATE_CAMPAIGN_DIRECTORY
Checks /metrics idle before and after every operation; aborts on busy/missing metrics.
Reads docker inspect dsv41-serve on sparkhead and sparkworker-wired; GET /v1/models;
fixed config/manifest/tokenizer/renderer/version/protocol metadata on both ranks.
No weights, /tokenize, inference requests, generator state, writes on remote hosts,
GPU imports, or service changes. Files over 1 MiB use dd iflag=direct only.
Raw outputs require semantic review; collection alone proves no effective defaults.
PLAN
    exit 0
fi
[[ $# -eq 2 && "${D2_COORDINATOR:-}" == Deepseek/coordinator ]] || {
    echo 'refused: --execute requires D2_COORDINATOR=Deepseek/coordinator and a NEW output directory' >&2; exit 2;
}
: "${HARNESS_PYTHON:?source paths.sh must set HARNESS_PYTHON}"
exec "$HARNESS_PYTHON" - "$2" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from urllib.request import urlopen

os.umask(0o077)
base = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
out = Path(sys.argv[1])
if (not out.is_absolute() or '..' in out.parts or out.exists()
        or out.parent.resolve() != out.parent or not out.is_relative_to(base)
        or any(p in ('W12', 'scratch') for p in out.relative_to(base).parts)):
    raise SystemExit('refused: NEW normalized private campaign directory outside W12/scratch required')
# Fixed endpoint, no caller-supplied URLs/commands/paths.
api = 'http://HEAD_IP:8000'

def get(path, limit=2 * 1024 * 1024):
    with urlopen(api + path, timeout=5) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError('metadata response exceeded bound')
    return data

def idle():
    data = get('/metrics').decode()
    for metric in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + metric + r'(?:\{[^\n]*\})?\s+(\S+)\s*$', data, re.M)
        if not values or any(not math.isfinite(float(v)) or float(v) != 0 for v in values):
            raise RuntimeError('refused: production not provably idle: ' + metric)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'running': 0, 'waiting': 0}

# No SSH traffic until the first idle check passes.
first = idle()
out.mkdir(mode=0o700)  # parent must already exist; never reuse/overwrite
records = []
def collect(label, action):
    before = idle()
    data = action()
    after = idle()  # busy means no successful collection receipt
    path = out / label
    path.write_bytes(data)
    records.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                    'before': before, 'after': after})

remote = r'''
import hashlib, json, pathlib, subprocess
root = pathlib.Path('/srv/dsv41/legal-v2a')
pkg = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
files = [root / n for n in ('CANDIDATE_MANIFEST.json', 'exl3_moe_plan.json',
    'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')]
files += [pkg / n for n in ('_version.py', 'version.py', 'tokenizers/deepseek_v41_encoding.py',
    'entrypoints/openai/completion/protocol.py', 'entrypoints/openai/completion/serving.py',
    'entrypoints/serve/tokenize/serving.py', 'tokenizers/deepseek_v41.py',
    'renderers/online_renderer.py', 'sampling_params.py', 'config/model.py')]
rows = []
for p in files:
    if not p.exists():
        rows.append({'path': str(p), 'status': 'ABSENT'})
        continue
    size = p.stat().st_size
    if not p.is_file() or size > 32 * 1024 * 1024:
        raise RuntimeError('non-small allowlisted metadata refused: ' + str(p))
    if size > 1024 * 1024:
        # No buffered read of large metadata on production. No file content copied.
        proc = subprocess.Popen(['dd', 'if=' + str(p), 'iflag=direct', 'bs=1M', 'status=none'], stdout=subprocess.PIPE)
        h = hashlib.sha256()
        while chunk := proc.stdout.read(1024 * 1024):
            h.update(chunk)
        if proc.wait() != 0:
            raise RuntimeError('O_DIRECT metadata hash failed: ' + str(p))
        row = {'path': str(p), 'bytes': size, 'sha256': h.hexdigest(), 'read': 'O_DIRECT'}
    else:
        data = p.read_bytes()
        row = {'path': str(p), 'bytes': size, 'sha256': hashlib.sha256(data).hexdigest(),
               'read': 'small-buffered', 'text': data.decode('utf-8')}
    if p.stat().st_size != size:
        raise RuntimeError('metadata size changed during collection')
    rows.append(row)
print(json.dumps({'schema': 'd2.pass2-raw-files.v1', 'files': rows}))
'''

def ssh(host, args, data=None):
    # Fixed hosts and fixed commands only; -T never addresses tmux.
    result = subprocess.run(['ssh', '-T', '-oBatchMode=yes', '-oConnectTimeout=5', host, *args],
                            input=data, stdout=subprocess.PIPE, check=True, timeout=45)
    if len(result.stdout) > 4 * 1024 * 1024:
        raise RuntimeError('remote metadata output exceeded bound')
    return result.stdout

collect('models.json', lambda: get('/v1/models'))
for host in ('sparkhead', 'sparkworker-wired'):
    collect(host + '-inspect.json', lambda h=host: ssh(h, ['docker', 'inspect', 'dsv41-serve']))
    collect(host + '-files.json', lambda h=host: ssh(h, ['docker', 'exec', '-i', 'dsv41-serve', 'python3', '-'], remote.encode()))
receipt = {'schema': 'd2.pass2-collection.v1', 'coordinator': 'Deepseek/coordinator',
           'status': 'RAW_REQUIRES_REVIEW', 'initial_idle': first, 'files': records,
           'limitations': ['snapshot idle checks cannot exclude arrivals between checks',
                          'no model/tokenize requests; renderer coverage remains coordinator evidence',
                          'manifest digests do not verify weights bytes',
                          'launch/config/source precedence needs semantic review; no engine import']}
(out / 'COLLECTED.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(str(out / 'COLLECTED.json'))
PY
