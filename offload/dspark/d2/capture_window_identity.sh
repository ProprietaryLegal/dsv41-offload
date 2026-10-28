#!/usr/bin/env bash
# Coordinator-only read probe, invoked on EACH Spark by replay_window.py.
# No writes, requests, model imports or service changes. Output stays private.
set -euo pipefail
python3 - <<'PY'
import json, subprocess
d=json.loads(subprocess.check_output(['docker','inspect','dsv41-serve']))[0]
print(json.dumps({'schema':'d2.window-container.v1','id':d['Id'],'image':d['Image'],
 'started':d['State']['StartedAt'],'state':d['State']['Status'],
 'mounts':d['Mounts'],'cmd':d['Config']['Cmd'],
 'env':[e for e in d['Config']['Env'] if e.startswith(('VLLM_','MODEL_','PORT='))]},sort_keys=True))
PY
d2_mounts=$(docker inspect --format '{{json .Mounts}}' dsv41-serve)
docker exec -i dsv41-serve python3 - "$d2_mounts" <<'PY'
import hashlib, importlib.metadata, json, pathlib, subprocess, sys
base=pathlib.Path('/usr/local/lib/python3.12/dist-packages')
model=pathlib.Path('/srv/dsv41/legal-v2a')
paths=[model/n for n in ('CANDIDATE_MANIFEST.json','exl3_moe_plan.json','tokenizer.json','tokenizer_config.json','config.json')]
paths += [base/n for n in ('vllm/tokenizers/deepseek_v41_encoding.py','vllm/sampling_params.py',
 'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py','vllm/entrypoints/openai/chat_completion/protocol.py',
 'vllm/v1/core/sched/scheduler.py','vllm/v1/core/sched/async_scheduler.py')]
required=paths[:8]
for p in required:
 if not p.is_file(): raise SystemExit('missing required identity file: '+str(p))
def digest(p):
 # O_DIRECT even for metadata: never fill Spark page cache while checking identity.
 proc=subprocess.Popen(['dd','if='+str(p),'iflag=direct','bs=4096','status=none'],stdout=subprocess.PIPE)
 try:
  result=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
 finally:
  proc.stdout.close()
  code=proc.wait()
 if code: raise SystemExit('O_DIRECT identity read failed: '+str(p))
 return result
mounted={}
for mount in json.loads(sys.argv[1]):
 p=pathlib.Path(mount['Destination'])
 # Model tensors and caches are excluded. Exact target metadata is hashed above.
 if not str(p).startswith(('/usr/local/lib/python3.12/dist-packages/','/opt/dsv41-ofl','/serve.sh')): continue
 if p.is_file(): mounted[str(p)]=digest(p)
 elif p.is_dir():
  for f in sorted(p.rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.so','.json','.sh'):
    mounted[str(f)]=digest(f)
g=model/'generation_config.json'
print(json.dumps({'schema':'d2.window-files.v1','engine':importlib.metadata.version('vllm'),
 'files':{str(p):digest(p) if p.is_file() else None for p in paths},
 'mounted_code_sha256':mounted,
 'generation_config':{'exists':g.exists(),'sha256':digest(g) if g.is_file() else None}},sort_keys=True))
PY
