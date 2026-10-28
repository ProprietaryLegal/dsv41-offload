"""CPU-only qualification source lock builder; never imports engine modules."""
from __future__ import annotations

import argparse
import json
from typing import cast

from offload.prefill_maxlen.evidence import Json
from offload.up3 import build as b

RUNTIME = ('engine_common.py', 'engine_split.py', 'engine_vor.py', 'engine_k5.py', 'engine_ring.py',
    'engine_affinity.py', 'bootstrap.py', 'qualify.py', 'qualification_lock.py', 'stock_smoe.py.txt',
    'build.py', 'gates.py', 'inputs.json')
IMAGE_MODULES = ('vllm_exl3_moe.forward', 'vllm_exl3_moe.smoe', 'vllm_exl3_moe.rdma',
    'vllm.v1.attention.backends.mla.sparse_swa', 'vllm.models.deepseek_v41.nvidia.model_state')


def generate() -> dict[str, Json]:
    common = {m: b.digest((b.IMAGE/(m.replace('.', '/')+'.py')).read_bytes()) for m in IMAGE_MODULES}
    common.update({
        'vllm.v1.worker.gpu.model_runner': b.digest(b.production_file('offload/kvpack/overlay/vllm/v1/worker/gpu/model_runner.py')),
        'vllm.v1.worker.gpu.spec_decode.remote_dspark.speculator': b.digest(b.production_file('offload/dspark/spark/remote_dspark/speculator.py')),
        'vllm.models.deepseek_v41.common.engram': b.digest(b.production_file('offload/engram/spark/engram.py')),
    })
    engine: dict[str, Json] = {}
    for arm in b.ARMS[1:]:
        files = dict(common)
        for path in b.FILES.get(arm, ()):
            module = path[:-3].replace('/', '.')
            files[module] = b.digest((b.HERE/'overlay'/arm/path).read_bytes())
        engine[arm] = cast(Json, files)
    paths = ['offload/up3/'+p for p in RUNTIME]
    paths += ['offload/up3/manifests/'+arm+'.json' for arm in b.ARMS]
    return {'schema': 'up3-qualification-lock-v1', 'production': b.PRODUCTION,
        'files': {p: b.digest((b.ROOT/p).read_bytes()) for p in paths}, 'engine': engine}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    value = generate()
    target = b.HERE/'qualification.lock.json'
    if args.write:
        target.write_text(json.dumps(value, indent=2)+'\n')
    elif json.loads(target.read_text()) != value:
        raise ValueError('UP3 qualification source lock drift')
    print('UP3 qualification lock: '+('written' if args.write else 'verified')+'; CPU only')


if __name__ == '__main__':
    main()
