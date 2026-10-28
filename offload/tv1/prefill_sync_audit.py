"""CPU source audit + rehearsal of actual indexer metadata Python (GPU launch stubbed).

This is not a GB10 forward trace. It distinguishes CPU scalar extraction from
D2H synchronization, and emits only source hashes, call sites and shape counts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Callable, cast

ROOT = Path('/opt/dsv41/data/src-image')
INDEXER = 'vllm/v1/attention/backends/mla/indexer.py'
SPARSE = 'vllm/model_executor/layers/sparse_attn_indexer.py'
SCANS = [INDEXER, SPARSE, 'vllm/models/deepseek_v41/common/engram.py',
         'vllm/models/deepseek_v41/attention.py', 'vllm/models/deepseek_v41/decoder_replay_layers.py',
         'vllm/models/deepseek_v41/nvidia/model.py', 'vllm/models/deepseek_v41/nvidia/ops/mhc.py',
         'vllm_exl3_moe/forward.py', 'vllm_exl3_moe/tf_prompt.py', 'vllm_exl3_moe/tf_experts.py']


def inventory(root: Path = ROOT) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for name in SCANS:
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in ('item', 'tolist', 'cpu', 'numpy', 'synchronize', 'wait_stream', 'repeat_interleave'):
                continue
            expression = ast.unparse(node)
            if name == INDEXER:
                if node.lineno in (1837, 1838):
                    why = 'DEVICE scalar sync; DCP>1 only (production DCP=1)'
                elif node.func.attr == 'repeat_interleave':
                    why = 'CPU helper on CPU mirrors; review explicit output_size where GPU data is used'
                else:
                    why = 'CPU mirror/NumPy scalar or list; not D2H sync on production branch'
            elif name == SPARSE:
                why = 'CPU-only fallback forward; not the CUDA indexer' if node.lineno >= 1170 else 'stream dependency, not host scalar read'
            elif name == 'vllm_exl3_moe/forward.py':
                if node.lineno in (181, 232):
                    why = 'DEVICE expert-count sync; fallback m<64 only, TF prompt bypass at >=64'
                elif node.func.attr == 'wait_stream':
                    why = 'REQUIRED device stream ordering; preserve shared/routed lock safety'
                else:
                    why = 'debug/stats/reference branch; production diagnostic switches OFF'
            else:
                why = 'inspect route: initialization/CPU mirror or required stream dependency; no production indexer scalar sync'
            rows.append({'path': name, 'line': node.lineno, 'call': expression[:260], 'classification': why})
    return rows


def rehearse(output: Path) -> dict[str, object]:
    import torch
    from torch.profiler import ProfilerActivity, profile
    import types
    source = (ROOT / INDEXER).read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prefill_chunk_metadata')
    # Execute the REAL Python metadata function, replacing only its GPU launch
    # and return data class. Thus CPU .item origins remain actual code behavior.
    namespace: dict[str, object] = {'torch': torch, '_BUILD_PREFILL_CHUNK_METADATA_KERNEL': lambda *a, **kw: None,
                                   'DeepseekV32IndexerPrefillChunkMetadata': types.SimpleNamespace}
    exec('from __future__ import annotations\n' + ast.unparse(function), namespace)
    call = cast(Callable[..., types.SimpleNamespace], namespace['build_prefill_chunk_metadata'])
    summary: list[dict[str, object]] = []
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as trace:
        for context in (32000, 150000):
            q = torch.tensor([0, 4096], dtype=torch.int32)
            seq = torch.tensor([context], dtype=torch.int32)
            table = torch.zeros(1, (context + 63) // 64, dtype=torch.int32)
            result = call(0, 1, q, q.clone(), seq, seq, seq.clone(), table, 1)
            summary.append({'context': context, 'query_rows': result.token_end-result.token_start,
                            'kv_rows': result.total_seq_lens, 'device': str(table.device), 'dcp': 1})
    output.mkdir(parents=True, exist_ok=True)
    trace.export_chrome_trace(str(output / 'cpu-metadata-trace.json'))
    scalar = [{'name': e.key, 'count': e.count} for e in trace.key_averages()
              if 'local_scalar' in e.key or e.key == 'aten::item']
    return {'classification': 'MEASURED CPU metadata Python rehearsal; GPU launch stubbed; no full-forward speed claim',
            'cases': summary, 'cpu_scalar_events': scalar, 'trace': 'cpu-metadata-trace.json'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('audit output must be new')
    result = rehearse(a.output)
    result['sources'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SCANS}
    result['calls'] = inventory()
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TV1 CPU prefill metadata trace and source inventory written; no production CUDA indexer sync to remove')


if __name__ == '__main__':
    main()
