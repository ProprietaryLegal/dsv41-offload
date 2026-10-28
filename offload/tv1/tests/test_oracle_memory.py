from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from offload.prefill_maxlen.evidence import Json
from offload.tv1 import oracle_memory as memory

# Public host-only fields from candidate-W145246Z cmd0153; no model data.
FAILED_BUDDY = '''Node 0, zone      DMA    279    327    453    446    416    291    206     85     21      2      3      3      3      6
Node 0, zone   Normal      0   1454  69970  50093    159    102     25      0      0      0      0      0      0      0
'''


def probe(page_size: int = 65536, *, failed: bool = False) -> dict[str, Json]:
    return {'type': 'tv1-oracle-memory-v1', 'page_size': page_size,
            'meminfo': 'MemAvailable: 9386836 kB\n',
            'buddyinfo': FAILED_BUDDY if failed else 'Node 0, zone Normal 0 0 0 0 0 0 0 1 0 0 0\n'}


@pytest.mark.parametrize('page_size', [4096, 16384, 65536])
def test_recorded_oom_is_refused_despite_eight_gib_and_plentiful_dma(page_size: int) -> None:
    result = memory.assess(probe(page_size, failed=True))
    assert result['mem_available_kb'] == 9386836 > memory.FLOOR_KB
    assert result['largest_free_normal_order'] == 6
    assert result['largest_free_normal_block_bytes'] == 64 * page_size
    assert result['cuda_allocation_guaranteed'] is False
    with pytest.raises(ValueError, match='pre-HOLD.*order=6'):
        memory.require(result)


def test_boundary_pass_is_a_physical_snapshot_not_cuda_guarantee() -> None:
    result = memory.assess(probe())
    assert result['largest_free_normal_block_bytes'] == 128 * 65536
    assert result['admitted'] is True
    assert result['cuda_allocation_guaranteed'] is False
    memory.require(result)


@pytest.mark.parametrize('change', ['low-memory', 'no-normal', 'node', 'duplicate', 'short',
                                    'page', 'bool', 'missing', 'negative', 'empty'])
def test_missing_invalid_or_insufficient_fields_cannot_admit(change: str) -> None:
    data = copy.deepcopy(probe())
    if change == 'low-memory':
        data['meminfo'] = f'MemAvailable: {memory.FLOOR_KB-1} kB'
    elif change == 'no-normal':
        data['buddyinfo'] = FAILED_BUDDY.splitlines()[0]
    elif change == 'node':
        data['buddyinfo'] = 'Node 1, zone Normal 0 0 0 0 0 0 0 900'
    elif change == 'duplicate':
        data['buddyinfo'] = str(data['buddyinfo']) * 2
    elif change == 'short':
        data['buddyinfo'] = 'Node 0, zone Normal 1 1'
    elif change == 'page':
        data['page_size'] = 123
    elif change == 'bool':
        data['page_size'] = True
    elif change == 'missing':
        del data['meminfo']
    elif change == 'negative':
        data['buddyinfo'] = 'Node 0, zone Normal -1 0 0 0 0 0 0 900'
    else:
        data['buddyinfo'] = 'Node 0, zone Normal 0 0 0 0 0 0 0 0 0'
    with pytest.raises(ValueError, match='TV1'):
        memory.require(memory.assess(data))


def test_emitted_probe_reads_only_host_memory_and_runs_on_cpu() -> None:
    code = memory.probe_script()
    assert 'torch' not in code and 'subprocess' not in code and 'write' not in code
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['type'] == 'tv1-oracle-memory-v1'
    assert data['page_size'] in (4096, 16384, 65536)
    assert data['meminfo'].startswith('MemAvailable:') and 'zone' in data['buddyinfo']
