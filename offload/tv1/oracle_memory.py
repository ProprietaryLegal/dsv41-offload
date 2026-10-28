"""Read-only pre-HOLD fragmentation screen, never a CUDA allocation guarantee.

Buddyinfo gives the largest already-free physical buddy block. CUDA can map many
physical blocks into one virtual allocation; its largest allocatable span cannot
be measured safely beside production. Order7 is a conservative admission policy
chosen to reject the observed failure, NOT a proven driver allocation granule.
The isolated prestart oracle performs the definitive CUDA allocation afterwards.
"""
from __future__ import annotations

import re
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

FLOOR_KB = 8 * 1024**2
MIN_NORMAL_ORDER = 7


def probe_script() -> str:
    """No torch, GPU, allocation, compaction, service or cache operation."""
    return '''import json,os,pathlib
memory=pathlib.Path('/proc/meminfo').read_text()
print(json.dumps({'type':'tv1-oracle-memory-v1','page_size':os.sysconf('SC_PAGE_SIZE'),
 'meminfo':'\\n'.join(x for x in memory.splitlines() if x.startswith('MemAvailable:')),
 'buddyinfo':pathlib.Path('/proc/buddyinfo').read_text()}))
'''


def assess(value: object) -> dict[str, Json]:
    data = record(cast(Json, value))
    page_size = data.get('page_size')
    if (data.get('type') != 'tv1-oracle-memory-v1' or type(page_size) is not int
            or page_size not in (4096, 16384, 65536)):
        raise ValueError('TV1 missing/unsupported oracle memory page geometry')
    memory, buddy = data.get('meminfo'), data.get('buddyinfo')
    if not isinstance(memory, str) or not isinstance(buddy, str):
        raise ValueError('TV1 oracle memory probe missing source text')
    matches = re.findall(r'^MemAvailable:\s+(\d+) kB$', memory, re.M)
    normal = re.findall(r'^Node\s+(\d+),\s+zone\s+Normal\s+([\d \t]+)$', buddy, re.M)
    # GB10's current single-NUMA topology. Never let plentiful DMA or another
    # node hide the exhausted Normal zone of the observed allocation failure.
    if len(matches) != 1 or len(normal) != 1 or normal[0][0] != '0':
        raise ValueError('TV1 oracle memory requires one MemAvailable and node0 Normal zone')
    counts = [int(n) for n in normal[0][1].split()]
    if len(counts) <= MIN_NORMAL_ORDER:
        raise ValueError('TV1 oracle buddy orders incomplete')
    largest_order = max((i for i, n in enumerate(counts) if n), default=-1)
    largest_bytes = page_size * 2**largest_order if largest_order >= 0 else 0
    available = int(matches[0])
    admitted = available >= FLOOR_KB and largest_order >= MIN_NORMAL_ORDER
    return {'type': 'tv1-oracle-memory-assessment-v1',
            'classification': 'MEASURED physical buddy snapshot; CUDA allocatability UNQUALIFIED',
            'mem_available_kb': available, 'floor_kb': FLOOR_KB, 'page_size': page_size,
            'normal_node': 0, 'normal_buddy_counts': cast(Json, counts),
            'largest_free_normal_order': largest_order,
            'largest_free_normal_block_bytes': largest_bytes,
            'minimum_normal_order': MIN_NORMAL_ORDER,
            'minimum_normal_block_bytes': page_size * 2**MIN_NORMAL_ORDER,
            'cuda_allocation_guaranteed': False, 'admitted': admitted}


def require(assessment: dict[str, Json]) -> None:
    if assessment.get('admitted') is not True:
        raise ValueError('TV1 pre-HOLD oracle memory refusal: '
                         f'MemAvailable={assessment.get("mem_available_kb")}kB '
                         f'(need {FLOOR_KB}); largest free Normal physical block='
                         f'{assessment.get("largest_free_normal_block_bytes")} bytes '
                         f'order={assessment.get("largest_free_normal_order")} '
                         f'(need order>={MIN_NORMAL_ORDER}); '
                         'do not spend HOLD or assume CUDA allocatability from total free memory')
