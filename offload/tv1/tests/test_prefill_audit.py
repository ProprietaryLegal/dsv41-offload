from __future__ import annotations

from pathlib import Path
from typing import cast

from offload.tv1.prefill_sync_audit import INDEXER, SPARSE, inventory, rehearse


def test_cpu_scalar_origins_and_no_device_reads_in_cuda_indexer(tmp_path: Path) -> None:
    calls = inventory()
    sparse = [r for r in calls if r['path'] == SPARSE and int(r['line']) < 1170]
    assert not any('item(' in str(r['call']) or '.cpu(' in str(r['call']) for r in sparse)
    device_items = [r for r in calls if r['path'] == INDEXER and str(r['classification']).startswith('DEVICE')]
    assert [r['line'] for r in device_items] == [1837, 1838]
    result = rehearse(tmp_path / 'trace')
    assert [r['device'] for r in cast(list[dict[str, object]], result['cases'])] == ['cpu', 'cpu']
    assert (tmp_path / 'trace/cpu-metadata-trace.json').stat().st_size > 0
