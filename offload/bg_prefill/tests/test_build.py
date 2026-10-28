"""Pin refusal and generated artifact/source composition checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from offload.bg_prefill.build_overlay import IMAGE, SCHEDULER, build, overlay_source, verify_inputs


def test_build_preserves_installed_0031_and_native_queue_guards(tmp_path: Path) -> None:
    receipt = build(tmp_path / 'overlay')
    source = (tmp_path / 'overlay' / SCHEDULER).read_text()
    assert source == overlay_source((IMAGE / SCHEDULER).read_text())
    assert 'request_queue = self.kv_holding_waiting or self.waiting' in source
    assert 'ready_to_schedule = self._handle_blocked_waiting_request(request)' in source
    assert 'throttle_prefills and not self.prefill_capacity_bound' in source
    assert 'r.priority < _DSV41_THROTTLE_MIN_PRIO' in source
    assert 'key=lambda r: (r.priority, r.arrival_time)' in source
    assert 'from so2_policy import' in source
    files = receipt['files']
    assert isinstance(files, dict) and len(files) == 2
    for name, sha in files.items():
        assert hashlib.sha256((tmp_path / 'overlay' / name).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match='destination must not exist'):
        build(tmp_path / 'overlay')


def test_reject_other_scheduler_mount_composition() -> None:
    source = (IMAGE / SCHEDULER).read_text()
    with pytest.raises(ValueError, match='pinned installed scheduler'):
        overlay_source(source + '\n# other scheduler overlay\n')
    assert len(verify_inputs()) == 11
