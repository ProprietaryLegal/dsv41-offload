"""Every UP3 test runs with a private WIN2 chain marker, no inherited chain environment (never the campaign marker) and a
recording stand-in for chain.abort, whose real form opens a scoped HOLD and runs W8 on the Sparks."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.prefill_maxlen.evidence import Json

def recorded_abort(out: Path, **kwargs: object) -> dict[str, Json]:
    """What a successful chain.abort returns (RESTORED + watch PASS, marker cleared); like the real one it creates `out`."""
    out.mkdir(parents=True, exist_ok=False)
    chain.clear()
    return {'status': 'RESTORED: chain ended by abort; watch PASS; marker cleared', 'chain': 'ABORTED',
            'restore': {'status': chain.ABORT_RESTORED, 'start': 'PASS', 'check': 'PASS', 'close': 'PASS'},
            'post_close_watch': {'status': 'PASS'}}


@pytest.fixture(autouse=True)
def isolated_chain(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(chain, 'MARKER', tmp_path_factory.mktemp('chain') / 'CHAINED')
    abort: Callable[..., dict[str, Json]] = recorded_abort
    monkeypatch.setattr(chain, 'abort', abort)
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    yield
