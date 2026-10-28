"""Every UP3 test runs with a private WIN2 chain marker and no inherited chain environment (never the campaign marker).

WIN2-R2: chain.abort is replaced by the real abort logic over fake HOLD probe / W8 recovery / watch, and chain's own
host probes refuse, so no CPU test can reach sparkhead, sparkworker-wired or production."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.prefill_maxlen.evidence import Json


@pytest.fixture(autouse=True)
def isolated_chain(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(chain, 'MARKER', tmp_path_factory.mktemp('chain') / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    yield


class AbortRig:
    """Records every chain.abort; `hold` / `recover` / `watch` are the fakes the real chain.abort runs with."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.hold = False
        self.recover: Callable[[Path, Callable[[str], None]], Mapping[str, Json]] = (
            lambda out, log: {'start': 'PASS', 'check': 'PASS', 'close': 'PASS'})
        self.watch: Callable[[Path, Callable[[str], None]], Mapping[str, Json]] = lambda out, log: {'status': 'PASS'}


@pytest.fixture(autouse=True)
def chain_abort(monkeypatch: pytest.MonkeyPatch) -> AbortRig:
    rig = AbortRig()
    real = chain.abort

    def fake(out: Path) -> dict[str, Json]:
        rig.calls.append(out)
        return real(out, hold=lambda: rig.hold, recover=rig.recover, watch=rig.watch, log=lambda message: None)

    def refuse() -> bool:
        raise AssertionError('chain host probe from a CPU test')

    monkeypatch.setattr(chain, 'abort', fake)
    for name in ('hold_exists', 'placeholders_present', 'production_serving'):
        monkeypatch.setattr(chain, name, refuse)
    return rig
