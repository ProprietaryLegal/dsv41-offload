"""WIN2-R3 smoke start: shared sequence, verdicts, always-stop, chain-head and x-first modes (fake backend, CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.launch.windows import smoke_start as sm
from offload.prefill_maxlen.evidence import Json, record


class Fake:
    def __init__(self, fail: dict[str, str] | None = None, *, rounds: float = 30.0, tokens: int = sm.SMOKE_TOKENS) -> None:
        self.fail = fail or {}
        self.rounds, self.tokens = rounds, tokens
        self.drafts = 0.0
        self.calls: list[tuple[str, str]] = []

    def _maybe(self, stage: str, arm: str) -> None:
        self.calls.append((stage, arm))
        if self.fail.get(arm) == stage:
            raise RuntimeError(f'{stage} failed for {arm}')

    def start(self, arm: str) -> None:
        self._maybe('start', arm)

    def healthy(self, arm: str) -> dict[str, Json]:
        self._maybe('healthy', arm)
        return {'id': 'fixture', 'max_model_len': 327680}

    def greedy(self, arm: str) -> dict[str, Json]:
        self._maybe('greedy', arm)
        return {'sha256': 'a' * 64, 'tokens': 300}

    def seeded(self, arm: str, body: dict[str, Json]) -> dict[str, Json]:
        self._maybe('seeded', arm)
        assert body['seed'] == sm.SMOKE_SEED and body['max_tokens'] == sm.SMOKE_TOKENS and body['ignore_eos'] is True
        self.drafts += self.rounds
        return {'usage': {'completion_tokens': self.tokens}, 'output_sha256': 'b' * 64}

    def metrics(self, arm: str) -> str:
        return f'vllm:spec_decode_num_drafts_total{{m="x"}} {self.drafts}\n'

    def stop(self, arm: str) -> None:
        self._maybe('stop', arm)


def test_pass_runs_every_stage_and_stops() -> None:
    fake = Fake()
    r = sm.smoke(fake, 'X')
    assert r['status'] == 'PASS' and r['stage'] == 'done' and record(r['seeded'])['rounds'] == 30.0
    assert [c[0] for c in fake.calls] == ['start', 'healthy', 'greedy', 'seeded', 'stop']


@pytest.mark.parametrize('stage', ['start', 'healthy', 'greedy', 'seeded'])
def test_any_stage_failure_is_fail_and_still_stops(stage: str) -> None:
    fake = Fake({'X': stage})
    r = sm.smoke(fake, 'X')
    assert r['status'] == 'FAIL' and r['stage'] == stage and stage in str(r['error'])
    assert fake.calls[-1] == ('stop', 'X')


@pytest.mark.parametrize('fake', [Fake(rounds=0.0), Fake(tokens=12)])
def test_seeded_request_must_run_speculation_to_length(fake: Fake) -> None:
    r = sm.smoke(fake, 'X')
    assert r['status'] == 'FAIL' and r['stage'] == 'seeded'


def test_stop_failure_raises_with_the_result() -> None:
    with pytest.raises(RuntimeError, match='arm stop failed; HOLD stays'):
        sm.smoke(Fake({'X': 'stop'}), 'X')


def test_budget_exhaustion_fails_and_stops() -> None:
    times = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0])
    fake = Fake()
    r = sm.smoke(fake, 'X', clock=lambda: next(times), budget_s=10.0)
    assert r['status'] == 'FAIL' and 'budget' in str(r['error']) and fake.calls[-1] == ('stop', 'X')


def test_chain_head_drops_failing_candidates() -> None:
    fake = Fake({'B': 'healthy'})
    r = sm.chain_head(fake, ['A', 'B', 'C'])
    assert r['passed'] == ['A', 'C'] and r['dropped'] == ['B']
    assert [c for c in fake.calls if c[0] == 'stop'] == [('stop', 'A'), ('stop', 'B'), ('stop', 'C')]
    with pytest.raises(ValueError, match='distinct'):
        sm.chain_head(fake, ['A', 'A'])


def test_x_first_ends_the_window_on_failure() -> None:
    assert sm.x_first(Fake(), 'X')['window'] == 'CONTINUE'
    assert str(sm.x_first(Fake({'X': 'greedy'}), 'X')['window']).startswith('END')


def factory(experiment: str, arms: list[str], out: Path) -> Fake:
    return Fake({'bad': 'start'})


def test_cli_dry_plan_and_coordinator_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    assert sm.main(['PC1', 'chunk16384']) == 0 and 'DRY RUN' in capsys.readouterr().out
    out = tmp_path / 'smoke'
    args = ['PC1', 'chunk16384', 'bad', '--mode', 'chain-head', '--execute', '--out', str(out),
            '--backend', f'{__name__}:factory']
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert sm.main(args) == 2
    monkeypatch.setenv('WINDOW_COORDINATOR', sm.COORDINATOR)
    assert sm.main(args) == 0
    saved = json.loads((out / 'smoke.json').read_text())
    assert saved['passed'] == ['chunk16384'] and saved['dropped'] == ['bad']
    with pytest.raises(SystemExit):
        sm.main(['PC1', 'a', 'b'])                         # x-first smokes exactly one arm
