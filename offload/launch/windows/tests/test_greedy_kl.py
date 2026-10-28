"""WIN2 greedy refinement: opt-in distribution recording in nbw_cells and the top-20 KL check (fake server, CPU only)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl as g
from offload.launch.windows import nbw_cells as nbw
from offload.prefill_maxlen.evidence import Json, record


def top(shift: float = 0.0, first: int = 100) -> dict[str, Json]:
    """Moves `shift` probability from the second token to the first; total stays a valid top-20 mass."""
    probs = [0.5 + shift, 0.3 - shift] + [0.2 / 18] * 18
    return {f'token_id:{first + i}': math.log(p) for i, p in enumerate(probs)}


class Api(wc.Http):
    def __init__(self, distribution: dict[str, Json]) -> None:
        super().__init__('http://unused.invalid')
        self.distribution = distribution
        self.bodies: list[tuple[str, dict[str, Json]]] = []

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        self.bodies.append((endpoint, body))
        if endpoint == '/tokenize':
            if 'messages' in body:
                return {'tokens': [1, 2]}
            return {'tokens': [7, 100 if str(body['prompt']).startswith('ref') else 101, 9]}
        return {'choices': [{'logprobs': {'top_logprobs': [self.distribution]}}]}


def test_identical_distributions_have_zero_kl_and_pass() -> None:
    r = g.top20_kl(top(), top())
    assert abs(r['kl_nats']) < 1e-12 and r['common_mass'] > .99 and g.passes(r, .01)


def test_shifted_distribution_raises_kl() -> None:
    r = g.top20_kl(top(), top(.2))
    assert r['kl_nats'] > .01 and not g.passes(r, .01)


def test_disjoint_tokens_fail_on_common_mass() -> None:
    r = g.top20_kl(top(first=100), top(first=500))
    assert r['common_mass'] == 0 and not g.passes(r, 10.0)


@pytest.mark.parametrize('bound', [0.0, -1.0, math.inf])
def test_bound_is_mandatory(bound: float) -> None:
    with pytest.raises(ValueError, match='registered KL bound'):
        g.passes(g.top20_kl(top(), top()), bound)


def test_near_tie_default_output_unchanged_and_opt_in_records() -> None:
    api = Api(top())
    plain = nbw.near_tie(api, 'q', 'ref', 'exp')
    assert set(plain) == {'token_index', 'near_tie', 'margin_nats', 'gap_to_top_nats'}
    assert 'cache_salt' not in api.bodies[-1][1]
    rich = nbw.near_tie(api, 'q', 'ref', 'exp', record_distribution=True, probe_salt='s:1')
    assert rich['probe_prefix_token_ids'] == [1, 2, 7] and rich['reference_token_id'] == 100
    assert record(rich['top20_logprobs']) == top() and api.bodies[-1][1]['cache_salt'] == 's:1'


def test_compare_recorded_reprobes_control_with_fresh_salt() -> None:
    candidate_row: dict[str, Json] = {'i': 2, 'identical': False, 'top20_logprobs': top(.2),
                                      'probe_prefix_token_ids': [1, 2, 7]}
    control = Api(top())
    rows = g.compare_recorded(control, [{'i': 0, 'identical': True}, candidate_row], 'win:s3', .01)
    assert len(rows) == 1 and rows[0]['pass'] is False and rows[0]['i'] == 2
    endpoint, body = control.bodies[-1]
    assert endpoint == '/v1/completions' and body['prompt'] == [1, 2, 7] and body['cache_salt'] == 'win:s3:kl:2'


def test_compare_recorded_requires_recorded_distribution() -> None:
    with pytest.raises(ValueError, match='record_distribution'):
        g.compare_recorded(Api(top()), [{'i': 1, 'identical': False}], 'x', .01)


def test_greedy_signature_defaults(tmp_path: Path) -> None:
    import inspect
    sig = inspect.signature(nbw.greedy)
    assert sig.parameters['near_tie_nats'].default == nbw.NEAR_TIE_NATS
    assert sig.parameters['record_distribution'].default is False and sig.parameters['probe_salt_prefix'].default is None


# ---------------------------------------------------------------- WIN2-R2 (R-NBW request): near-tie limit on the command line

@pytest.mark.parametrize('extra,expected', [
    ([], (nbw.NEAR_TIE_NATS, False, None)),
    (['--near-tie-nats', '1.101', '--record-distribution', '--probe-salt-prefix', 'W-x'], (1.101, True, 'W-x'))])
def test_cli_passes_the_registered_near_tie_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str],
                                                  expected: tuple[float, bool, str | None]) -> None:
    seen: list[tuple[float, bool, str | None]] = []

    def fake_greedy(api: wc.Http, deadline: float, prompts: list[str], reference: list[dict[str, Json]], output: Path,
                    near_tie_nats: float = nbw.NEAR_TIE_NATS, *, record_distribution: bool = False,
                    probe_salt_prefix: str | None = None) -> dict[str, Json]:
        seen.append((near_tie_nats, record_distribution, probe_salt_prefix))
        return {'checks': [], 'identical': 6, 'near_tie': 0}
    monkeypatch.setattr(nbw, 'greedy', fake_greedy)
    monkeypatch.setattr(nbw, 'greedy_inputs', lambda: (['p'] * 6, [{'i': i, 'text': 't'} for i in range(6)]))
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    out = tmp_path / 'greedy.json'
    assert nbw.main(['--execute', '--mode', 'greedy', '--output', str(out), '--deadline-utc', '9999999999', *extra]) == 0
    assert seen == [expected] and record(cast(Json, json.loads(out.read_text())))['identical'] == 6


@pytest.mark.parametrize('bad', ['0', '-1', 'nan', 'inf', '5'])
def test_cli_refuses_a_bad_near_tie_limit(bad: str) -> None:
    with pytest.raises(SystemExit):
        nbw.main(['--execute', '--mode', 'greedy', '--output', '/nonexistent/x', '--near-tie-nats', bad])
