"""Round 14 near-tie contract (NDM 1.1-nat ceiling) and retained Round 12 fresh-cache diagnostics."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pp4_cells as c
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

ROOT = Path(__file__).resolve().parents[4]
PRIMARY = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/PP4-429-admission-W-20261027T133630Z/admission-B-greedy.json')
PRIMARY_SHA = 'ace208cbb3135015577c163417f4504da9ab95ebefd8910e97d2dd7c78ef93eb'
REGISTRATION = 'pp4-r14-near-tie-1p1'


def top20(margin: float, step: float) -> dict[str, Json]:
    """Supplemental CPU fixture, NOT recovered historical logprobs."""
    return {'token_id:200': -.25, 'token_id:100': -.25-margin,
            **{'token_id:'+str(300+j): -.25-(j+2)*step for j in range(18)}}


class Api(wc.Http):
    def __init__(self, cases: list[tuple[int, float, float]]) -> None:
        super().__init__('http://unused.invalid')
        self.cases = cases
        self.chats = 0
        self.calls: list[tuple[str, dict[str, Json]]] = []
        self.override: dict[str, Json] | None = None
        self.length_change = False
        self.prefix_only = False
        self.shorter = False

    def metrics(self) -> str:
        return 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        self.calls.append((endpoint, body))
        if endpoint in ('/v1/chat/completions', '/v1/completions'):
            assert isinstance(body['cache_salt'], str) and str(body['cache_salt']).strip()
        if endpoint == '/v1/chat/completions':
            i = self.chats
            self.chats += 1
            text = f'public-experiment-{i}' if i < len(self.cases) else f'public-reference-{i}'
            return {'choices': [{'message': {'content': text}}]}
        i = self.chats-1
        index, margin, step = self.cases[i]
        if endpoint == '/tokenize':
            if 'messages' in body:
                return {'tokens': [1, 2, 3]}
            tokens = [10]*120
            tokens[index] = 100
            if str(body['prompt']).startswith('public-experiment'):
                tokens[index] = 200
                if self.length_change:
                    tokens.append(7)
                if self.prefix_only:
                    tokens[index] = 100
                if self.shorter:
                    tokens = tokens[:100]
            return {'tokens': cast(Json, tokens)}
        assert endpoint == '/v1/completions'
        assert body['prompt'] == [1, 2, 3, *([10]*index)]
        assert body['temperature'] == 0 and body['max_tokens'] == 1 and body['logprobs'] == 20
        assert body['return_tokens_as_token_ids'] is True
        return {'choices': [{'logprobs': {'top_logprobs': [
            self.override if self.override is not None else top20(margin, step)]}}]}


def run(api: Api, output: Path) -> dict[str, Json]:
    return c.greedy(api, 1e12, ['public-fixture']*6,
                    [{'i': i, 'text': f'public-reference-{i}'} for i in range(6)], output)


def test_real_admission_margins_pass_under_one_point_one_nat_ceiling(tmp_path: Path) -> None:
    raw = PRIMARY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PRIMARY_SHA
    historical = ck.rows(record(cast(Json, json.loads(raw)))['checks'])
    assert [r['token_index'] for r in historical] == [97, 78, 63]
    assert [r['margin_nats'] for r in historical] == [0., .5, 1.]
    assert [r['gap_to_top_nats'] for r in historical] == [0., 0., 0.]
    assert all('top20_logprobs' not in r for r in historical)
    cases = [(int(number(r['token_index'])), number(r['margin_nats']), .5 if i < 2 else 1.)
             for i, r in enumerate(historical)]
    api = Api(cases)
    output = tmp_path/'real-margins.json'
    result = run(api, output)
    assert result['greedy_registration'] == REGISTRATION and result['near_tie'] == 3 and result['identical'] == 3
    rows = ck.rows(ck.read(output)['checks'])
    assert [row['near_tie'] for row in rows[:3]] == [True, True, True]
    assert all(row.get('identical') is True for row in rows[3:])
    for i, row in enumerate(rows[:3]):
        assert row['token_index'] == historical[i]['token_index']
        assert row['margin_nats'] == historical[i]['margin_nats']
        assert row['gap_to_top_nats'] == 0
        assert row['top20_logprobs'] == top20(cases[i][1], cases[i][2])
    assert PRIMARY.read_bytes() == raw  # primary evidence is read-only; no retroactive pass


def test_generation_and_probe_salts_are_distinct_and_retries_are_fresh(tmp_path: Path) -> None:
    cases = [(97, 0., .5), (78, .5, .5), (63, .9, 1.)]
    api = Api(cases)
    output = tmp_path/'pass.json'
    result = run(api, output)
    assert result['greedy_registration'] == REGISTRATION
    assert result['identical'] == result['near_tie'] == 3 and api.chats == 6
    posts = [(endpoint, body) for endpoint, body in api.calls if endpoint != '/tokenize']
    salts = [body['cache_salt'] for _, body in posts]
    assert len(salts) == len(set(cast(list[str], salts))) == 9
    for _, body in posts:
        assert body['model'] == c.MODEL and body['temperature'] == 0
    assert all(body['max_tokens'] == 300 and body['chat_template_kwargs'] == {'thinking': False}
               for endpoint, body in posts if endpoint == '/v1/chat/completions')
    persisted = output.read_text()
    assert 'public-reference' not in persisted and 'public-experiment' not in persisted and 'public-fixture' not in persisted
    retry = Api(cases)
    run(retry, output)
    assert not set(cast(list[str], salts)) & {str(body['cache_salt']) for endpoint, body in retry.calls if endpoint != '/tokenize'}


@pytest.mark.parametrize('margin', [1.11, 1.5, 2., 1.1011])
def test_margin_above_registered_allowance_stops(tmp_path: Path, margin: float) -> None:
    api = Api([(63, margin, .5)])
    output = tmp_path/'stop.json'
    with pytest.raises(ValueError, match='fails pp4-r14-near-tie-1p1'):
        run(api, output)
    assert api.chats == 1
    row = ck.rows(ck.read(output)['checks'])[0]
    assert row['near_tie'] is False and row['top20_logprobs'] == top20(margin, .5)
    returned = top20(margin, .5)
    assert row['margin_nats'] == abs(number(returned['token_id:100'])-number(returned['token_id:200']))
    assert row['diagnostic_error']


@pytest.mark.parametrize('count', [4, 6])
def test_no_extra_divergence_count_limit(tmp_path: Path, count: int) -> None:
    api = Api([(10+i, .5, .5) for i in range(count)])
    result = run(api, tmp_path/'many.json')
    rows = ck.rows(result['checks'])
    assert result['near_tie'] == count and result['identical'] == 6-count
    assert api.chats == len(rows) == 6
    rows = rows[:count]
    assert all(len(record(r['top20_logprobs'])) == 20 for r in rows)


@pytest.mark.parametrize('change', ['missing', 'short-top20', 'clamped', 'bad-id', 'alias-id', 'nan', 'inf'])
def test_bad_evidence_stops_and_retains_full_returned_values(tmp_path: Path, change: str) -> None:
    api = Api([(63, .5, .5)])
    top = top20(.5, .5)
    if change == 'missing': top['token_id:99'] = top.pop('token_id:100')
    if change == 'short-top20': top.pop('token_id:317')
    if change == 'clamped': top['token_id:317'] = -9999.
    if change == 'bad-id': top['malformed'] = top.pop('token_id:317')
    if change == 'alias-id': top['token_id:0200'] = top.pop('token_id:317')
    if change == 'nan': top['token_id:317'] = float('nan')
    if change == 'inf': top['token_id:317'] = float('-inf')
    if change in ('nan', 'inf'):
        # Nonfinite JSON cannot be persisted; exercise the validator directly.
        assert c.near_tie_values(top, 100, 200)['near_tie'] is False
        return
    api.override = top
    output = tmp_path/'stop.json'
    with pytest.raises(ValueError, match='fails pp4-r14-near-tie-1p1'):
        run(api, output)
    row = ck.rows(ck.read(output)['checks'])[0]
    assert row['near_tie'] is False and row['top20_logprobs'] == top


@pytest.mark.parametrize('kind', ['longer', 'shorter'])
def test_token_flip_with_unequal_lengths_is_probed(tmp_path: Path, kind: str) -> None:
    """Round 15 (PP4-941 admission stop 2026-10-27): a flip at 63 whose continuations end at different lengths is a token near-tie."""
    api = Api([(63, 0., .5)])
    api.length_change = kind == 'longer'
    api.shorter = kind == 'shorter'
    result = run(api, tmp_path/'unequal.json')
    assert result['near_tie'] == 1 and result['identical'] == 5
    row = ck.rows(ck.read(tmp_path/'unequal.json')['checks'])[0]
    assert row['token_index'] == 63 and row['near_tie'] is True
    assert row['reference_tokens'] == 120 and row['experiment_tokens'] == (121 if kind == 'longer' else 100)


@pytest.mark.parametrize('kind', ['prefix-equal', 'prefix-longer'])
def test_pure_prefix_differences_stop_before_probe(tmp_path: Path, kind: str) -> None:
    api = Api([(63, 0., .5)])
    api.prefix_only = True
    api.length_change = kind == 'prefix-longer'
    output = tmp_path/'prefix.json'
    with pytest.raises(ValueError, match='pure prefix difference'):
        run(api, output)
    assert not any(endpoint == '/v1/completions' for endpoint, _ in api.calls)
    assert ck.rows(ck.read(output)['checks'])[0]['near_tie'] is False


@pytest.mark.parametrize('margin', [0., .5, .75, .9, 1., 1.1, 1.1005, 1.101])
def test_registered_margin_and_rounding_allowance_pass(tmp_path: Path, margin: float) -> None:
    api = Api([(63, margin, .5)])
    result = run(api, tmp_path/'boundary.json')
    assert result['near_tie'] == 1 and result['identical'] == 5
    assert ck.rows(result['checks'])[0]['margin_nats'] == pytest.approx(margin)


@pytest.mark.parametrize('gap,passes', [(1.1, True), (1.1005, True), (1.101, True), (1.1011, False), (1.11, False)])
@pytest.mark.parametrize('reference_better', [False, True])
def test_better_of_two_gap_to_top_is_independent_of_margin(
    tmp_path: Path, gap: float, passes: bool, reference_better: bool,
) -> None:
    api = Api([(63, .5, .5)])
    top = top20(.5, .5)
    # Neither token is top; both are in top20 and half a nat apart.
    top['token_id:300'] = -.25
    top['token_id:100'] = -.25-gap-(0. if reference_better else .5)
    top['token_id:200'] = -.25-gap-(.5 if reference_better else 0.)
    api.override = top
    output = tmp_path/'gap.json'
    if passes:
        assert run(api, output)['near_tie'] == 1
    else:
        with pytest.raises(ValueError, match='fails pp4-r14-near-tie-1p1'):
            run(api, output)
    row = ck.rows(ck.read(output)['checks'])[0]
    assert row['near_tie'] is passes and row['gap_to_top_nats'] == pytest.approx(gap)
    assert row['experiment_gap_to_top_nats'] == pytest.approx(gap+(.5 if reference_better else 0.))
    assert row['top20_logprobs'] == top


@pytest.mark.parametrize('change', ['off-grid', 'all-equal', 'reference-top'])
def test_grid_and_experiment_is_top_restrictions_are_retired(tmp_path: Path, change: str) -> None:
    api = Api([(63, .5, .5)])
    top = top20(.5, .5)
    if change == 'off-grid': top['token_id:317'] = -10.1
    if change == 'all-equal': top = dict.fromkeys(top, -.25)
    if change == 'reference-top': top['token_id:200'], top['token_id:100'] = -.75, -.25
    api.override = top
    result = run(api, tmp_path/'no-grid.json')
    assert result['near_tie'] == 1 and result['identical'] == 5


@pytest.mark.parametrize('endpoint', ['/v1/chat/completions', '/v1/completions'])
@pytest.mark.parametrize('salt', [None, '', ' ', 1])
def test_actual_http_refuses_missing_salt_before_transport(endpoint: str, salt: Json) -> None:
    with patch.object(urllib.request, 'urlopen') as opened:
        with pytest.raises(ValueError, match='cache_salt'):
            c.Http('http://unused.invalid').post(endpoint, {'cache_salt': salt})
        opened.assert_not_called()


def test_actual_http_forwards_salts_on_both_endpoints() -> None:
    seen: list[tuple[str, dict[str, Json]]] = []
    def opened(request: urllib.request.Request, *, timeout: int) -> io.BytesIO:
        assert timeout == 900 and isinstance(request.data, bytes)
        seen.append((request.full_url, record(cast(Json, json.loads(request.data)))))
        return io.BytesIO(b'{"ok":true}')
    with patch.object(urllib.request, 'urlopen', opened):
        for endpoint in ('/v1/chat/completions', '/v1/completions'):
            assert c.Http('http://unused.invalid').post(endpoint, {'cache_salt': 'fresh-'+endpoint}) == {'ok': True}
    assert [body['cache_salt'] for _, body in seen] == ['fresh-/v1/chat/completions', 'fresh-/v1/completions']


def test_worker_and_shared_nbw_bytes_unchanged_from_qualified_candidate() -> None:
    # Includes new tracked files/deletions and actual worktree contents, not just HEAD.
    result = subprocess.run(['git', 'diff', '--exit-code', '42928552727ee85561600d886a704d3342ad969a', '--',
        'offload/dspark/worker'], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    # R-PP4: the shared NBW module is exactly WIN2's reviewed version (WIN2-R2 2b48eec: defaulted threshold/distribution
    # parameters, NBW's own defaults unchanged); PP4 never edits it.
    shared = subprocess.run(['git', 'diff', '--exit-code', '2b48eeccf325770e3c7dc48fdbf863b80e4387ce', '--', 'offload/launch/windows/nbw_cells.py'],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert shared.returncode == 0, shared.stdout+shared.stderr
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '--', 'offload/dspark/worker'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert not untracked.stdout
    restore = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh')
    assert hashlib.sha256(restore.read_bytes()).hexdigest() == 'd1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49'
