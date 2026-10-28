"""NBW pure gates: no network, host or GPU access."""
from __future__ import annotations

import ast
import copy
import json
import shlex
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import nbw_cells as c
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

NATIVE = n.Native('dsv41-offload/nbw-source/repo', 'd' * 40, 'c' * 64, 'b' * 64)


def production(rank: int = 0) -> dict[str, Json]:
    row = base.prod_identity(rank)
    record(row['env']).update(dict(zip(n.IDENTITY_KEYS, ('a' * 64, 'b' * 64))))
    return row


def identity(arm: n.Arm, rank: int = 0) -> dict[str, Json]:
    row = production(rank)
    row['id'] = arm.name + str(rank)
    row['argv'] = str(row['argv']).replace('--port 8000', '--port 8001')
    record(row['mounted'])['/serve.sh'] = '9' * 64
    if arm.role == 'I':
        record(row['env']).update(dict(zip(n.IDENTITY_KEYS, (NATIVE.identity, NATIVE.markov))))
    return row


def sample_cell(name: str, tps: float = 50) -> dict[str, Json]:
    batch = c.CELLS[name].batch
    return {'cell': name, 'batch': batch, 'median_decode_tps': tps, 'acceptance_per_round': 1.5,
            'streams': [{'stream': i, 'prompt_sha256': str(i) * 64, 'decode_tps': tps,
                         'sampling': {'temperature': 1, 'top_p': 1, 'max_tokens': 1500},
                         'usage': {'completion_tokens': 1500}, 'finish': 'length',
                         'ttft_s': 10, 'first_by_kind_s': {'reasoning': 10},
                         'last_by_kind_s': {'reasoning': 10 + 1499 / tps}} for i in range(batch)]}


def starts() -> dict[str, Json]:
    return {arm.name: {'status': 'COMPLETE', 'role': arm.role, 'greedy': {'identical': 6},
                       'cells': {name: sample_cell(name, 55 if arm.role == 'I' else 50 + i * .1)
                                 for name in c.CELLS}} for i, arm in enumerate(n.ARMS)}


def test_gate_uses_control_spread_and_requires_both_b1_wins() -> None:
    rows = starts()
    assert n.decision(rows)['status'] == 'CANDIDATE'
    for name in c.CELLS:
        changed = copy.deepcopy(rows)
        for arm in n.ARMS:
            if arm.role == 'I':
                record(record(changed[arm.name])['cells'])[name] = sample_cell(name, 50)
        assert n.decision(changed)['status'] == ('NOT A CANDIDATE' if 'b1' in name else 'CANDIDATE')
    for arm in n.ARMS:
        if arm.role == 'I':
            record(record(rows[arm.name])['cells'])['32k-b3'] = sample_cell('32k-b3', 49)
    assert n.decision(rows)['status'] == 'NOT A CANDIDATE'
    assert n.decision({})['status'] == 'NOT A CANDIDATE'
    assert n.decision({**starts(), n.ARMS[-1].name: {'status': 'SKIPPED_TIME', 'role': 'K'}})['status'] == 'CANDIDATE'


def test_gate_fails_incomplete_and_mismatched_evidence() -> None:
    rows = starts()
    record(rows['s2-I'])['status'] = 'STARTED'
    assert n.decision(rows)['status'] == 'NOT A CANDIDATE'
    rows = starts()
    record(rows['s2-I'])['greedy'] = {'identical': 5}
    with pytest.raises(ValueError, match='six identical'):
        n.decision(rows)
    rows = starts()
    record(record(record(rows['s2-I'])['cells'])['32k-b1'])['streams'] = [{'prompt_sha256': 'other'}]
    with pytest.raises(ValueError, match='public prompts differ'):
        n.decision(rows)


@pytest.mark.parametrize('arm', n.ARMS)
def test_runtime_preserves_all_but_port_and_declared_worker_identity(arm: n.Arm) -> None:
    n.runtime(production(), identity(arm), arm, NATIVE)
    for key, value in (('VLLM_ENGRAM_REMOTE_REQUIRED', '1'), ('VLLM_DSPARK_REMOTE_DRAFT_MS', '1.0'),
                       ('VLLM_DSV41_KV_PAIR', '2')):
        changed = identity(arm)
        record(changed['env'])[key] = value
        with pytest.raises(ValueError, match='env drift'):
            n.runtime(production(), changed, arm, NATIVE)
    changed = identity(arm)
    record(changed['mounted'])['extra'] = 'x'
    with pytest.raises(ValueError, match='mounted'):
        n.runtime(production(), changed, arm, NATIVE)


def test_metrics_come_from_server_counter_deltas_and_complete_output() -> None:
    before: dict[str, float] = {'vllm:spec_decode_num_drafts_total{m="x"}': 100,
              'vllm:spec_decode_num_accepted_tokens_total{m="x"}': 150,
              'vllm:request_decode_time_seconds_sum{m="x"}': 3}
    after = {k: v * 2 for k, v in before.items()}
    row = c.summarize(sample_cell('32k-b2'), before, after)
    assert row['acceptance_per_round'] == 1.5 and row['server_decode_ms_per_spec_request_round'] == 30
    assert row['median_decode_tps'] == pytest.approx(50)
    with pytest.raises(ValueError, match='missing/reset'):
        c.summarize(sample_cell('32k-b2'), after, before)
    with pytest.raises(ValueError, match='missing/reset'):
        c.summarize(sample_cell('32k-b2'), {}, {})
    bad = sample_cell('32k-b2')
    record(record(bad['streams'][0])['usage'])['completion_tokens'] = 1499  # type: ignore[index]
    with pytest.raises(ValueError, match='budget incomplete'):
        c.summarize(bad, before, after)


def test_native_commands_pin_source_switch_pair_and_refuse_builds(tmp_path: Path) -> None:
    NATIVE.validate()
    script = n.native_script(NATIVE, 'tag-s2-I')
    assert script.index('prod_window.sh pair COORD') < script.index('head -1 ~/dsv41-offload/PAIR_OK')
    assert 'timeout --signal=TERM --kill-after=15s --foreground 120 bash' in script
    assert 'DSPARK_NATIVE_BATCH_INVARIANT=2' in script and pf.WORKER_GPUS in script
    assert '--tp 2' in script and '--max-model-len 1048576' in script and '--ofl 10.202.28.1:18620' in script
    assert 'prod/dspark-worker/current' in script and 'native-hybrid-kx2' in script
    assert 'read-only Triton cache miss' in script and 'build' in script
    assert 'nbw/tag-s2-I/status.json' in script and NATIVE.commit in script
    subprocess.run(['bash', '-n', '-c', script], check=True)
    ast.parse(n.NATIVE_BOOTSTRAP)
    ast.parse(n.native_probe_script(NATIVE, 'tag-s2-I'))
    ast.parse(n.native_identity_script())
    for native in (n.Native('dsv41-deploy', NATIVE.commit, NATIVE.identity, NATIVE.markov),
                   n.Native(NATIVE.tree, 'short', NATIVE.identity, NATIVE.markov),
                   n.Native(NATIVE.tree, NATIVE.commit, '5c3be5c7' + 'a' * 56, NATIVE.markov)):
        with pytest.raises(ValueError):
            native.validate()


def test_dry_and_unauthorized_execute_never_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: pytest.fail('dry mode ran git'))
    assert n.main([]) == 0 and n.main(['--dry-run']) == 0
    assert capsys.readouterr().out.startswith('DRY RUN')
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert n.main(['--execute']) == 1
    assert 'before any remote action' in capsys.readouterr().err
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    assert n.main(['--execute']) == 1  # missing reviewed identities must refuse before git or remote access


def ck_rows(value: Json) -> list[Json]:
    assert isinstance(value, list)
    return value


class GreedyApi(c.Http):
    # margin=None: identical outputs; otherwise prompt 3 diverges at token 2 with that logprob margin.
    def __init__(self, margin: float | None = None) -> None:
        super().__init__('unused-no-network')
        self.margin = margin
        self.calls: list[dict[str, Json]] = []
        self.chats = 0

    def metrics(self) -> str:
        return 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'

    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
        self.calls.append({'endpoint': endpoint, **body})
        if endpoint == '/v1/chat/completions':
            self.chats += 1
            text = 'public changed output' if self.margin is not None and self.chats == 4 else 'public reference'
            return {'choices': [{'message': {'content': text}}]}
        if endpoint == '/tokenize':
            if 'messages' in body:
                return {'tokens': [1, 2, 3]}
            return {'tokens': [10, 11, 12, 13] if body['prompt'] == 'public reference' else [10, 11, 99, 13]}
        assert endpoint == '/v1/completions' and body['prompt'] == [1, 2, 3, 10, 11] and body['max_tokens'] == 1
        return {'choices': [{'logprobs': {'top_logprobs': [{'token_id:12': -0.6, 'token_id:99': -0.6 - float(cast(float, self.margin)),
                                                             'token_id:7': -3.0}]}}]}


def test_real_greedy_driver_counts_six_and_stops_at_first_divergence(tmp_path: Path) -> None:
    reference: list[dict[str, Json]] = [{'i': i, 'text': 'public reference'} for i in range(6)]
    api = GreedyApi()
    output = tmp_path / 'pass.json'
    result = c.greedy(api, 1e12, ['public prompt'] * 6, reference, output)
    assert result['identical'] == 6 and result['near_tie'] == 0 and api.chats == 6
    assert all(body['temperature'] == 0 and body['max_tokens'] == 300 for body in api.calls)
    assert 'public reference' not in output.read_text() and 'public prompt' not in output.read_text()
    tie = GreedyApi(0.5)
    result = c.greedy(tie, 1e12, ['public prompt'] * 6, reference, tmp_path / 'tie.json')
    assert result['identical'] == 5 and result['near_tie'] == 1 and tie.chats == 6
    row = record(ck_rows(result['checks'])[3])
    assert row['near_tie'] is True and row['token_index'] == 2 and row['margin_nats'] == 0.5
    assert 'public changed output' not in (tmp_path / 'tie.json').read_text()
    failed = GreedyApi(0.75)
    partial = tmp_path / 'stop.json'
    with pytest.raises(ValueError, match='reference prompt 3 is not a near-tie'):
        c.greedy(failed, 1e12, ['public prompt'] * 6, reference, partial)
    assert failed.chats == 4
    assert 'public changed output' not in partial.read_text()
    last = json.loads(partial.read_text())['checks'][-1]
    assert last['identical'] is False and last['near_tie'] is False
