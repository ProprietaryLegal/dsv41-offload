from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from offload.prefill_maxlen.evidence import Json, record, number
from offload.up3 import cells
from offload.launch.windows import nbw_cells as greedy_driver


def test_delivery_p99_uses_continuous_actual_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Json]] = []
    def data(count: int, content: str, finish: Json = None) -> bytes:
        return ('data: ' + json.dumps({'id': 'public', 'choices': [{'delta': {'content': content}, 'finish_reason': finish}],
                                      'usage': {'completion_tokens': count, 'prompt_tokens': 10}}) + '\n').encode()
    @contextmanager
    def urlopen(request: Any, timeout: float) -> Iterator[list[bytes]]:
        seen.append(json.loads(request.data))
        yield [data(1, 'a'), data(3, 'bc'), data(3, '', 'length'), b'data: [DONE]\n']
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    first = threading.Event()
    row = cells.Http('http://fixture').stream({}, first)
    assert first.is_set()
    assert row['delivery_gap_samples'] == 2
    assert number(row['p99_inter_token_delivery_gap_ms']) >= 0
    assert record(seen[0]['stream_options'])['continuous_usage_stats'] is True
    assert cells.p99([0.] * 99 + [100.]) == 0
    assert cells.p99([0.] * 98 + [100., 100.]) == 100


def test_missing_continuous_usage_refuses_p99(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def urlopen(request: Any, timeout: float) -> Iterator[list[bytes]]:
        yield [b'data: {"id":"public","choices":[{"delta":{"content":"abc"},"finish_reason":"length"}],"usage":{"completion_tokens":3}}\n',
               b'data: [DONE]\n']
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    with pytest.raises(ValueError, match='per-token delivery timing unavailable'):
        cells.Http('http://fixture').stream({}, threading.Event())


class Api:
    def __init__(self) -> None:
        self.requests: list[dict[str, Json]] = []
        self.stamps: list[float] = []
        self.active = 0
        self.completed = 0
        self.lock = threading.Lock()

    def model(self) -> dict[str, Json]:
        return {'id': 'fixture', 'max_model_len': 327680}

    def tokens(self, text: str) -> int:
        return 100

    def metrics(self) -> str:
        return (f'vllm:num_requests_running {self.active}\nvllm:num_requests_waiting 0\n'
                f'vllm:spec_decode_num_drafts_total {100*self.completed}\n'
                f'vllm:spec_decode_num_accepted_tokens_total {300*self.completed}\n'
                f'vllm:request_decode_time_seconds_sum {4*self.completed}\n'
                f'vllm:request_decode_time_seconds_count {self.completed}\n')

    def profile(self, verb: str) -> None:
        raise AssertionError('UP3 never profiles')

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        with self.lock:
            self.active += 1
            self.requests.append(body)
            self.stamps.append(time.monotonic())
        first.set()
        time.sleep(.15)
        with self.lock:
            self.active -= 1
            self.completed += 1
        n = 150000 if record(cast(list[Json], body['messages'])[0])['content'] == 'long' else 32000
        return {'usage': {'completion_tokens': body['max_tokens'], 'prompt_tokens': n, 'prompt_tokens_details': {'cached_tokens': 0}},
            'finish': 'length', 'ttft_s': .01, 'first_by_kind_s': {'reasoning': .01},
            'last_by_kind_s': {'reasoning': .3}, 'p99_inter_token_delivery_gap_ms': 20.}


def test_real_mixed_submission_and_fixed_matched_work(tmp_path: Path) -> None:
    api = Api()
    prompts = {key: ('long' if key == '150k' else 'public', 150000 if key == '150k' else 32000)
               for key in cells.KEYS}
    result = cells.run(api, cells.CELLS[-1], prompts, deadline=time.time() + 500, salt='scope', ledger=tmp_path / 'ledger')
    assert result['batch'] == 3
    assert result['mixed_three_active_overlap'] is True
    assert [r['max_tokens'] for r in api.requests] == [512, 512, 512]
    assert api.stamps[2] >= max(api.stamps[:2])
    assert len({r['cache_salt'] for r in api.requests}) == 3
    assert all(r['temperature'] == 1 and r['seed'] == 42 and r['priority'] == 10 for r in api.requests)
    assert number(result['acceptance_per_round']) == 3


@pytest.mark.parametrize('argv,ok', [
    (['--mode', 'greedy'], True),
    (['--mode', 'greedy', '--near-tie-nats', repr(cells.REPLAY_NEAR_TIE_NATS)], True),
    (['--mode', 'greedy', '--near-tie-nats', '0.9'], False),
    (['--mode', 'greedy', '--near-tie-nats', '1.101'], False),
    (['--mode', 'cell', '--cell', 'short-b1', '--near-tie-nats', repr(cells.REPLAY_NEAR_TIE_NATS)], False)])
def test_near_tie_threshold_is_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], ok: bool) -> None:
    assert cells.REPLAY_NEAR_TIE_NATS == float('0.901')
    monkeypatch.setattr('sys.argv', ['cells', *argv, '--output', str(tmp_path / 'o.json'), '--data-dir', str(tmp_path),
                                     '--deadline', '0'])
    if ok:
        assert cells.main() == 0
    else:
        with pytest.raises(ValueError, match='unregistered near-tie threshold'):
            cells.main()


def test_replay_greedy_passes_threshold_and_records_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    def greedy(api: object, deadline: float, prompts: object, reference: object, output: Path, *,
               near_tie_nats: float, record_distribution: bool, probe_salt_prefix: str | None) -> dict[str, Json]:
        assert not record_distribution and probe_salt_prefix is None
        seen.append(near_tie_nats)
        return {'status': 'PASS'}
    monkeypatch.setattr(greedy_driver, 'greedy', greedy)
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: ([], []))
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    out = tmp_path / 'greedy.json'
    monkeypatch.setattr('sys.argv', ['cells', '--execute', '--mode', 'greedy', '--near-tie-nats', repr(cells.REPLAY_NEAR_TIE_NATS),
                                     '--output', str(out), '--data-dir', str(tmp_path), '--deadline', str(time.time() + 60)])
    assert cells.main() == 0
    assert seen == [cells.REPLAY_NEAR_TIE_NATS]
    assert json.loads(out.read_text()) == {'status': 'PASS', 'greedy_registration': cells.REPLAY_REGISTRATION,
                                           'near_tie_nats': cells.REPLAY_NEAR_TIE_NATS, 'distribution_recorded': False}


@pytest.mark.parametrize('argv', [
    ['--mode', 'greedy', '--record-distribution', '--salt', 's'],                       # NBW default limit
    ['--mode', 'greedy', '--near-tie-nats', repr(cells.REPLAY_NEAR_TIE_NATS), '--record-distribution'],   # no salt
    ['--mode', 'greedy-kl', '--salt', 's', '--kl-bound', '.005'],                         # no candidate
    ['--mode', 'greedy-kl', '--salt', 's', '--candidate', 'c.json', '--kl-bound', '.004']])  # below UP2's floor
def test_distribution_options_are_the_replay_registration_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               argv: list[str]) -> None:
    monkeypatch.setattr('sys.argv', ['cells', *argv, '--output', str(tmp_path / 'o.json'), '--data-dir', str(tmp_path),
                                     '--deadline', '0'])
    with pytest.raises(ValueError):
        cells.main()


def test_greedy_kl_reprobes_recorded_divergences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.launch.windows import greedy_kl
    calls: list[tuple[str, float, int]] = []
    def compare(api: object, rows: list[dict[str, Json]], salt: str, bound: float) -> list[dict[str, Json]]:
        calls.append((salt, bound, len(rows)))
        return [{'i': 2, 'kl_nats': .001, 'common_mass': .999, 'bound': bound, 'pass': True}]
    monkeypatch.setattr(greedy_kl, 'compare_recorded', compare)
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    candidate = tmp_path / 'candidate.json'
    rows = [{'i': i, 'identical': i != 2} for i in range(6)]
    candidate.write_text(json.dumps({'checks': rows, 'greedy_registration': cells.REPLAY_REGISTRATION,
                                     'distribution_recorded': True}))
    out = tmp_path / 'kl.json'
    monkeypatch.setattr('sys.argv', ['cells', '--execute', '--mode', 'greedy-kl', '--candidate', str(candidate),
        '--kl-bound', '0.0123', '--salt', 'tag-s3', '--output', str(out), '--data-dir', str(tmp_path),
        '--deadline', str(time.time() + 60)])
    assert cells.main() == 0
    assert calls == [('tag-s3', .0123, 6)]
    result = json.loads(out.read_text())
    assert result['status'] == 'PASS' and result['bound'] == .0123 and result['registration'] == cells.REPLAY_REGISTRATION
    candidate.write_text(json.dumps({'checks': rows, 'greedy_registration': cells.REPLAY_REGISTRATION}))
    out.unlink()
    with pytest.raises(ValueError, match='recorded distributions'):
        cells.main()
