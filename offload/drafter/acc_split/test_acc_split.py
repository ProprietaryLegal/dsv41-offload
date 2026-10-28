"""ACC-SPLIT tests on synthetic files (no campaign data, no text): every layout, de-duplication, the role filter, the
estimators and the shift decomposition."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from offload.drafter.acc_split import __main__ as cli
from offload.drafter.acc_split import analysis as a
from offload.drafter.acc_split import sources as s

SHA_A, SHA_B = 'a' * 64, 'b' * 64
NAME = 'DeepSeek'


def _spec(rounds: float, drafts: float, accepted: float, pos: list[float]) -> dict[str, object]:
    return {'rounds': rounds, 'proposed_tokens': drafts, 'accepted_tokens': accepted,
            'accepted_per_position': {str(i): v for i, v in enumerate(pos)}}


def _bench2(path: Path, rid: str, sha: str, workload: str, ctx: int, max_tokens: int, rounds: float, accepted: float,
            *, ts: str = '2026-10-23T10:00:00Z', label: str = 'X', error: object = None, answer: int = 0,
            tps: float = 50.0, extra: dict[str, object] | None = None) -> None:
    run = {'run_id': rid, 'config_label': label, 'type': 'run', 'ts': ts, 'target_ctx': ctx, 'workload': workload,
           'concurrency': 1, 'cache_state': 'cold', 'spec': _spec(rounds, 3 * rounds, accepted, [accepted * .6, accepted * .4])}
    req = {'run_id': rid, 'config_label': label, 'type': 'request', 'request_index': 0, 'prompt_sha256': sha,
           'prompt_tokens': 100, 'completion_tokens': max_tokens, 'reasoning_tokens': max_tokens - answer,
           'answer_tokens': answer, 'decode_tps': tps, 'error': error, 'step_events': int(rounds),
           'step_gap_ms': {'mean': 40.0, 'p50': 39.0}, 'settings': {'max_tokens': max_tokens}}
    req.update(extra or {})
    with path.open('a') as f:
        f.write(json.dumps(run) + '\n' + json.dumps(req) + '\n')


def test_bench2_dedupes_copies_and_drops_errors(tmp_path: Path) -> None:
    one = tmp_path / 'r1' / 'A-short.jsonl'
    one.parent.mkdir()
    _bench2(one, 'r1', SHA_A, 'legal', 0, 1500, 700, 700)
    _bench2(one, 'r2', SHA_A, 'legal', 0, 1500, 700, 350, error='boom')
    copy = tmp_path / 'checkout' / 'A-short.jsonl'
    copy.parent.mkdir()
    copy.write_bytes(one.read_bytes())
    files = s.bench2_files([tmp_path])
    assert len(files) == 1
    obs = s.load_bench2(files + [copy])            # a copy passed explicitly is still de-duplicated by run_id
    assert [o.where for o in obs] == ['r1']
    assert obs[0].accepted_per_round == 1.0 and obs[0].depth == 3.0 and obs[0].cell == 'legal-0'


def _metrics(drafts: float, draft_tokens: float, accepted: float, pos: list[float]) -> dict[str, float]:
    m = {f'{s.DRAFTS}{{engine="0",model_name="{NAME}"}}': drafts, f'{s.DRAFT_TOKENS}{{engine="0"}}': draft_tokens,
         f'{s.ACCEPTED}{{engine="0"}}': accepted}
    m.update({f'{s.PER_POS}{{engine="0",position="{i}"}}': v for i, v in enumerate(pos)})
    return m


def _windows(root: Path) -> None:
    sp1 = root / 'SP1-W-20261026T144153Z' / 'cells'
    sp1.mkdir(parents=True)
    stream = {'prompt': 'short', 'prompt_tokens': 1014, 'prompt_sha256': SHA_B, 'reasoning_tokens': 1500,
              'answer_and_control_tokens': 0.0, 'sampling': {'max_tokens': 1500}}
    (sp1 / 'ctl-short-b1.json').write_text(json.dumps({'streams': [stream], 'spec': _spec(600, 1800, 690, [400, 200, 90])}))
    (sp1 / 'warmup.json').write_text(json.dumps({'streams': [stream], 'spec': _spec(10, 30, 10, [10])}))
    pc1 = root / 'PC1-x-W-20261027T100908Z'
    pc1.mkdir()
    (pc1 / 'summary.json').write_text(json.dumps({'starts': {'s1-control': {'role': 'control'},
                                                             's2-cand': {'role': 'candidate'}}}))
    rep_stream = {'prompt_sha256': SHA_B, 'prompt_tokens': 1014, 'reasoning_tokens': 512, 'answer_and_control_tokens': 0.0,
                  'sampling': {'max_tokens': 512}, 'decode_tps': 55.0}
    for start, acc in (('s1-control', 260.0), ('s2-cand', 100.0)):
        reps = [{'streams': [rep_stream], 'metrics_before': _metrics(100, 300, 50, [30, 15, 5]),
                 'metrics_after': _metrics(300, 1000, 50 + acc, [30 + acc * .6, 15 + acc * .3, 5 + acc * .1]),
                 'server_decode_ms': 7800.0, 'spec_request_rounds': 200.0}]
        (pc1 / f'{start}-short-b1.json').write_text(json.dumps({'cell': 'short-b1', 'repetitions': reps}))
        (pc1 / f'{start}-greedy.json').write_text(json.dumps({'cell': 'greedy'}))


def test_window_layouts_and_role_filter(tmp_path: Path) -> None:
    _windows(tmp_path)
    obs = s.load_windows(s.window_dirs(tmp_path))
    by = {(o.label[:3], o.role): o for o in obs}
    assert len(obs) == 3                                     # warmup and greedy are skipped
    sp1 = by[('SP1', 'control')]
    assert sp1.cell == 'ctl-short-b1' and sp1.accepted_per_round == pytest.approx(1.15) and sp1.date == '2026-10-26'
    pc1 = by[('PC1', 'control')]
    assert pc1.rounds == 200 and pc1.accepted_per_round == pytest.approx(1.3) and pc1.depth == pytest.approx(3.5)
    assert pc1.ms_per_round == pytest.approx(39.0) and pc1.max_tokens == 512 and pc1.per_position[0] == pytest.approx(156)
    rows = a.by_prompt(obs)                                  # candidate arm excluded
    assert {(r['max_tokens'], record_n(r)) for r in rows} == {(512, 1), (1500, 1)}


def record_n(row: Mapping[str, object]) -> int:
    acc = row['accepted_per_round']
    assert isinstance(acc, dict)
    n = acc['n']
    assert isinstance(n, int)
    return n


def test_summarize_interval() -> None:
    sm = a.summarize([1.0, 1.2, 1.4])
    assert sm.mean == pytest.approx(1.2) and sm.sd == pytest.approx(.2)
    assert sm.hi - sm.mean == pytest.approx(4.303 * .2 / math.sqrt(3))
    assert a.summarize([2.0]).lo == 2.0
    with pytest.raises(ValueError):
        a.summarize([])


def test_decompose_prompt_then_length(tmp_path: Path) -> None:
    f = tmp_path / 'b.jsonl'
    for i, acc in enumerate((1000.0, 1020.0, 980.0)):           # old prompt, 1500 tokens: ~1.0
        _bench2(f, f'o{i}', SHA_A, 'legal', 0, 1500, 1000, acc)
    obs = s.load_bench2([f])
    _windows(tmp_path / 'w')
    obs += s.load_windows(s.window_dirs(tmp_path / 'w'))
    d = a.decompose(obs, 'bench2:legal-0', 'corpus:short-b1', 1500, 512)
    assert d['prompt_step'] == pytest.approx(0.15) and d['length_step'] == pytest.approx(0.15)
    assert d['total'] == pytest.approx(0.30)
    with pytest.raises(ValueError):
        a.decompose(obs, 'bench2:legal-0', 'corpus:nothing', 1500, 512)


def test_segment_estimators(tmp_path: Path) -> None:
    f = tmp_path / 'b.jsonl'
    # reasoning-only run with SSE times: 700 events over the reasoning span, 1400 tokens, server 700 rounds / 700 accepted
    times = [i * 40.0 for i in range(700)]
    _bench2(f, 'r', SHA_A, 'legal', 0, 1400, 700, 700, extra={
        'step_times_ms': times, 'first_by_kind_s': {'reasoning': 0.0}, 'last_by_kind_s': {'reasoning': times[-1] / 1000}})
    # mixed run without SSE times, 40 ms rounds: reasoning 600 tok at 50 tok/s (A=1, 300 rounds), answer 450 tok at
    # 75 tok/s (A=2, 150 rounds): 450 rounds, 600 accepted
    _bench2(f, 'm', SHA_B, 'summary', 32000, 16384, 450, 600, answer=450, extra={
        'completion_tokens': 1050, 'reasoning_tokens': 600, 'reasoning_decode_tps': 50.0, 'answer_decode_tps': 75.0})
    seg = a.segment_split(s.load_bench2([f]))
    assert seg['validation_runs'] == 1
    (m,) = cast(list[dict[str, object]], seg['mixed_runs'])
    assert m['method'] == 'tps_equal_round_time'
    assert m['round_s_estimated'] == pytest.approx(0.04)
    assert m['reasoning_estimated'] == pytest.approx(1.0) and m['answer_estimated'] == pytest.approx(2.0)


def test_metric_reconciliation(tmp_path: Path) -> None:
    f = tmp_path / 'b.jsonl'
    for i in range(3):
        _bench2(f, f'r{i}', SHA_A, 'legal', 0, 1500, 750, 750, tps=50.0)
    (row,) = a.metric_reconciliation(s.load_bench2([f]))
    assert row['server'] == 1.0 and row['tps_x_p50_gap'] == pytest.approx(0.95) and row['tps_x_mean_gap'] == 1.0
    assert row['tokens_per_sse_event'] == 1.0 and row['rounds_per_sse_event'] == 1.0


def test_cli_writes_and_refuses_overwrite(tmp_path: Path) -> None:
    f = tmp_path / 'bench' / 'b.jsonl'
    f.parent.mkdir()
    for i, acc in enumerate((1000.0, 1020.0)):
        _bench2(f, f'o{i}', SHA_A, 'legal', 0, 1500, 1000, acc)
        _bench2(f, f's{i}', 'c' * 64, 'summary', 32000, 1500, 1000, 1600 + acc / 100)
        _bench2(f, f'y{i}', 'd' * 64, 'synth', 150000, 1500, 1000, 1200 + acc / 100)
    camp = tmp_path / 'camp'
    _windows(camp)
    for name, sha in (('32k-b1', 'e' * 64), ('150k-b1', 'f' * 64)):
        for w in ('SP1-W-20261026T144153Z', 'PC1-x-W-20261027T100908Z'):
            src = camp / w / ('cells/ctl-short-b1.json' if w.startswith('SP1') else 's1-control-short-b1.json')
            dst = src.with_name(src.name.replace('short-b1', name))
            dst.write_text(src.read_text().replace(SHA_B, sha))
    out = tmp_path / 'out'
    assert cli.main(['--campaign', str(camp), '--bench2-root', str(f.parent), '--out', str(out)]) == 0
    rep = json.loads((out / 'acc-split.json').read_text())
    assert set(rep['shifts']) == {'32K', 'short', '150K'} and (out / 'acc-split.md').read_text().startswith('# ACC-SPLIT')
    with pytest.raises(SystemExit):
        cli.main(['--campaign', str(camp), '--bench2-root', str(f.parent), '--out', str(out)])
