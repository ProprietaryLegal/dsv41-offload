from __future__ import annotations

import copy
import inspect
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import nbw_cells as nbw
from offload.launch.windows.tests.test_nbw_window import GreedyApi
from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import cells, greedy as g


def public_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    references: list[dict[str, Json]] = [{'i': i, 'text': 'public reference'} for i in range(6)]
    source = tmp_path / 'references.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in references)+'\n')
    pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'references':[{'i':i, 'text_sha256':hashlib.sha256(b'public reference').hexdigest()} for i in range(6)]}))
    monkeypatch.setattr(nbw, 'REFERENCE', source)
    monkeypatch.setattr(g, 'REFERENCE_PINS', pins)
    monkeypatch.setattr(nbw, 'greedy_inputs', lambda: (['public prompt']*6, references))


@pytest.mark.parametrize('margin', [None, 0., .5, .501, .75, .9, .90101, .91, 1.])
def test_delegates_real_nbw_checker_and_persists_diagnosis(
        margin: float | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_references(tmp_path, monkeypatch)
    actual = nbw.greedy
    calls: list[Path] = []
    def delegated(api: wc.Http, deadline: float, prompts: list[str], reference: list[dict[str, Json]],
                  output: Path, near_tie_nats: float = nbw.NEAR_TIE_NATS) -> dict[str, Json]:
        assert near_tie_nats == g.NEAR_TIE_NATS
        calls.append(output)
        return actual(api, deadline, prompts, reference, output, near_tie_nats)
    monkeypatch.setattr(nbw, 'greedy', delegated)
    output = tmp_path / 'greedy.json'
    api = GreedyApi(margin)
    accepted = margin is None or margin <= g.NEAR_TIE_NATS
    if accepted:
        result = g.run(api, 1e12, output)
        g.validate(result)
    else:
        with pytest.raises(ValueError, match='not a near-tie'):
            g.run(api, 1e12, output)
    receipt = json.loads(output.read_text())
    assert calls == [output] and receipt['registration_id'] == 'tv1-r10-near-tie-0p9'
    assert receipt['greedy_rule'] == g.RULE and receipt['near_tie_nats'] == g.NEAR_TIE_NATS == 0.9 + 1e-3
    assert api.chats == (6 if accepted else 4)
    assert all(text not in output.read_text() for text in ('public prompt', 'public reference', 'public changed output'))
    if margin is not None:
        row = receipt['checks'][3]
        assert row['identical'] is False and row['near_tie'] is accepted
        assert row['token_index'] == 2 and row['margin_nats'] == round(margin, 4)
        assert row['gap_to_top_nats'] == 0
        probes = [r for r in api.calls if r['endpoint'] == '/v1/completions']
        assert len(probes) == 1
        assert probes[0]['prompt'] == [1, 2, 3, 10, 11]
        assert probes[0]['logprobs'] == 20 and probes[0]['max_tokens'] == 1
    else:
        assert not any(r['endpoint'] == '/v1/completions' for r in api.calls)
    if not accepted:
        assert 'not a near-tie' in receipt['diagnostic_error']


@pytest.mark.parametrize('failure', ['outside-top20', 'far-from-top', 'prefix', 'transport'])
def test_nbw_diagnostic_failures_stop_and_keep_partial_receipt(
        failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_references(tmp_path, monkeypatch)
    class Api(GreedyApi):
        def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]:
            if endpoint == '/tokenize' and failure == 'prefix' and body.get('prompt') == 'public changed output':
                return {'tokens':[10,11]}
            result = super().post(endpoint, body)
            if endpoint == '/v1/completions':
                if failure == 'transport':
                    raise ValueError('public diagnostic RPC failed')
                top = record(cast(list[Json], record(record(cast(list[Json], result['choices'])[0])['logprobs'])['top_logprobs'])[0])
                if failure == 'outside-top20':
                    del top['token_id:99']
                elif failure == 'far-from-top':
                    top['token_id:7'] = .5  # better compared token -0.6 is 1.1 nats below the top: > 0.901
            return result
    path = tmp_path/'failed.json'
    with pytest.raises(ValueError):
        g.run(Api(.1),1e12,path)
    result = json.loads(path.read_text())
    assert result['registration_id'] == g.REGISTRATION_ID and result['diagnostic_error']
    if failure in ('outside-top20','far-from-top'):
        row = result['checks'][-1]
        assert row['i'] == 3 and row['token_index'] == 2 and row['near_tie'] is False
        assert row['margin_nats'] == (None if failure == 'outside-top20' else .1)
    else:
        assert [r['i'] for r in result['checks']] == [0,1,2]
    with pytest.raises(ValueError):
        g.validate(result)


def test_cell_cli_keeps_nbw_partial_file_and_does_not_reopen_exclusive(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_references(tmp_path, monkeypatch)
    api = GreedyApi(.5)
    monkeypatch.setattr(cells,'Http',lambda url: api)
    output = tmp_path/'cli.json'
    args = ['--execute','greedy','--deadline-utc','1000000000000','--output',str(output)]
    assert cells.main(args) == 0
    g.validate(json.loads(output.read_text()))
    with pytest.raises(ValueError,match='must be new'):
        cells.main(args)


@pytest.mark.parametrize('bad', ['registration', 'threshold', 'missing', 'duplicate', 'false', 'margin', 'gap', 'position',
                                 'counts'])
def test_receipt_rejects_wrong_registration_or_missing_diagnosis(bad: str) -> None:
    rows: list[Json] = [{'i':i,'identical':True} for i in range(6)]
    rows[3] = {'i':3,'identical':False,'near_tie':True,'token_index':2,'margin_nats':.5,'gap_to_top_nats':0.}
    result = g.registered({'checks':rows,'identical':5,'near_tie':1})
    g.validate(result)
    broken = copy.deepcopy(result)
    checks = cast(list[Json],broken['checks'])
    if bad == 'registration':
        broken['registration_id'] = 'tv1-r7-nbw-near-tie'
    elif bad == 'threshold':
        broken['near_tie_nats'] = nbw.NEAR_TIE_NATS
    elif bad == 'missing':
        checks.pop()
    elif bad == 'duplicate':
        record(checks[5])['i'] = 4
    elif bad == 'counts':
        broken['identical'] = 6
    else:
        record(checks[3])[{'false':'near_tie','margin':'margin_nats','gap':'gap_to_top_nats','position':'token_index'}[bad]] = {
            'false':False,'margin':.902,'gap':.902,'position':None}[bad]
    with pytest.raises((ValueError,TypeError)):
        g.validate(broken)


# Real compact evidence, numeric fields only (no text or text hashes): TV1-compact-W-20261027T141057Z/arms/s2-compact/
# greedy.json (sha256 74f2d31b60e6e5868e3c4582d9a34594e59c41bc017df57528e4999f07576876), registered tv1-r7: K1 prompt 0
# diverged at token 64, margin 0.75 nats, gap to top 0.0 -> STOP at 0.501. Old evidence is not reclassified (WINDOW.md).
COMPACT_141057Z_MARGIN = .75


def test_real_compact_margin_passes_r10_but_not_the_nbw_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_references(tmp_path, monkeypatch)
    result = g.run(GreedyApi(COMPACT_141057Z_MARGIN), 1e12, tmp_path / 'r10.json')
    g.validate(result)
    row = record(cast(list[Json], result['checks'])[3])
    assert row['near_tie'] is True and row['margin_nats'] == COMPACT_141057Z_MARGIN and row['gap_to_top_nats'] == 0
    prompts, references = nbw.greedy_inputs()
    with pytest.raises(ValueError, match='not a near-tie'):
        nbw.greedy(GreedyApi(COMPACT_141057Z_MARGIN), 1e12, prompts, references, tmp_path / 'nbw-default.json')


def test_nbw_default_threshold_is_unchanged() -> None:
    assert nbw.NEAR_TIE_NATS == 0.5 + 1e-3
    for function, name in ((nbw.greedy, 'near_tie_nats'), (nbw.near_tie, 'threshold')):
        assert inspect.signature(function).parameters[name].default == 0.5 + 1e-3
