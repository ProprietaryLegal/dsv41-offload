"""WIN2-R3 seeded A/A pilot recipe: two starts, PASS on identical B1 streams, measured diff CV and the A/A band input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.test_window_cells_seeded import SeededFake, plan, register_fake_texts, texts
from offload.launch.windows import aa_pilot as ap
from offload.launch.windows import decision_v2 as d
from offload.launch.windows import dist_gate as dg
from offload.launch.windows.tests.test_dist_gate import FakeDist, materials
from offload.prefill_maxlen.evidence import Json, number, record

CELLS = ('seeded-short-b1', 'seeded-32k-b1', 'seeded-32k-b3')


BRIDGE = {'bench2-legal-short': ('PUBLIC bench2 legal instruction', 124),
          'bench2-summary-32k': ('PUBLIC bench2 summary record', 32000)}
TOKENIZER = Path('/opt/dsv41/data/d2/tokenizer/tokenizer.json')
BENCH_DATA = Path('/opt/dsv41/data/bench-data')


@pytest.fixture(autouse=True)
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    register_fake_texts(monkeypatch)
    monkeypatch.setattr(wc, 'PROMPT_SHA256', {**wc.PROMPT_SHA256, **{k: hashlib.sha256(t.encode()).hexdigest()
                                                                       for k, (t, _) in BRIDGE.items()}})


class BridgeFake(SeededFake):
    @staticmethod
    def _tokens(body: dict[str, Json]) -> int:
        messages = body['messages']
        assert isinstance(messages, list)
        return {t[0]: t[1] for t in [*texts().values(), *BRIDGE.values()]}[str(record(messages[0])['content'])]


@pytest.fixture(scope='module')
def setup() -> tuple[str, dict[str, Json]]:
    text = materials()
    prompts = dg.selected(['short'])[:3]
    return text, record(dg.cuts(FakeDist(), text, {'x': 'y'}, prompts)['prompts'])


def _start(tmp_path: Path, setup: tuple[str, dict[str, Json]], seeded: SeededFake, dist: FakeDist,
           start: str = 'B') -> dict[str, Json]:
    text, rows = setup
    return ap.run(seeded, dist, texts(), plan(tmp_path), text, rows, cells=CELLS, start=start,
                  bridge_texts=BRIDGE if start == 'A' else None)


def test_identical_starts_pass_and_give_the_band_input(tmp_path: Path, setup: tuple[str, dict[str, Json]]) -> None:
    a = _start(tmp_path, setup, SeededFake(), FakeDist())
    b = _start(tmp_path, setup, SeededFake(speed=1.01), FakeDist(shift=0.001))
    r = ap.compare(a, b)
    assert r['verdict'] == 'PASS'
    cells = record(r['cells'])
    short = record(cells['seeded-short-b1'])
    assert short['identical'] is True and short['pairs_same_rounds'] == 6 and short['pairs_same_output'] == 6
    tps = record(record(short['metrics'])['decode_tps'])
    assert number(tps['mean_diff']) == pytest.approx(0.01) and number(tps['diff_cv']) < 1e-9
    band = dg.register_band([record(r['dist'])])
    assert band['positions'] == 3 * dg.POSITIONS and band['aa_comparisons'] == 1


def test_b1_divergence_fails_but_multi_divergence_is_reported(tmp_path: Path, setup: tuple[str, dict[str, Json]]) -> None:
    a = _start(tmp_path, setup, SeededFake(), FakeDist())
    b = _start(tmp_path, setup, SeededFake(shift=1), FakeDist())
    r = ap.compare(a, b)
    assert r['verdict'] == 'FAIL' and 'B1 seeded cell differs' in str(r['reason'])
    multi_only = {**a, 'cells': {k: v for k, v in record(a['cells']).items() if k == 'seeded-32k-b3'}}
    multi_b = {**b, 'cells': {k: v for k, v in record(b['cells']).items() if k == 'seeded-32k-b3'}}
    assert ap.compare(multi_only, multi_b)['verdict'] == 'FAIL'                 # no B1 cell ran: cannot PASS
    with pytest.raises(ValueError, match='different pilot cells'):
        ap.compare(a, multi_b)


def test_cli_dry_plan_compare_and_coordinator_gate(tmp_path: Path, setup: tuple[str, dict[str, Json]],
                                                   monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    common = ['--url', wc.URL, '--cuts', 'c', '--dist-cuts', 'd', '--salt', 's', '--deadline-utc', '1', '--out',
              str(tmp_path / 'x.json'), '--start', 'A']
    assert ap.main(['run', *common]) == 0 and 'DRY RUN' in capsys.readouterr().out
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert ap.main(['run', '--execute', *common]) == 2
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(_start(tmp_path, setup, SeededFake(), FakeDist())))
    b.write_text(json.dumps(_start(tmp_path, setup, SeededFake(), FakeDist())))
    assert ap.main(['compare', str(a), str(b), '--out', str(tmp_path / 'cmp.json')]) == 0
    assert json.loads((tmp_path / 'cmp.json').read_text())['verdict'] == 'PASS'


# ---------------------------------------------------------------- DP-2 bridging cell (start A only, report only)

def test_start_a_runs_the_dp2_bridge_and_compare_ignores_it(tmp_path: Path, setup: tuple[str, dict[str, Json]]) -> None:
    api = BridgeFake()
    a = _start(tmp_path, setup, api, FakeDist(), start='A')
    b = _start(tmp_path, setup, SeededFake(), FakeDist())
    assert a['start'] == 'A' and 'bridge' not in b and ap.compare(a, b)['verdict'] == 'PASS'
    bridge = record(a['bridge'])
    assert list(bridge) == [c.name for c in ap.BRIDGE_CELLS]
    for cell in ap.BRIDGE_CELLS:
        r = record(bridge[cell.name])
        assert r['report_only'] is True and r['max_tokens'] == cell.max_tokens and len(cast(list[Json], r['pairs'])) == 3
        assert r['prompt_sha256'] == wc.PROMPT_SHA256[cell.prompts[0]]
    measured = [b for b in api.bodies if b['max_tokens'] in (512, 1500)]
    assert len(measured) == 6 * 3 and all(b['temperature'] == 1.0 and b['priority'] == 10 for b in measured)
    assert sorted({int(str(b['max_tokens'])) for b in measured}) == [512, 1500]
    report = record(a['bridge_report'])
    assert set(record(report['length_vs_date'])) == {'short', '32k-0'}
    assert 'consistent with prompt identity' in str(report['h4'])
    with pytest.raises(ValueError, match='start A only'):
        ap.run(SeededFake(), FakeDist(), texts(), plan(tmp_path), setup[0], setup[1], cells=CELLS, start='A')
    with pytest.raises(ValueError, match='start A only'):
        ap.run(SeededFake(), FakeDist(), texts(), plan(tmp_path), setup[0], setup[1], cells=CELLS, start='B',
               bridge_texts=BRIDGE)


def test_bridge_cells_are_report_only_and_never_decide(tmp_path: Path) -> None:
    r = wc.run_seeded_cell(BridgeFake(), ap.BRIDGE_CELLS[0], {**texts(), **BRIDGE}, plan(tmp_path), sleep=lambda s: None)
    with pytest.raises(ValueError, match='report-only'):
        d.seeded_series('accepted_per_round', True, r, r)
    assert sum(c.estimate_s for c in ap.BRIDGE_CELLS) / 60 < 10                       # DP-2: about 8-10 minutes


def _bridge_results(summary: float) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for cell in ap.BRIDGE_CELLS:
        value = summary if cell.name == 'bridge-bench2-summary32k-1500' else ap.DP2_REFERENCE[cell.name][0]
        out[cell.name] = {'pairs': [{'accepted_per_round': value} for _ in range(3)]}
    return out


@pytest.mark.parametrize('summary,flag', [(1.06, 'H4 FLAG'), (1.30, 'H4 FLAG'), (1.35, 'consistent'), (1.59, 'consistent')])
def test_bridge_report_flags_a_production_change(summary: float, flag: str) -> None:
    """DP-2 §5: a summary-32K reading near 1.06 instead of 1.589 would mean a production change, not prompt identity."""
    report = ap.bridge_report(_bridge_results(summary))
    assert flag in str(report['h4'])
    cells = record(report['cells'])
    assert record(cells['bridge-bench2-legal-1500'])['within_dp2_ci'] is True
    short = record(record(report['length_vs_date'])['short'])
    assert short['same_day_512_minus_1500'] == pytest.approx(1.294 - 1.146)


def test_bench2_texts_refuses_a_different_prompt(tmp_path: Path) -> None:
    class Words:
        def tokens(self, text: str) -> int:
            return len(text.split()) + 30

        def raw_tokens(self, text: str) -> int:
            return len(text.split())
    data = tmp_path / 'bench-data'
    data.mkdir()
    for name in ('opinions_cl.jsonl', 'filings_recap.jsonl'):
        (data / name).write_text('\n'.join(json.dumps({'title': f't{i}', 'court': 'c', 'date': 'd',
                                                        'text': 'word ' * 4000}) for i in range(12)) + '\n')
    with pytest.raises(ValueError, match='registered'):
        ap.bench2_texts(Words(), data)


def test_bench2_workloads_are_read_from_bench2_and_hash_to_dp2s_legal_prompt() -> None:
    legal = ap.bench2_workloads()['legal']
    assert legal.startswith('Draft a memorandum') and hashlib.sha256(legal.encode()).hexdigest().startswith('dbc88a4f5274')


@pytest.mark.skipif(not (TOKENIZER.is_file() and BENCH_DATA.is_dir()), reason='Controller tokenizer and bench data only')
def test_bench2_port_rebuilds_dp2s_summary_32k_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline proof (local DeepSeek tokenizer stands in for /tokenize; the chat-template overhead is pinned by DP-2's
    124-token legal-short count): the port of bench2 build_real/make_prompt gives DP-2's registered summary-32K prompt."""
    monkeypatch.undo()                                                     # the real registry
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))

    def n(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)
    overhead = 124 - n(ap.bench2_workloads()['legal'])

    class Local:
        def tokens(self, text: str) -> int:
            return n(text) + overhead

        def raw_tokens(self, text: str) -> int:
            return n(text)
    out = ap.bench2_texts(Local(), BENCH_DATA)
    assert out['bench2-summary-32k'][1] == 32000 and out['bench2-legal-short'][1] == 124
