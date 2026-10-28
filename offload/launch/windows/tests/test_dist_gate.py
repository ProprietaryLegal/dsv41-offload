"""WIN2-R3 distributional gate: registered prompts, fixed-position probes, KL/top-1 compare, A/A band, gate (CPU only)."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import dist_gate as dg
from offload.prefill_maxlen.evidence import Json, number, record

def materials() -> str:
    from offload.prefill_trace.window_requests import HEADER
    words = ' '.join(f'w{i % 997}' for i in range(2_600_000))          # > 10 M characters of synthetic public text
    return HEADER + words


class FakeDist:
    """1 token = 4 characters; top-20 depends on the prefix only; `shift` moves that much probability from the top token
    to the second (a candidate change); `flip` swaps the top-1 at positions whose prefix hash is divisible by `flip`."""
    def __init__(self, shift: float = 0.0, flip: int = 0) -> None:
        self.shift, self.flip = shift, flip
        self.probes = 0
        self.salts: set[str] = set()

    def model(self) -> dict[str, Json]:
        return {'id': 'fixture', 'max_model_len': wc.CAP}

    def token_ids(self, text: str) -> list[int]:
        return [ord(c) for c in text[::4]]

    def top20(self, prefix: Sequence[int], salt: str) -> dict[str, Json]:
        self.probes += 1
        self.salts.add(salt)
        h = int(hashlib.sha256(json.dumps(list(prefix[-50:])).encode()).hexdigest()[:8], 16)
        probs = [0.5 - self.shift, 0.3 + self.shift] + [0.19 / 18] * 18
        if self.flip and h % self.flip == 0:
            probs[0], probs[1] = probs[1], probs[0]
        return {f'token_id:{100 + (h + i) % 5000}': math.log(p) for i, p in enumerate(probs)}


def test_registered_prompt_set() -> None:
    targets = [p.target for p in dg.DIST_PROMPTS]
    assert len(dg.DIST_PROMPTS) == 64 and targets.count(1024) == 32 and targets.count(32000) == 24
    assert targets.count(150000) == 8 and len({p.key for p in dg.DIST_PROMPTS}) == 64
    assert len(dg.selected(['short', '32k'])) == 56
    with pytest.raises(ValueError):
        dg.selected(['huge'])


@pytest.fixture(scope='module')
def run() -> dict[str, Json]:
    text = materials()
    prompts = dg.selected(['short'])[:4] + dg.selected(['32k'])[:2]
    api = FakeDist()
    cut = dg.cuts(api, text, {'x': 'y'}, prompts)
    rows = record(cut['prompts'])
    control = dg.probe(api, text, rows, 'W-c')
    return {'materials': text, 'rows': rows, 'control': control, 'probes': api.probes}


def test_probe_records_eight_fixed_positions_without_text(run: dict[str, Json]) -> None:
    raw = record(run['control'])['positions']
    assert isinstance(raw, list)
    positions = [record(p) for p in raw]
    assert len(positions) == 6 * dg.POSITIONS and run['probes'] == 6 * dg.POSITIONS
    assert [p['k'] for p in positions[:8]] == list(range(8))
    assert 'w1' not in json.dumps(run['control'])                          # no corpus or generated text


def test_compare_identical_and_shifted(run: dict[str, Json]) -> None:
    text, rows = str(run['materials']), record(run['rows'])
    control = record(run['control'])
    same = dg.compare(control, dg.probe(FakeDist(), text, rows, 'W-x'))
    assert same['mean_kl'] == 0.0 and same['top1_agree'] == same['positions'] == 48
    shifted = dg.compare(control, dg.probe(FakeDist(shift=0.08), text, rows, 'W-y'))
    assert number(shifted['mean_kl']) > 0.01 and shifted['top1_agree'] == 48
    flipped = dg.compare(control, dg.probe(FakeDist(flip=3), text, rows, 'W-z'))
    assert number(flipped['top1_agree']) < 48
    with pytest.raises(ValueError, match='different prompts'):
        dg.compare(control, {'positions': record(dg.probe(FakeDist(), text, {'k': rows[next(iter(rows))]}, 's'))['positions']})


def test_band_and_gate(run: dict[str, Json], tmp_path: Path) -> None:
    text, rows = str(run['materials']), record(run['rows'])
    control = record(run['control'])
    aa = dg.compare(control, dg.probe(FakeDist(shift=0.002), text, rows, 'W-aa'))
    band = dg.register_band([aa])
    assert number(band['kl_ceiling']) == pytest.approx(1.5 * number(band['aa_mean_kl_p99'])) and band['top1_min'] == 48
    assert dg.gate(dg.compare(control, dg.probe(FakeDist(shift=0.002), text, rows, 'W-n')), band)['verdict'] == 'PASS'
    bad = dg.gate(dg.compare(control, dg.probe(FakeDist(shift=0.08), text, rows, 'W-b')), band)
    assert bad['verdict'] == 'LOSS' and 'mean KL' in str(bad['reasons'])
    flips = dg.gate(dg.compare(control, dg.probe(FakeDist(shift=0.002, flip=3), text, rows, 'W-f')), band)
    assert flips['verdict'] == 'LOSS' and 'top-1' in str(flips['reasons'])
    tampered = {**band, 'kl_ceiling': 10.0}
    with pytest.raises(ValueError, match='changed after registration'):
        dg.gate(aa, tampered)
    with pytest.raises(ValueError, match='different position sets'):
        dg.gate({**aa, 'positions': 8}, band)


def test_cli_compare_band_gate(run: dict[str, Json], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text, rows = str(run['materials']), record(run['rows'])
    paths = {}
    for name, api in (('c', FakeDist()), ('aa', FakeDist(shift=0.002)), ('x', FakeDist(shift=0.08))):
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(dg.probe(api, text, rows, f'W-{name}')))
    assert dg.main(['compare', str(paths['c']), str(paths['aa']), '--output', str(tmp_path / 'aa-cmp.json')]) == 0
    assert dg.main(['band', str(tmp_path / 'aa-cmp.json'), '--output', str(tmp_path / 'band.json')]) == 0
    assert dg.main(['compare', str(paths['c']), str(paths['x']), '--output', str(tmp_path / 'x-cmp.json')]) == 0
    assert dg.main(['gate', str(tmp_path / 'x-cmp.json'), '--band', str(tmp_path / 'band.json')]) == 2
    assert dg.main(['probe', '--cuts', 'x', '--salt', 's', '--output', str(tmp_path / 'p.json')]) == 0   # dry
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert dg.main(['probe', '--execute', '--cuts', 'x', '--salt', 's', '--output', str(tmp_path / 'p.json')]) == 2


def test_top1_test_is_exact_and_calibrated() -> None:
    assert dg.top1_p(0, 5, 1) == 1.0
    assert dg.top1_p(7, 0, 1) == pytest.approx(0.5 ** 7)            # 7 flips against none: p 0.0078 -> LOSS
    assert dg.top1_p(6, 0, 1) == pytest.approx(0.5 ** 6)            # 6 against none: 0.0156 -> PASS
    assert dg.top1_p(5, 3, 1) > 0.1                                  # literal F-METHOD rule would call this a LOSS
    assert dg.top1_p(4, 0, 3) == pytest.approx(0.25 ** 4)            # three A/A comparisons: q = 1/4
    rng = __import__('random').Random(5)
    rate, trials, losses = 3 / 512, 3000, 0
    for _ in range(trials):
        aa = sum(rng.random() < rate for _ in range(512))
        cand = sum(rng.random() < rate for _ in range(512))
        losses += dg.top1_p(cand, aa, 1) < dg.TOP1_ALPHA
    assert losses / trials < 0.012                                   # no-effect false LOSS <= ~1 %
    with pytest.raises(ValueError):
        dg.top1_p(-1, 0, 1)


def test_low_common_mass_positions_can_lose(run: dict[str, Json]) -> None:
    text, rows = str(run['materials']), record(run['rows'])
    control = record(run['control'])
    aa = dg.compare(control, dg.probe(FakeDist(shift=0.002), text, rows, 'W-aa2'))
    band = dg.register_band([aa])
    assert band['aa_low_common_mass'] == 0
    shifted = dict(dg.compare(control, dg.probe(FakeDist(shift=0.002), text, rows, 'W-lc')))
    shifted['low_common_mass_positions'] = 9
    g = dg.gate(shifted, band)
    assert g['verdict'] == 'LOSS' and 'common mass' in str(g['reasons'])
