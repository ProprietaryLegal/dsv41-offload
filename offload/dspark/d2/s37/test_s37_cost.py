"""S37 item 5: recompute FULL-RUN.md's cost sheet from the MEASURED receipt copies (S36-inputs) and stated rates."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[4]
INPUTS = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S36-inputs')
needs_real = pytest.mark.skipif(not (INPUTS/'COPY_RECEIPT.json').exists(), reason='S36-inputs absent')
FAST_S, REF_S, EVAL_S = .85, 40., .5            # brief-stated MEASURED pilot rates; amendment-4 historical eval rate
TOKENS, SPARK_H = 974303, 3.0                    # brief-stated MEASURED pass-2 generation


def doc() -> str:
    return (REPO/'offload/dspark/d2/train/FULL-RUN.md').read_text()


def table() -> dict[str, tuple[float, str]]:
    rows = {}
    for line in doc().splitlines():
        m = re.fullmatch(r'\| (.+?) \| (?:<=)?([\d.]+) \| (.+?) \|', line)
        if m:
            rows[m[1]] = (float(m[2]), m[3])
    return rows


@needs_real
def test_training_blocks_and_updates_from_manifest() -> None:
    m = json.loads((INPUTS/'shards/MANIFEST.json').read_text())
    s = m['splits']
    assert {k: (v['n_seq'], v['n_rows']) for k, v in s.items()} == {
        'train': (65, 496416), 'dev': (30, 233500), 'heldout_legal': (24, 220458), 'heldout_general': (12, 40566)}
    assert sum(v['n_rows'] for v in s.values()) == 990940 and sum(v['n_span'] for v in s.values()) == 974172
    assert 990940-974172 == 131*128 and s['train']['n_rows']-s['train']['n_span'] == 65*128
    # D2Blocks: L-gen-1 blocks per sequence with L-gen > 1. Every captured sequence has >= 738 tap rows
    # (copy receipt sizes / 30720 B per bf16 15360-wide row), i.e. hundreds of generated positions, so all 65 count.
    receipt = json.loads((INPUTS/'COPY_RECEIPT.json').read_text())
    rows = [f['size']//30720 for f in receipt['files'] if f['relative'].endswith('/taps.bin')]
    assert len(rows) == 131 and min(rows) == 738 and min(rows)-128 > 6
    blocks = s['train']['n_span']-s['train']['n_seq']
    assert blocks == 488031
    updates = 3*math.ceil(blocks/128)
    assert updates == 11439 and 3*blocks == 1464093
    assert '488,031 unique training blocks' in doc() and '11,439 updates' in doc()


@needs_real
def test_gpu_hours_and_wall_time() -> None:
    header = json.loads((INPUTS/'pilot-plan-header.json').read_text())
    pop = header['sample']['population_rounds']
    general = json.loads((INPUTS/'heldout_general.json').read_text())['rounds']
    assert (pop['dev'], pop['heldout_legal'], general) == (90837, 77949, 14624)
    # G1 rounds include determinate partial tails; complete rounds are fewer (dev 90,857 -> 90,837).
    assert json.loads((INPUTS/'dev.json').read_text())['rounds'] == 90857
    fast, ref = 11439*FAST_S/3600, 11439*REF_S/3600
    dev, legal, gen = (2*n*EVAL_S/3600 for n in (pop['dev'], pop['heldout_legal'], general))
    t = table()
    expect = {'Full FAST, three epochs, one solo': fast,
              'Same full run in REF (cost comparison only; not scheduled)': ref,
              'Full dev baseline + candidate, 90,837 complete rounds each': dev,
              'Full legal baseline + candidate, 77,949 complete rounds each': legal,
              'General baseline + candidate, at most 14,624 rounds each': gen,
              'Full G2 including general': dev+legal+gen,
              'FAST followed by full G2': fast+dev+legal+gen}
    for row, value in expect.items():
        assert t[row][0] == pytest.approx(value, abs=5e-4), row
    assert max(dev, legal+gen) == pytest.approx(25.715, abs=5e-4)
    assert fast+max(dev, legal+gen) == pytest.approx(28.416, abs=5e-4)
    assert (round(fast, 3), round(dev+legal+gen, 3)) == (2.701, 50.947)


def test_extra_generation_spark_hours() -> None:
    rate = TOKENS/SPARK_H
    deficit = 1_000_000-488031
    train_share = 488096/974172
    assert deficit == 511969 and round(train_share, 3) == .501
    assert round(deficit/rate, 2) == 1.58
    assert round(deficit/train_share/rate, 2) == 3.15
    assert 1_000_000-TOKENS == 25697 and round(25697/rate, 3) == .079


def test_three_epochs_is_the_documented_recipe() -> None:
    from offload.dspark.d2.train.tests.snapshot_git import git
    old = git(REPO, 'show', 'b6f79e9:offload/dspark/d2/train/PILOT.md').stdout.decode()
    assert '--epochs 3 --batch-size 128 --chunk 8192 --rank 32' in old  # pre-lane PILOT.md step 5
    brief = (REPO/'documents/plans/lanes/D2-TRAIN.md').read_text()
    assert '3 epochs' in brief
    from offload.dspark.d2.train.loop import Run
    assert Run().epochs == 3
