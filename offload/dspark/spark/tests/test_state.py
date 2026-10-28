# SPDX-License-Identifier: Apache-2.0
import pytest

from remote_dspark.state import SlotTable, WindowRange

W = 128


def test_fresh_and_append():
    r = WindowRange(W)
    r.apply(0, 5, 5)
    assert (r.lo, r.hi) == (0, 4)
    r.apply(5, 3, 8)
    assert (r.lo, r.hi) == (0, 7)
    assert r.rows_for_anchor(8) == 8 and r.expected_rows(8) == 8


def test_window_clamp_long_chunk():
    r = WindowRange(W)
    r.apply(3968, 128, 4096)        # trailing 128 rows of a 4096-token chunk
    assert (r.lo, r.hi) == (3968, 4095)
    r.apply(4096, 50, 4146)         # short next chunk
    assert (r.lo, r.hi) == (4146 - W, 4145)
    assert r.rows_for_anchor(4146) == W == r.expected_rows(4146)


def test_gap_restarts_range():
    r = WindowRange(W)
    r.apply(0, 10, 10)
    r.apply(300, 128, 428)          # gap: older rows are not contiguous any more
    assert (r.lo, r.hi) == (300, 427)


def test_rollback_on_regression():
    r = WindowRange(W)
    r.apply(0, 100, 100)
    r.apply(60, 20, 80)             # re-add / replay at a lower position
    assert (r.lo, r.hi) == (0, 79)
    r.apply(0, 128, 128)            # replay from the start
    assert (r.lo, r.hi) == (0, 127)


def test_rollback_before_lo_extends_down():
    r = WindowRange(W)
    r.apply(200, 50, 250)
    r.apply(150, 60, 210)
    assert (r.lo, r.hi) == (150, 209)


def test_zero_rows_pure_rollback():
    r = WindowRange(W)
    r.apply(0, 10, 10)
    r.apply(0, 0, 7)
    assert (r.lo, r.hi) == (0, 6)
    with pytest.raises(ValueError):
        r.apply(0, 0, 9)            # cannot roll forward without rows
    r.apply(0, 0, 0)
    assert r.empty


@pytest.mark.parametrize("args", [(0, 129, 129), (5, 3, 9), (-1, 2, 1), (0, 1, -3)])
def test_invalid_entries_do_not_mutate(args):
    r = WindowRange(W)
    r.apply(0, 4, 4)
    with pytest.raises(ValueError):
        r.apply(*args)
    assert (r.lo, r.hi) == (0, 3)


def test_rows_for_anchor_partial_and_mismatch():
    r = WindowRange(W)
    r.apply(500, 10, 510)            # only 10 rows held near position 510
    assert r.rows_for_anchor(510) == 10 and r.expected_rows(510) == W
    assert r.rows_for_anchor(511) == 0   # window does not end at anchor-1
    assert list(r.positions(510)) == list(range(500, 510))


def test_slot_table_generations():
    t = SlotTable(4, W)
    assert t.begin_request(1, "a") and t[1].generation == 1
    assert not t.begin_request(1, "a")
    t[1].rng.apply(0, 5, 5)
    t[1].dirty = False
    t[1].announced = True
    assert t.begin_request(1, "b") and t[1].generation == 2
    assert t[1].rng.empty and t[1].dirty and not t[1].announced
    t[2].generation = 0xFFFFFFFF
    t.begin_request(2, "c")
    assert t[2].generation == 0
    assert t.live_slots() == [1, 2]


def test_anchor_reads_the_block_anchored_128_row_window():
    """CONTRACT 2 / 4.3 (v1.7): the DSpark block kernel gives every query token of the block
    the context [anchor_pos - 128, anchor_pos - 1] (sparse_swa.py:1253-1255), i.e. W = 128
    rows, not the causal 127. The window the worker drafts over, the rows it reports and the
    expected count all use exactly that range (guards against the off-by-one S5 pointed at)."""
    W = 128
    r = WindowRange(W)
    for first in range(0, 600, 100):
        r.apply(first, 100, first + 100)
    a = 600
    assert list(r.positions(a)) == list(range(a - 128, a))
    assert r.rows_for_anchor(a) == 128 == r.expected_rows(a)
    # near position 0 the window is all rows from 0
    r = WindowRange(W)
    r.apply(0, 57, 57)
    assert list(r.positions(57)) == list(range(57)) and r.rows_for_anchor(57) == 57 == r.expected_rows(57)
    # one row short of the 128 the kernel reads is partial, not complete
    r = WindowRange(W)
    r.apply(473, 127, 600)
    assert r.rows_for_anchor(600) == 127 < r.expected_rows(600) == 128
