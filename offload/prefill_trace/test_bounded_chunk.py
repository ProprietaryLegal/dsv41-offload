from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol, cast

import pytest

from offload.prefill_trace.bounded_chunk import CostModel, propose


class Window(Protocol):
    lo: int | None
    hi: int | None
    def apply(self, first_pos: int, n_rows: int, anchor_pos: int) -> None: ...
    def rows_for_anchor(self, anchor_pos: int) -> int: ...


def test_disabled_choice_is_unchanged() -> None:
    model = CostModel(1, .1, .02)
    for upstream in (0, 1, 64, 128, 129, 2048, 4096):
        assert propose(upstream, 500000, 0, 4096, 0, model).tokens == upstream


def test_prefix_sensitive_budget_and_floor_not_zero() -> None:
    model = CostModel(1, .1, .08)
    short = propose(4096, 500000, 0, 4096, 100, model, enabled=True)
    long = propose(4096, 500000, 450000, 4096, 100, model, enabled=True)
    assert 0 < long.tokens < short.tokens <= 4096
    floor = propose(4096, 500000, 450000, 4096, 0, model, enabled=True, minimum_tokens=32)
    assert floor.tokens == 32 and floor.exceeds_soft_budget
    # Persistent 0ms estimates do not hold admitted requests indefinitely.
    positions = [0, 200000, 450000]
    for _ in range(7):
        for i, pos in enumerate(positions):
            d = propose(4096, 500000-pos, pos, 4096, 0, model, enabled=True, minimum_tokens=32)
            positions[i] += d.tokens
    assert positions == [224, 200224, 450224]


def test_0075_successor_floor_and_no_synthetic_zero() -> None:
    path = Path('offload/upstream_adopt/overlay/vllm/v1/core/sched/scheduler.py')
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                and n.name == '_dsv41_replay_min_chunk')
    scope: dict[str, object] = {'_DSV41_REPLAY_SKIP': True}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual 0075 rule>', 'exec'), scope)
    class Rule(Protocol):
        def __call__(self, new: int, remaining: int, window: int, mm: bool, enabled: bool) -> int: ...
    rule = cast(Rule, scope[node.name])
    assert [rule(64, 10000, 128, False, True) for _ in range(8)] == [64] * 8
    model = CostModel(1, .1, .08)
    d = propose(4096, 10000, 450000, 4096, 0, model, enabled=True, must_trim_successor=True)
    assert d.tokens == 129 and d.exceeds_soft_budget
    with pytest.raises(ValueError, match='floor unavailable'):
        propose(64, 10000, 450000, 64, 1, model, enabled=True, must_trim_successor=True)
    with pytest.raises(ValueError, match='invalid replay obligation'):
        propose(64, 64, 450000, 64, 1, model, enabled=True, must_trim_successor=True)


def test_actual_ring_never_fabricates_rollback_history() -> None:
    # Extract the actual production range implementation without importing the engine.
    source = Path('offload/dspark/spark/remote_dspark/state.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'WindowRange')
    prefix = ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass\n').body
    scope: dict[str, object] = {}
    exec(compile(ast.Module(body=prefix + [node], type_ignores=[]), '<real ring>', 'exec'), scope)
    class Factory(Protocol):
        def __call__(self, window: int) -> Window: ...
    ring = cast(Factory, scope['WindowRange'])(128)
    ring.apply(3968, 128, 4096)
    ring.apply(8191, 1, 8192)  # skipped gap must reset the real history
    assert ring.rows_for_anchor(8192) == 1
    ring.apply(8180, 0, 8180)  # pure rollback cannot recreate 127 missing taps
    assert ring.rows_for_anchor(8180) == 0
    ring.apply(8064, 128, 8192)  # only an actual full-suffix replay restores it
    assert ring.rows_for_anchor(8192) == 128


@pytest.mark.parametrize('values', [(float('nan'), .1, .1), (1, -.1, .1), (1, 0, 0)])
def test_bad_model_refuses(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        CostModel(*values)
