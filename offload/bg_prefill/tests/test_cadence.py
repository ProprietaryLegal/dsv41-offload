"""Keep deployed 0031 cadence calls and SO2 calls in phase (S28 O3)."""
from __future__ import annotations

import ast
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from offload.bg_prefill.build_overlay import IMAGE
from offload.bg_prefill.cpu import engine
from offload.bg_prefill.tests.test_scheduler import add, deliver


def production_cadence() -> Callable[[SimpleNamespace], bool]:
    tree = ast.parse((IMAGE / 'vllm/v1/engine/core.py').read_text())
    core = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'EngineCore')
    # Every installed schedule call must advance the cadence exactly once via
    # its argument. Catch a new call site that bypasses _should_throttle_prefills.
    sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and ast.unparse(node.func) == 'self.scheduler.schedule']
    assert len(sites) == 2
    for call in sites:
        assert len(call.args) == 1 and not call.keywords
        assert ast.unparse(call.args[0]) == 'self._should_throttle_prefills()'
    for name in ('step', 'step_with_batch_queue'):
        method = next(node for node in core.body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
        assert sum(node in sites for node in ast.walk(method)) == 1
    cadence = next(node for node in core.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_should_throttle_prefills')
    scope: dict[str, object] = {'_DSV41_PREFILL_EVERY_N': 24}
    # Compile the image method verbatim, without importing the GPU engine.
    exec(compile(ast.Module(body=[cadence], type_ignores=[]),
                 'so2_0031_cadence', 'exec'), scope)
    return cast(Callable[[SimpleNamespace], bool], scope[cadence.name])


@pytest.mark.parametrize('asynchronous', [False, True])
def test_0031_and_so2_counters_remain_in_phase(asynchronous: bool) -> None:
    throttle = production_cadence()
    core = SimpleNamespace()
    e, s = engine(asynchronous=asynchronous)
    add(e, s, 'fg', 0, decode=True)
    add(e, s, 'bg', 20)
    serviced: list[int] = []
    for step in range(1, 97):
        # Native dependency fallback must still advance SO2's call counter.
        fallback = 25 <= step <= 48
        s.num_prefill_lookahead = 5 if fallback else 1
        throttled = throttle(core)
        counts = dict(s.schedule(throttled).num_scheduled_tokens)
        assert core._dsv41_pf_steps == s.so2_step == step
        assert throttled == (step % 24 != 0)
        assert counts['fg'] == 1
        if not fallback:
            assert ('bg' in counts) == (not throttled)
            if 'bg' in counts:
                assert counts['bg'] == 32
                serviced.append(step)
        deliver(e, s, counts)
    assert serviced == [24, 72, 96]
    assert s.so2_stats['steps'] == 96
    assert s.so2_stats['native'] == 24
    assert s.so2_stats['mixed'] == 72
