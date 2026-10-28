"""PF1-CAUSE2 fix candidate: OFF exact, ON = cause span positions with event records only, outputs unchanged."""
from __future__ import annotations

import ast
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from offload.prefill_cause import build_overlay as bo
from offload.prefill_cause import test_overlay as cause_test
from offload.prefill_cause2 import build_fix as bf
from offload.prefill_cause2 import fence
from offload.prefill_trace import runtime as rt
from offload.prefill_trace.test_trace import batch

K1 = bf.K1


def test_bases_are_the_cause_bases_and_the_k1_manifest_bytes() -> None:
    pins = json.loads((bf.HERE / 'BASES.json').read_text())
    assert pins == json.loads((bo.HERE / 'BASES.json').read_text())
    manifest = {f['path']: f['overlay_sha256'] for f in json.loads((bf.REPO / 'offload/kvpack/MANIFEST.json').read_text())['files']}
    for path in bf.FILES:
        assert pins[path] == manifest[path] == bo.sha((K1 / path).read_bytes())


def test_off_is_exact_on_is_pinned_and_outputs_are_guarded(tmp_path: Path) -> None:
    off = bf.build(tmp_path / 'off')
    assert set(off) == set(bf.FILES)
    for path in bf.FILES:
        assert (tmp_path / 'off' / path).read_bytes() == (K1 / path).read_bytes()
    on = bf.build(tmp_path / 'on', True)
    assert set(on) == {*bf.FILES, bf.HELPER}
    pins = {path: digest for digest, path in (line.split() for line in (bf.HERE / 'OVERLAY.sha256').read_text().splitlines())}
    assert pins == on, 'OVERLAY.sha256 must pin the reviewed ON output'
    assert (tmp_path / 'on' / bf.HELPER).read_bytes() == (bf.HERE / 'fence.py').read_bytes()
    with pytest.raises(ValueError, match='different output'):
        bf.build(tmp_path / 'off', True)
    for forbidden in (K1 / 'x', bf.REPO / 'offload/prefill_trace/generated/x', bf.REPO / 'offload/prefill_cause/generated/x'):
        with pytest.raises(ValueError, match='must not modify'):
            bf.build(forbidden, True)


def test_drifted_base_and_lost_anchor_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = 'vllm_exl3_moe/tf_experts.py'
    with pytest.raises(ValueError, match='ext.down_combine found 0 times'):
        bf.instrument((K1 / path).read_text().replace('ext.down_combine(', 'ext.down_combined('), path)
    fake = tmp_path / 'k1'
    for f in bf.FILES:
        (fake / f).parent.mkdir(parents=True, exist_ok=True)
        (fake / f).write_bytes((K1 / f).read_bytes() + (b'\n' if f == path else b''))
    monkeypatch.setattr(bf, 'K1', fake)
    with pytest.raises(ValueError, match='base SHA mismatch'):
        bf.build(tmp_path / 'out', True)


class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        names = {n.name for n in node.names}
        if node.module == 'vllm.v1.worker.gpu' and names & {'dsv41_moe_fence', 'dsv41_prefill_trace', 'dsv41_prefill_cause'}:
            return None
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        name = ast.unparse(node.func)
        if name in ('_mf.call', '_pf.call'):
            skip = 1 if name == '_mf.call' else 2
            return cast(ast.expr, self.visit(ast.Call(func=node.args[skip], args=node.args[skip + 1:],
                                                      keywords=node.keywords)))
        return cast(ast.expr, self.generic_visit(node))


def sites(text: str, wrapper: str) -> list[str]:
    skip = 1 if wrapper == '_mf.call' else 2
    found = [(n.lineno, n.col_offset, ast.unparse(n.args[skip])) for n in ast.walk(ast.parse(text))
             if isinstance(n, ast.Call) and ast.unparse(n.func) == wrapper]
    return [callee for _, _, callee in sorted(found)]


@pytest.mark.parametrize('path', bf.FILES)
def test_strip_gives_the_original_ast_and_sites_equal_the_cause_spans(path: str) -> None:
    source = (K1 / path).read_text()
    fixed = bf.instrument(source, path)
    stripped = Strip().visit(ast.parse(fixed))
    assert ast.dump(stripped, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    cause = bo.instrument(source, path)
    assert sites(fixed, '_mf.call') == sites(cause, '_pf.call') and len(sites(fixed, '_mf.call')) > 0
    for node in ast.walk(ast.parse(fixed)):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == '_mf.call':
            # the row count is a name / shape read already in scope: no call, no side effect
            assert ast.unparse(node.args[0]) == bf.ROWS[path]
            assert not any(isinstance(c, ast.Call) for c in ast.walk(node.args[0]))


# ------------------------------------------------------------------ the real routed(): fence vs cause spans vs base

class Log:
    def __init__(self) -> None:
        self.items: list[str] = []


class LoggedEvent:
    def __init__(self, log: Log) -> None:
        self.log, self.ready = log, False

    def record(self) -> None:
        self.log.items.append('event.record')

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, end: rt.Event) -> float:
        return 1.0


class LoggedBackend:
    """Both runtimes' backend: event records land in the same log as the kernel launches."""

    def __init__(self, log: Log, capturing: bool = False) -> None:
        self.log, self.capture, self.events = log, capturing, cast(list[LoggedEvent], [])

    def event(self) -> LoggedEvent:
        self.events.append(LoggedEvent(self.log))
        return self.events[-1]

    def rank(self) -> int:
        return 1

    def capturing(self) -> bool:
        return self.capture


class LoggedExt(cause_test.FakeExt):
    def __init__(self, log: Log) -> None:
        super().__init__()
        self.shared = log

    def __getattr__(self, name: str) -> Any:
        launch = cast(Callable[..., None], super().__getattr__(name))

        def logged(*args: object) -> None:
            self.shared.items.append(name)
            launch(*args)
        return logged


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    cause_test.stub_vllm(monkeypatch)
    gpu = sys.modules['vllm.v1.worker.gpu']
    setattr(gpu, 'dsv41_moe_fence', fence)
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu.dsv41_moe_fence', fence)


def load(monkeypatch: pytest.MonkeyPatch, pkg: str, how: str) -> tuple[types.ModuleType, types.ModuleType]:
    package = types.ModuleType(pkg)
    setattr(package, '__path__', [])
    monkeypatch.setitem(sys.modules, pkg, package)
    mods = []
    for short in ('tf_experts', 'tf_prompt'):
        path = f'vllm_exl3_moe/{short}.py'
        text = (K1 / path).read_text()
        text = bf.instrument(text, path) if how == 'fix' else bo.instrument(text, path) if how == 'cause' else text
        mod = types.ModuleType(f'{pkg}.{short}')
        mod.__package__ = pkg
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        setattr(package, short, mod)
        exec(compile(text, path, 'exec'), mod.__dict__)
        mods.append(mod)
    return mods[0], mods[1]


def run(monkeypatch: pytest.MonkeyPatch, how: str, tag: str, log: Log, *, transient: bool = True,
        rows: int = 96) -> tuple[torch.Tensor, LoggedExt]:
    T, P = load(monkeypatch, f'pfc2_{tag}', how)
    ext = LoggedExt(log)
    monkeypatch.setattr(T, '_ext', lambda: ext)
    monkeypatch.setattr(P, 'TRANSIENT', transient)
    monkeypatch.setattr(T, 'TRANSIENT_ALLOC', transient)
    monkeypatch.setattr(T, 'ALIAS_ROT_DOWN', transient)
    k, E = 2, 8
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(rows, 256, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, E, (rows, k), generator=gen, dtype=torch.int32)
    ids[5, 1] = -1
    w = torch.rand(rows, k, generator=gen)
    out = P.routed(x, w, ids, torch.arange(E), cause_test.bank(), 0.0, False, True)
    return out, ext


@pytest.mark.parametrize('transient', [True, False])
def test_fence_records_exactly_where_the_cause_spans_do(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         transient: bool) -> None:
    install(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: {})
    base_log, cause_log, fix_log = Log(), Log(), Log()
    monkeypatch.setattr(rt, 'ENABLED', False)
    expected, base_ext = run(monkeypatch, 'base', f'b{transient}', base_log, transient=transient)
    # cause: an open traced prefill chunk, as in window CW
    rec = rt.Recorder(tmp_path / 'r1.jsonl', LoggedBackend(cause_log))
    monkeypatch.setattr(rt, '_recorder', rec)
    monkeypatch.setattr(rt, 'ENABLED', True)
    monkeypatch.setenv('VLLM_DSV41_PREFILL_TRACE_RUN', 'cpu')
    rt.begin(batch(), False, 'NONE')
    cause_log.items.clear()                         # chunk.total's own start record is not an MoE-half record
    traced, cause_ext = run(monkeypatch, 'cause', f'c{transient}', cause_log, transient=transient)
    monkeypatch.setattr(rt, 'ENABLED', False)
    monkeypatch.setattr(rt, '_recorder', None)
    # fix: no trace runtime at all, the fence on
    monkeypatch.setattr(fence, 'ENABLED', True)
    monkeypatch.setattr(fence, 'backend', LoggedBackend(fix_log))
    fixed, fix_ext = run(monkeypatch, 'fix', f'f{transient}', fix_log, transient=transient)
    assert torch.equal(fixed, expected) and torch.equal(traced, expected)
    assert fix_ext.log == base_ext.log == cause_ext.log              # same launches, args, shapes, strides, scalars
    assert base_log.items == [name for name, _ in base_ext.log]      # production: no event record
    assert fix_log.items == cause_log.items                          # the same records at the same places
    assert fix_log.items.count('event.record') == 2 * (len(base_ext.log) + 1) == 18
    assert fix_log.items[0] == fix_log.items[1] == 'event.record' and fix_log.items[-2:] == ['event.record'] * 2


def test_fence_gates_off_below_min_rows_while_capturing_and_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    log = Log()
    monkeypatch.setattr(fence, 'backend', LoggedBackend(log))
    monkeypatch.setattr(fence, 'ENABLED', False)
    def add(a: int, b: int = 0) -> int:
        return a + b
    assert fence.call(4092, add, 1, b=2) == 3 and log.items == []
    monkeypatch.setattr(fence, 'ENABLED', True)
    assert fence.call(fence.MIN_ROWS - 1, lambda: 'decode') == 'decode' and log.items == []
    monkeypatch.setattr(fence, 'backend', LoggedBackend(log, capturing=True))
    assert fence.call(4092, lambda: 'captured') == 'captured' and log.items == []
    monkeypatch.setattr(fence, 'backend', LoggedBackend(log))
    out = object()
    assert fence.call(fence.MIN_ROWS, lambda: out) is out and log.items == ['event.record'] * 2

    def boom() -> None:
        log.items.append('launch')
        raise RuntimeError('kernel failed')
    with pytest.raises(RuntimeError, match='kernel failed'):
        fence.call(4092, boom)
    assert log.items[2:] == ['event.record', 'launch', 'event.record']
    # a callee's own `rows` / `fn` keywords pass through (positional-only wrapper parameters)
    def named(rows: int, fn: int) -> tuple[int, int]:
        return rows, fn
    assert fence.call(4092, named, rows=1, fn=2) == (1, 2)


def test_ring_bounds_retained_events(monkeypatch: pytest.MonkeyPatch) -> None:
    log = Log()
    backend = LoggedBackend(log)
    monkeypatch.setattr(fence, 'backend', backend)
    monkeypatch.setattr(fence, 'ENABLED', True)
    monkeypatch.setattr(fence, '_kept', type(fence._kept)(maxlen=3))
    for _ in range(5):
        fence.call(4092, lambda: None)
    assert len(fence._kept) == 3 and fence._kept[0][0] is backend.events[4]


def test_switch_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    monkeypatch.delenv('VLLM_DSV41_MOE_FENCE', raising=False)
    assert importlib.reload(fence).ENABLED is False
    monkeypatch.setenv('VLLM_DSV41_MOE_FENCE', '1')
    assert importlib.reload(fence).ENABLED is True
    monkeypatch.setenv('VLLM_DSV41_MOE_FENCE', '0')
    assert importlib.reload(fence).ENABLED is False
