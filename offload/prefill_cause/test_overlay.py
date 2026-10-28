"""PF1-CAUSE MoE spans: OFF exact, ON strips back to the original AST, real routed() unchanged at the kernel boundary."""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from offload.prefill_cause import build_overlay as bo
from offload.prefill_trace import runtime as rt
from offload.prefill_trace.test_trace import FakeBackend, batch

K1 = bo.K1


def test_bases_are_the_k1_manifest_overlay_bytes() -> None:
    pins = json.loads((bo.HERE / 'BASES.json').read_text())
    manifest = {f['path']: f['overlay_sha256'] for f in json.loads((bo.REPO / 'offload/kvpack/MANIFEST.json').read_text())['files']}
    assert set(pins) == set(bo.FILES)
    for path in bo.FILES:
        assert pins[path] == manifest[path] == bo.sha((K1 / path).read_bytes())


def test_off_is_exact_on_compiles_and_outputs_are_guarded(tmp_path: Path) -> None:
    off = bo.build(tmp_path / 'off')
    assert set(off) == set(bo.FILES)
    for path in bo.FILES:
        assert (tmp_path / 'off' / path).read_bytes() == (K1 / path).read_bytes()
    on = bo.build(tmp_path / 'on', True)
    assert set(on) == {*bo.FILES, bo.HELPER}
    for path in on:
        compile((tmp_path / 'on' / path).read_text(), path, 'exec')
    pins = {path: digest for digest, path in (line.split() for line in (bo.HERE / 'OVERLAY.sha256').read_text().splitlines())}
    assert pins == on, 'OVERLAY.sha256 must pin the reviewed ON output'
    with pytest.raises(ValueError, match='different output'):
        bo.build(tmp_path / 'off', True)
    for forbidden in (K1 / 'x', bo.REPO / 'offload/prefill_trace/generated/x'):
        with pytest.raises(ValueError, match='must not modify'):
            bo.build(forbidden, True)


def test_foreign_base_and_lost_anchor_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = 'vllm_exl3_moe/tf_experts.py'
    text = (K1 / path).read_text()
    with pytest.raises(ValueError, match='ext.rot_in found 0 times'):
        bo.instrument(text.replace('ext.rot_in(', 'ext.rotate_in('), path)
    with pytest.raises(ValueError, match='T.routed found 1 times'):
        p = 'vllm_exl3_moe/tf_prompt.py'
        source = (K1 / p).read_text()
        bo.instrument(source.replace('return T.routed(', 'return T.routed_other(', 1), p)
    fake = tmp_path / 'k1'
    for f in bo.FILES:
        (fake / f).parent.mkdir(parents=True, exist_ok=True)
        (fake / f).write_bytes((K1 / f).read_bytes() + (b'\n' if f == path else b''))
    monkeypatch.setattr(bo, 'K1', fake)
    with pytest.raises(ValueError, match='base SHA mismatch'):
        bo.build(tmp_path / 'out', True)


class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module == 'vllm.v1.worker.gpu' and {n.name for n in node.names} & {'dsv41_prefill_trace', 'dsv41_prefill_cause'}:
            return None
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if ast.unparse(node.func) == '_pf.call':
            return cast(ast.expr, self.visit(ast.Call(func=node.args[2], args=node.args[3:], keywords=node.keywords)))
        return cast(ast.expr, self.generic_visit(node))


@pytest.mark.parametrize('path', bo.FILES)
def test_removing_spans_restores_original_ast(path: str) -> None:
    source = (K1 / path).read_text()
    instrumented = bo.instrument(source, path)
    stripped = Strip().visit(ast.parse(instrumented))
    assert ast.dump(stripped, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    # metadata is side-effect free: only literals, attribute/shape reads, data_ptr() and the helper's alloc()
    for node in ast.walk(ast.parse(instrumented)):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == '_pf.call':
            meta = node.args[1]
            calls = {ast.unparse(c.func) for c in ast.walk(meta) if isinstance(c, ast.Call)}
            assert calls <= {'_pc.alloc', 's.xd.data_ptr', 's.xg.data_ptr', 'out.data_ptr'}, calls


def test_decoder_layer_labels_and_only_forward_is_touched() -> None:
    path = bo.FILES[0]
    text = bo.instrument((K1 / path).read_text(), path)
    labels = [ast.literal_eval(n.args[0]) for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.Call) and ast.unparse(n.func) == '_pf.call']
    assert sorted(labels) == ['attn.total', 'mhc.ffn_pre', 'moe.ffn', 'moe.ffn_unfinalized']
    assert text.count('_pf.call(') == 4 and 'def write_kv' in text


# ------------------------------------------------------------------ the real routed() against a recording stub

class FakeExt:
    """Records every launch with its arguments' shapes/dtypes/identity and writes deterministic outputs."""

    def __init__(self) -> None:
        self.log: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def launch(*args: object) -> None:
            sig = tuple((tuple(a.shape), str(a.dtype), a.stride()) if isinstance(a, torch.Tensor) else a for a in args)
            self.log.append((name, sig))
            if name == 'down_combine':
                out = cast(torch.Tensor, args[6])
                out.copy_(torch.arange(out.numel(), dtype=out.dtype).view_as(out))
        return launch


def stub_vllm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    names = ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu')
    mods = {n: types.ModuleType(n) for n in names}
    for name, mod in mods.items():
        setattr(mod, '__path__', [])
        monkeypatch.setitem(sys.modules, name, mod)
    gpu = mods['vllm.v1.worker.gpu']
    setattr(gpu, 'dsv41_prefill_trace', rt)
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu.dsv41_prefill_trace', rt)
    helper = types.ModuleType('vllm.v1.worker.gpu.dsv41_prefill_cause')
    exec(compile((bo.HERE / 'runtime.py').read_text(), 'dsv41_prefill_cause.py', 'exec'), helper.__dict__)
    setattr(gpu, 'dsv41_prefill_cause', helper)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    return helper


def load(monkeypatch: pytest.MonkeyPatch, pkg: str, instrumented: bool) -> tuple[types.ModuleType, types.ModuleType]:
    package = types.ModuleType(pkg)
    setattr(package, '__path__', [])
    monkeypatch.setitem(sys.modules, pkg, package)
    mods = []
    for short in ('tf_experts', 'tf_prompt'):
        path = f'vllm_exl3_moe/{short}.py'
        text = (K1 / path).read_text()
        mod = types.ModuleType(f'{pkg}.{short}')
        mod.__package__ = pkg
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        setattr(package, short, mod)
        exec(compile(bo.instrument(text, path) if instrumented else text, path, 'exec'), mod.__dict__)
        mods.append(mod)
    return mods[0], mods[1]


def bank(E: int = 8, D: int = 256, I: int = 128) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(7)
    w13 = torch.randint(-100, 100, (E, 2, D // 16, I // 16, 64), dtype=torch.int16, generator=gen)
    w2 = torch.randint(-100, 100, (E, I // 16, D // 16, 64), dtype=torch.int16, generator=gen)
    t = {'w13_trellis': w13, 'w13_suh': torch.ones(E, 2, D, dtype=torch.float16),
         'w13_svh': torch.ones(E, 2, I, dtype=torch.float16), 'w2_suh': torch.ones(E, I, dtype=torch.float16),
         'w2_svh': torch.ones(E, D, dtype=torch.float16)}
    views = lambda p: ((w13[p, 0],), (w13[p, 1],), (w2[p],))  # noqa: E731
    return SimpleNamespace(t=t, num_experts=E, views=views)


def run(monkeypatch: pytest.MonkeyPatch, instrumented: bool, tag: str, *, transient: bool = True) -> tuple[torch.Tensor, FakeExt]:
    T, P = load(monkeypatch, f'pfc_{tag}', instrumented)
    fake = FakeExt()
    monkeypatch.setattr(T, '_ext', lambda: fake)
    monkeypatch.setattr(P, 'TRANSIENT', transient)    # T0 holds scratch instead of allocating per call
    monkeypatch.setattr(T, 'TRANSIENT_ALLOC', transient)
    monkeypatch.setattr(T, 'ALIAS_ROT_DOWN', transient)
    m, k, E = 96, 2, 8
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(m, 256, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, E, (m, k), generator=gen, dtype=torch.int32)
    ids[5, 1] = -1
    w = torch.rand(m, k, generator=gen)
    out = P.routed(x, w, ids, torch.arange(E), bank(), 0.0, False, True)
    return out, fake


def test_real_routed_is_unchanged_at_the_kernel_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = stub_vllm(monkeypatch)
    rec = rt.Recorder(tmp_path / 'r1.jsonl', FakeBackend(1))
    monkeypatch.setattr(rt, '_recorder', rec)
    monkeypatch.setattr(rt, 'ENABLED', True)
    monkeypatch.setenv('VLLM_DSV41_PREFILL_TRACE_RUN', 'cpu')
    stats = {'num_alloc_retries': 0, 'num_device_alloc': 11, 'num_device_free': 2, 'reserved_bytes.all.current': 5 << 30,
             'active_bytes.all.current': 4 << 30, 'segment.all.current': 9}
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: stats)
    expected, original = run(monkeypatch, False, 'orig')
    rt.begin(batch(), False, 'NONE')          # an open prefill chunk, as in an ON arm
    got, traced = run(monkeypatch, True, 'traced')
    assert torch.equal(got, expected)
    assert traced.log == original.log and [n for n, _ in original.log] == [
        'group_count', 'group_place', 'rot_in', 'work_list', 'grouped_rows', 'gateup_epilogue', 'grouped_rows',
        'down_combine']
    assert rec.whole is not None
    rec.whole.__exit__(None, None, None)
    cast(FakeBackend, rec.backend).finish()
    rec.drain()
    rows = [json.loads(line) for line in rec.path.read_text().splitlines()]
    phases = [r['phase'] for r in rows if r['phase'] not in ('chunk.begin', 'chunk.total')]
    assert phases == ['tf.group_count', 'tf.group_place', 'tf.rot_in', 'tf.work_list', 'tf.grouped_rows.gateup',
                      'tf.gateup_epilogue', 'tf.grouped_rows.down', 'tf.down_combine', 'moe.routed.tf']
    tf = next(r for r in rows if r['phase'] == 'moe.routed.tf')
    assert tf['rows'] == 96 and tf['slots'] == 2 and tf['device_allocs'] == 11 and tf['reserved_bytes'] == 5 << 30
    assert isinstance(tf['xd_ptr'], int) and all('cuda_ms' in r for r in rows if r['phase'] != 'chunk.begin')
    assert all(r['rows'] == 96 and r['slots_total'] == 192 for r in rows if r['phase'].startswith('tf.'))
    # outside an open chunk (decode, capture, OFF env) the helper reads nothing and the spans pass straight through
    rec.context = None
    calls: list[int] = []

    def counted() -> dict[str, int]:
        calls.append(1)
        return stats
    monkeypatch.setattr(torch.cuda, 'memory_stats', counted)
    assert helper.alloc(rows=1) == {} and not calls
    again, _ = run(monkeypatch, True, 'idle')
    assert torch.equal(again, expected) and not calls
    monkeypatch.setattr(rt, 'ENABLED', False)
    assert helper.alloc(rows=1) == {}


def test_t0_spans_preserve_full_kernel_boundary_with_persistent_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_vllm(monkeypatch)
    monkeypatch.setattr(rt, 'ENABLED', False)
    original, launches = run(monkeypatch, False, 't0_original', transient=False)
    traced, traced_launches = run(monkeypatch, True, 't0_traced', transient=False)
    # Within T0, every argument descriptor and launch is identical with/without spans.
    assert traced_launches.log == launches.log
    assert torch.equal(traced, original)
    # Unlike per-call scratch, T0 retains capacity rounded up to 2048 rows (2 routed slots).
    # Full tensor shapes therefore are NOT an across-switch equality invariant; logical scalar bounds are.
    _, transient_launches = run(monkeypatch, False, 'transient_control')
    assert [name for name, _ in launches.log] == [name for name, _ in transient_launches.log]
    for (_, held), (_, per_call) in zip(launches.log, transient_launches.log):
        assert [arg for arg in held if not isinstance(arg, tuple)] == [arg for arg in per_call if not isinstance(arg, tuple)]
    held_rot = next(args for name, args in launches.log if name == 'rot_in')
    transient_rot = next(args for name, args in transient_launches.log if name == 'rot_in')
    assert held_rot[5] == ((4096, 256), 'torch.float16', (256, 1))
    assert transient_rot[5] == ((192, 256), 'torch.float16', (256, 1))
