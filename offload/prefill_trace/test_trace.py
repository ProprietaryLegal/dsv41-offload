from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from offload.prefill_trace import runtime as rt
from offload.prefill_trace.build_overlay import FILES, HELPER, REPO, build, instrument
from offload.prefill_maxlen import test_workspace as cpu

IMAGE = Path('/opt/dsv41/data/src-image')


class FakeEvent:
    ready = False
    recorded = 0

    def record(self) -> None:
        self.recorded += 1

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, end: rt.Event) -> float:
        assert self.ready and end.query()
        return 2.5


class FakeBackend:
    def __init__(self, rank: int = 0) -> None:
        self.events: list[FakeEvent] = []
        self.tp_rank = rank

    def event(self) -> rt.Event:
        event = FakeEvent()
        self.events.append(event)
        return event

    def rank(self) -> int:
        return self.tp_rank

    def capturing(self) -> bool:
        return False

    def finish(self) -> None:
        for event in self.events:
            event.ready = True


def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int = 0, limit: int = 8192) -> rt.Recorder:
    rec = rt.Recorder(tmp_path / f'r{rank}.jsonl', FakeBackend(rank), limit)
    monkeypatch.setattr(rt, '_recorder', rec)
    monkeypatch.setattr(rt, 'ENABLED', True)
    monkeypatch.setenv('VLLM_DSV41_PREFILL_TRACE_RUN', 'public-window')
    return rec


def rows(rec: rt.Recorder) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], [json.loads(line) for line in rec.path.read_text().splitlines()])


def batch() -> rt.Batch:
    return cast(rt.Batch, SimpleNamespace(req_ids=['request-A', 'request-B'], num_reqs=2,
        has_prefill=True, is_prefilling_np=[True, False], query_start_loc_np=[0, 4096, 4102],
        num_computed_prefill_tokens_np=[300000, 10000]))


def test_no_sync_deferred_events_bound_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(tmp_path, monkeypatch, limit=3)
    rt.begin(batch(), False, 'NONE')
    output = object()
    assert rt.call('indexer.gather', {}, lambda: output) is output
    rec.drain()
    assert all('cuda_ms' not in row for row in rows(rec))
    with pytest.raises(ValueError, match='actual failure'):
        rt.call('indexer.topk', {}, lambda: (_ for _ in ()).throw(ValueError('actual failure')))
    # A full bounded queue refuses rather than dropping data or synchronizing.
    rt.call('last-free-slot', {}, lambda: None)
    with pytest.raises(RuntimeError, match='bound exceeded'):
        rt.call('overflow', {}, lambda: None)
    cast(FakeBackend, rec.backend).finish()
    rec.drain()
    assert len(rec.pending) == 0
    assert any(row.get('failed') for row in rows(rec))
    assert all(row['cuda_ms'] == 2.5 for row in rows(rec) if 'cuda_ms' in row)
    # Close the explicit overall span and prove it too drains only when complete.
    assert rec.whole is not None
    rec.whole.__exit__(None, None, None)
    cast(FakeBackend, rec.backend).finish()
    rec.drain()


def test_two_ranks_same_identity_and_scope_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = []
    for rank in (0, 1):
        rec = install(tmp_path, monkeypatch, rank)
        @rt.execution
        def execute() -> int:
            rt.begin(batch(), False, 'NONE')
            with rt.span('collective.all_reduce', {}):
                pass
            return 7
        assert execute() == 7
        cast(FakeBackend, rec.backend).finish()
        rec.drain()
        data = rows(rec)
        ids.append(data[0]['members'])
        assert rec.context is None and rec.whole is None
        assert {row['phase'] for row in data} >= {'chunk.begin', 'chunk.end', 'chunk.total', 'collective.all_reduce'}
        assert 'request-A' not in str(data)
    assert ids[0] == ids[1]


def test_off_is_exact_and_foreign_sources_refused(tmp_path: Path) -> None:
    hashes = build(tmp_path / 'off', IMAGE)
    assert HELPER not in hashes
    for path, kind in FILES.items():
        base = (REPO / 'offload/kvpack/overlay' if kind == 'k1' else IMAGE) / path
        assert (tmp_path / 'off' / path).read_bytes() == base.read_bytes()
    build(tmp_path / 'on', IMAGE, True)
    for path in (*FILES, HELPER):
        compile((tmp_path / 'on' / path).read_text(), path, 'exec')
    with pytest.raises(ValueError, match='different output'):
        build(tmp_path / 'off', IMAGE, True)
    with pytest.raises(ValueError, match='reference/production'):
        build(IMAGE / 'must-not-write', IMAGE, True)
    with pytest.raises(FileNotFoundError):
        build(tmp_path / 'foreign-out', tmp_path / 'missing', True)


@pytest.mark.parametrize('fp4', [False, True])
def test_instrumented_actual_indexer_output_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fp4: bool) -> None:
    rec = install(tmp_path, monkeypatch)
    rt.begin(batch(), False, 'NONE')
    original = cpu.definitions
    def definitions(text: str, names: set[str]) -> dict[str, object]:
        scope = original(text, names)
        scope['_pf'] = rt
        return scope
    monkeypatch.setattr(cpu, 'definitions', definitions)
    source = (IMAGE / 'vllm/model_executor/layers/sparse_attn_indexer.py').read_text()
    expected, original_ops, _ = cpu.execute(source, False, 256, fp4)
    actual, ops, _ = cpu.execute(instrument(source, 'vllm/model_executor/layers/sparse_attn_indexer.py'), False, 256, fp4)
    assert torch.equal(actual, expected)
    for key in ('seen_values', 'seen_scales', 'logits'):
        assert all(torch.equal(a, b) for a, b in zip(getattr(original_ops, key), getattr(ops, key)))
    cast(FakeBackend, rec.backend).finish()
    rec.drain()
    data = rows(rec)
    assert sum(row['phase'] == 'indexer.gather' for row in data) == 2
    assert sum(row['phase'] == 'indexer.topk' for row in data) == 3
    assert all(cast(int, row.get('logits_elements', 0)) > 0 for row in data if row['phase'] == 'indexer.topk')


def test_actual_planner_overflow_and_query_chunking_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(tmp_path, monkeypatch)
    rt.begin(batch(), False, 'NONE')
    path = 'vllm/v1/attention/backends/mla/indexer.py'
    text = instrument((IMAGE / path).read_text(), path)
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == '_split_indexer_prefill_chunks')
    scope = cpu.definitions(ast.unparse(node), {node.name})
    scope['_pf'] = rt
    split = cast(Callable[..., list[tuple[slice, slice]]], scope[node.name])
    cases = [(torch.tensor([12, 12]), torch.tensor([4, 4]), 16, 1024),
             (torch.tensor([24]), torch.tensor([8]), 16, 96)]
    for seq, query, workspace, budget in cases:
        result = split(seq, query, workspace, budget)
        original_node = next(n for n in ast.walk(ast.parse((IMAGE / path).read_text())) if isinstance(n, ast.FunctionDef) and n.name == node.name)
        original_scope = cpu.definitions(ast.unparse(original_node), {node.name})
        expected = cast(Callable[..., list[tuple[slice, slice]]], original_scope[node.name])(seq, query, workspace, budget)
        assert result == expected
    data = rows(rec)
    assert any(row.get('workspace_group_split') for row in data)
    assert any(row.get('workspace_exceeded') for row in data)
    assert any(row.get('query_subchunked') for row in data)


def test_disabled_runtime_never_opens_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, 'ENABLED', False)
    monkeypatch.setattr(rt, '_recorder', None)
    rt.begin(batch(), False, 'NONE')
    value = object()
    assert rt.call('off', {}, lambda: value) is value
    with rt.span('off', {}):
        pass
    assert rt._recorder is None


def test_config_off_retains_exact_base_and_on_unique_mounts(tmp_path: Path) -> None:
    import os
    import subprocess
    output = tmp_path / 'overlay'
    build(output, IMAGE, True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PF1_', 'K1_'))}
    env.update(PF1_TRACE_ROOT=str(output), PF1_TRACE_RUN='cpu-test')
    def source(config: str, enabled: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', '-euc', 'source ~/.config/ndm/paths.sh\n. ' + config + '\nprintf "%s\\n" "$OVERLAYS" "$EXP_DOCKER_ENV" "$EXTRA_DOCKER_ARGS"'],
                              env={**env, 'PF1_TRACE': enabled}, cwd=REPO, text=True, capture_output=True)
    base = source('offload/launch/configs/D-kv6-async-pack.env', '0')
    off = source('offload/launch/configs/PF1-R2-trace.env', '0')
    assert base.returncode == off.returncode == 0
    assert base.stdout == off.stdout
    on = source('offload/launch/configs/PF1-R2-trace.env', '1')
    assert on.returncode == 0, on.stderr
    pairs = on.stdout.splitlines()[0].split()
    targets = [pair.split('=', 1)[1] for pair in pairs]
    assert len(targets) == len(set(targets)) == 19
    (output / HELPER).write_text('tampered')
    rejected = source('offload/launch/configs/PF1-R2-trace.env', '1')
    assert rejected.returncode != 0


def test_removing_telemetry_restores_original_ast() -> None:
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
            if node.module == 'vllm.v1.worker.gpu' and any(n.name == 'dsv41_prefill_trace' for n in node.names):
                return None
            return node

        def visit_Call(self, node: ast.Call) -> ast.expr:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == '_pf':
                if node.func.attr == 'call':
                    return cast(ast.expr, self.visit(ast.Call(func=node.args[2], args=node.args[3:], keywords=node.keywords)))
                if node.func.attr == 'plan':
                    return node.args[0]
            return cast(ast.expr, self.generic_visit(node))

        def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == '_pf.begin':
                return None
            return cast(ast.Expr, self.generic_visit(node))

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            node.decorator_list = [n for n in node.decorator_list if ast.unparse(n) != '_pf.execution']
            return cast(ast.FunctionDef, self.generic_visit(node))

        def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
            if ast.unparse(node.items[0].context_expr).startswith('_pf.span('):
                return node.body
            return cast(ast.With, self.generic_visit(node))

    for path, kind in FILES.items():
        source = ((REPO / 'offload/kvpack/overlay' if kind == 'k1' else IMAGE) / path).read_text()
        stripped = Strip().visit(ast.parse(instrument(source, path)))
        assert ast.dump(stripped, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False), path


def test_aborted_execution_marks_total_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(tmp_path, monkeypatch)
    @rt.execution
    def execute() -> None:
        rt.begin(batch(), False, 'NONE')
        raise ValueError('engine failure')
    with pytest.raises(ValueError, match='engine failure'):
        execute()
    cast(FakeBackend, rec.backend).finish()
    rec.drain()
    assert any(row.get('failed') for row in rows(rec) if row['phase'] == 'chunk.total')
    assert rec.context is None and rec.whole is None
