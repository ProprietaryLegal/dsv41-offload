"""CPU-only behaviour of the 0085b port: the real overlay functions (extracted from the overlay files, not copies) run over
a scripted input queue, a fake executor and fake ranks with a counting torch. Covers the brief's three properties:
the idle trigger (>= 131,072 prefill tokens, then 5 s without a request), an idle collective_rpc that never raises on a
rank, and exactly one empty_cache per idle period. Switch-off is compared call for call with the image's functions."""
from __future__ import annotations

import ast
import logging
import queue
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from offload.upx import build_0085b as b

THRESHOLD = 131072


class Exhausted(Exception):
    """The script has no more requests: ends a simulated busy loop."""


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(found) == 1, name
    return found[0]


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    found = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, (cls.name, name)
    node = found[0]
    node.decorator_list = []
    return node


def _switch_nodes(tree: ast.Module, names: set[str]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            out.append(node)
    return out


def _exec(nodes: list[ast.stmt], scope: dict[str, object], filename: str) -> None:
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)


@dataclass
class Logger:
    records: list[tuple[str, str]] = field(default_factory=list)

    def isEnabledFor(self, level: int) -> bool:
        return False

    def debug(self, msg: str, *args: object) -> None:
        self.records.append(('debug', msg % args if args else msg))

    def info(self, msg: str, *args: object) -> None:
        self.records.append(('info', msg % args if args else msg))

    def warning(self, msg: str, *args: object) -> None:
        self.records.append(('warning', msg % args if args else msg))


def fake_os(env: dict[str, str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(environ=dict(env))


def engine_functions(path: Path, env: dict[str, str], overlay: bool) -> dict[str, object]:
    tree = _module(path)
    scope: dict[str, object] = {'os': fake_os(env), 'queue': queue, 'DEBUG': logging.DEBUG,
                                'logger': Logger(), 'time': types.SimpleNamespace(sleep=lambda s: None)}
    if overlay:
        _exec(_switch_nodes(tree, {'_dsv41_int_env', '_DSV41_RELEASE_IDLE', '_DSV41_RELEASE_GRACE_S'}), scope, str(path))
    cls = _class(tree, 'EngineCoreProc')
    names = ['_process_input_queue', '_process_engine_step'] + (['_dsv41_idle_release'] if overlay else [])
    _exec([_method(cls, n) for n in names], scope, str(path))
    return scope


@dataclass
class TorchCounters:
    reserved: int = 6 * 2**30
    empty_calls: int = 0
    sync_calls: int = 0
    fail_empty: bool = False
    fail_sync: bool = False


def worker_functions(env: dict[str, str], torch_state: TorchCounters, logger: Logger) -> dict[str, object]:
    path = b.OVERLAY / b.WORKER
    tree = _module(path)

    def synchronize() -> None:
        torch_state.sync_calls += 1
        if torch_state.fail_sync:
            raise RuntimeError('CUDA error: an illegal memory access was encountered')

    def empty_cache() -> None:
        torch_state.empty_calls += 1
        if torch_state.fail_empty:
            raise RuntimeError('empty_cache failed')
        torch_state.reserved = 2 * 2**30

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(synchronize=synchronize, memory_reserved=lambda: torch_state.reserved),
        accelerator=types.SimpleNamespace(empty_cache=empty_cache))
    import time as real_time
    scope: dict[str, object] = {'os': fake_os(env), 'torch': torch, 'time': real_time, 'logger': logger}
    _exec(_switch_nodes(tree, {'_dsv41_int_env', '_DSV41_RELEASE_TOKENS', '_DSV41_RELEASE_FILE', '_DSV41_RELEASE_COUNT'}),
          scope, str(path))
    worker = _class(tree, 'Worker')
    _exec([_method(worker, 'dsv41_release_if_due'), _method(worker, '_dsv41_release_if_due')], scope, str(path))
    execute = _method(worker, 'execute_model')
    count: list[ast.stmt] = [n for n in execute.body
                             if isinstance(n, ast.If) and ast.unparse(n.test) == '_DSV41_RELEASE_COUNT and forward_pass']
    assert len(count) == 1
    wrapper = ast.parse('def _count(self, scheduler_output, forward_pass):\n    pass').body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = count
    _exec([wrapper], scope, str(path))
    return scope


class Rank:
    """One TP rank: the overlay's worker methods bound to a plain object."""

    def __init__(self, env: dict[str, str], torch_state: TorchCounters | None = None) -> None:
        self.torch = torch_state or TorchCounters()
        self.log = Logger()
        self._fns = worker_functions(env, self.torch, self.log)

    def dsv41_release_if_due(self) -> None:
        fn = self._fns['dsv41_release_if_due']
        assert callable(fn)
        fn(self)

    def _dsv41_release_if_due(self) -> None:
        fn = self._fns['_dsv41_release_if_due']
        assert callable(fn)
        fn(self)

    def count(self, scheduled: dict[str, int]) -> None:
        fn = self._fns['_count']
        assert callable(fn)
        fn(self, types.SimpleNamespace(num_scheduled_tokens=scheduled), sum(scheduled.values()) > 0)


@dataclass
class Executor:
    ranks: list[Rank]
    calls: list[str] = field(default_factory=list)
    fail: bool = False

    def collective_rpc(self, method: str) -> list[None]:
        """Like the MP executor: every rank runs the method; a raising rank becomes a RuntimeError at the caller."""
        self.calls.append(method)
        if self.fail:
            raise RuntimeError("Worker failed with error 'x', please check the stack trace above for the root cause")
        out: list[None] = []
        for rank in self.ranks:
            fn = getattr(rank, method)
            out.append(fn())
        return out


@dataclass
class Request:
    kind: str
    steps: list[dict[str, int]] = field(default_factory=list)   # per engine step: request id -> scheduled tokens


class ScriptedQueue:
    """Each item arrives `delay` seconds after the previous get began; records every call for comparison."""

    def __init__(self, items: list[tuple[float, Request]]) -> None:
        self.items = list(items)
        self.calls: list[tuple[str, object]] = []
        self.clock = 0.0

    def empty(self) -> bool:
        return not self.items or self.items[0][0] > 0

    def get(self, block: bool = True, timeout: float | None = None) -> tuple[str, Request]:
        self.calls.append(('get', (block, timeout)))
        if not self.items:
            if block and timeout is None:
                raise Exhausted
            raise queue.Empty
        delay, request = self.items[0]
        if timeout is not None and delay > timeout:
            self.items[0] = (delay - timeout, request)
            self.clock += timeout
            raise queue.Empty
        if not block and delay > 0:
            raise queue.Empty
        self.items.pop(0)
        self.clock += delay
        return request.kind, request

    def get_nowait(self) -> tuple[str, Request]:
        return self.get(block=False)


class Engine:
    """The attributes _process_input_queue / _process_engine_step touch; the step runs every rank's prefill count."""

    def __init__(self, fns: dict[str, object], script: list[tuple[float, Request]], executor: Executor) -> None:
        self._fns = fns
        self.input_queue = ScriptedQueue(script)
        self.aborts_queue: queue.Queue[list[str]] = queue.Queue()
        self.process_input_queue_block = True
        self.model_executor = executor
        self.output_queue: queue.Queue[object] = queue.Queue()
        self.scheduler = types.SimpleNamespace(has_requests=lambda: False)
        self.pending: list[dict[str, int]] = []
        self.handled: list[str] = []
        self.steps = 0

    def __getattr__(self, name: str) -> Callable[..., object]:
        fn = self.__dict__['_fns'].get(name)
        if not callable(fn):
            raise AttributeError(name)
        return types.MethodType(fn, self)

    def has_work(self) -> bool:
        return bool(self.pending)

    def is_running(self) -> bool:
        return True

    def _notify_idle_state_callbacks(self) -> None:
        pass

    def _handle_client_request(self, kind: str, request: Request) -> None:
        self.handled.append(kind)
        self.pending.extend(request.steps)

    def step_fn(self) -> tuple[dict[int, object], bool]:
        scheduled = self.pending.pop(0)
        for rank in self.model_executor.ranks:
            rank.count(scheduled)
        self.steps += 1
        return {}, True

    def post_step(self, model_executed: bool) -> None:
        pass

    def run(self) -> None:
        """run_busy_loop without shutdown handling: until the script is exhausted."""
        try:
            while True:
                self._process_input_queue()
                if self.has_work():
                    self._process_engine_step()
        except Exhausted:
            pass


def prefill(tokens: int, chunk: int = 4096, rid: str = 'r') -> list[dict[str, int]]:
    """A B1 prompt in production chunks, then 20 decode rounds of 6 verify rows (k = 5)."""
    steps = [{rid: min(chunk, tokens - a)} for a in range(0, tokens, chunk)]
    return steps + [{rid: 6}] * 20


def setup(env: dict[str, str], script: list[tuple[float, Request]], *, overlay: bool = True,
          states: tuple[TorchCounters, TorchCounters] | None = None) -> tuple[Engine, Executor]:
    path = (b.OVERLAY if overlay else b.IMAGE_SRC) / b.CORE
    ranks = [Rank(env, states[0] if states else None), Rank(env, states[1] if states else None)]
    executor = Executor(ranks)
    return Engine(engine_functions(path, env, overlay), script, executor), executor


ON = {'VLLM_DSV41_RELEASE_TOKENS': str(THRESHOLD)}


def test_idle_after_a_131072_token_prompt_releases_once_on_every_rank_after_the_grace() -> None:
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD))), (30, Request('ADD', prefill(512)))])
    engine.run()
    assert executor.calls == ['dsv41_release_if_due'] * 2      # one idle collective_rpc per idle period after work
    for rank in executor.ranks:
        assert rank.torch.empty_calls == 1, 'only the first idle period had >= 131,072 prefill tokens'
        assert rank.torch.sync_calls == 1
        released = [m for k, m in rank.log.records if k == 'info']
        assert len(released) == 1 and released[0].startswith('dsv41 patch 0085b: idle, after 131072 prefill tokens released 4096 MiB')
    grace = [c for c in engine.input_queue.calls if c == ('get', (True, 5.0))]
    assert len(grace) == 2, engine.input_queue.calls


def test_below_the_threshold_nothing_is_released_but_counts_carry_over() -> None:
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD - 1))), (30, Request('ADD', prefill(33)))])
    engine.run()
    assert executor.calls == ['dsv41_release_if_due'] * 2
    first = [r.torch.empty_calls for r in executor.ranks]
    assert first == [1, 1], 'the 33-token prompt brought the count to the threshold; the second idle period releases'


def test_decode_rows_and_short_prompts_are_not_counted() -> None:
    rank = Rank(ON)
    for scheduled in ({'a': 6, 'b': 6, 'c': 5, 'd': 5}, {'a': 32}, {'a': 0}):
        rank.count(scheduled)
    assert getattr(rank, '_dsv41_prefill_since_release', 0) == 0
    rank.count({'a': 33, 'b': 6})
    rank.count({'x': 4096, 'y': 1024, 'z': 6})
    assert getattr(rank, '_dsv41_prefill_since_release') == 33 + 4096 + 1024


def test_a_request_inside_the_grace_period_cancels_the_release() -> None:
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD))), (2.5, Request('ADD', prefill(64))),
                                  (60, Request('UTILITY'))])
    engine.run()
    assert engine.handled == ['ADD', 'ADD', 'UTILITY']
    # first idle: the request at 2.5 s arrived inside the 5 s grace -> no rpc; second idle (60 s gap): one rpc, one release
    assert executor.calls == ['dsv41_release_if_due']
    assert [r.torch.empty_calls for r in executor.ranks] == [1, 1]


def test_one_empty_cache_per_idle_period_even_with_requests_that_bring_no_work() -> None:
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD))), (10, Request('UTILITY')),
                                  (10, Request('UTILITY')), (10, Request('ABORT'))])
    engine.run()
    assert executor.calls == ['dsv41_release_if_due'], 'the step count resets at the release; idle requests do not re-arm it'
    assert [r.torch.empty_calls for r in executor.ranks] == [1, 1]
    # after the release the loop blocks exactly as the image does
    after = engine.input_queue.calls[engine.input_queue.calls.index(('get', (True, 5.0))) + 1:]
    assert after and all(c == ('get', (True, None)) for c in after), after


def test_a_second_release_needs_new_prefill_on_the_rank() -> None:
    rank = Rank(ON)
    rank.count({'a': THRESHOLD})
    rank.dsv41_release_if_due()
    rank.dsv41_release_if_due()
    rank.count({'a': 6})
    rank.dsv41_release_if_due()
    assert rank.torch.empty_calls == 1


@pytest.mark.parametrize('failure', ['empty', 'sync'])
def test_the_rank_method_never_raises_and_resets_its_count(failure: str) -> None:
    state = TorchCounters(fail_empty=failure == 'empty', fail_sync=failure == 'sync')
    rank = Rank(ON, state)
    rank.count({'a': THRESHOLD})
    rank.dsv41_release_if_due()                          # returns normally: no exception reaches the rpc
    warnings = [m for k, m in rank.log.records if k == 'warning']
    assert len(warnings) == 1 and warnings[0].startswith('dsv41 patch 0085b: release failed: RuntimeError(')
    rank.dsv41_release_if_due()
    assert len([m for k, m in rank.log.records if k == 'warning']) == 1, 'no retry storm: the count was reset first'


def test_idle_rpc_with_a_failing_rank_still_returns_on_every_rank() -> None:
    states = (TorchCounters(fail_empty=True), TorchCounters())
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD)))], states=states)
    engine.run()
    assert executor.calls == ['dsv41_release_if_due']
    assert [s.empty_calls for s in states] == [1, 1]


def test_an_executor_error_is_not_swallowed_by_the_engine() -> None:
    engine, executor = setup(ON, [(0, Request('ADD', prefill(THRESHOLD)))])
    executor.fail = True
    with pytest.raises(RuntimeError, match='Worker failed'):
        engine.run()


def test_grace_period_switch() -> None:
    engine, _ = setup({**ON, 'VLLM_DSV41_RELEASE_GRACE_S': '2'}, [(0, Request('ADD', prefill(THRESHOLD))),
                                                                  (3, Request('ADD', prefill(64)))])
    engine.run()
    assert ('get', (True, 2.0)) in engine.input_queue.calls


def test_file_switch_overrides_live_and_bad_files_never_raise(tmp_path: Path) -> None:
    path = tmp_path / 'dsv41-release-tokens'
    env = {'VLLM_DSV41_RELEASE_FILE': str(path)}
    rank = Rank(env)
    rank.count({'a': THRESHOLD})
    rank.dsv41_release_if_due()                          # missing file -> env threshold 0 -> off
    path.write_text('garbage\n')
    rank.dsv41_release_if_due()                          # invalid -> env threshold 0 -> off
    path.write_text('0\n')
    rank.dsv41_release_if_due()
    assert rank.torch.empty_calls == 0 and not [r for r in rank.log.records if r[0] == 'warning']
    path.write_text(f'{THRESHOLD}\n')
    rank.dsv41_release_if_due()
    assert rank.torch.empty_calls == 1
    path.write_text(f'{THRESHOLD * 2}\n')
    rank.count({'a': THRESHOLD})
    rank.dsv41_release_if_due()
    assert rank.torch.empty_calls == 1, 'a raised threshold applies at the next idle check'


def test_file_mode_engine_side_is_on_even_when_the_file_says_zero(tmp_path: Path) -> None:
    """With only the file switch set the engine core always idles through the grace and calls the rpc; the ranks
    decide. This is the mounted-and-switched-but-off state of a file-toggled window (MEASURE-0085B.md)."""
    path = tmp_path / 'f'
    path.write_text('0\n')
    engine, executor = setup({'VLLM_DSV41_RELEASE_FILE': str(path)}, [(0, Request('ADD', prefill(THRESHOLD)))])
    engine.run()
    assert executor.calls == ['dsv41_release_if_due']
    assert [r.torch.empty_calls for r in executor.ranks] == [0, 0]


@pytest.mark.parametrize('env', [{}, {'VLLM_DSV41_RELEASE_TOKENS': '0'}, {'VLLM_DSV41_RELEASE_TOKENS': 'x'},
                                 {'VLLM_DSV41_RELEASE_TOKENS': ''}, {'VLLM_DSV41_RELEASE_FILE': ''}])
def test_switch_off_is_the_image_call_for_call(env: dict[str, str]) -> None:
    script = [(0, Request('ADD', prefill(THRESHOLD))), (2.5, Request('UTILITY')), (30, Request('ADD', prefill(40))),
              (100, Request('ABORT'))]
    off, off_exec = setup(env, script)
    image, image_exec = setup(env, script, overlay=False)
    off.run()
    image.run()
    assert off.input_queue.calls == image.input_queue.calls
    assert off.handled == image.handled and off.steps == image.steps
    assert off_exec.calls == image_exec.calls == []
    for rank in off_exec.ranks:
        assert getattr(rank, '_dsv41_prefill_since_release', None) is None, 'switch off: no count kept'
        assert rank.torch.empty_calls == 0
    scope = engine_functions(b.OVERLAY / b.CORE, env, overlay=True)
    assert scope['_DSV41_RELEASE_IDLE'] is False
    worker = worker_functions(env, TorchCounters(), Logger())
    assert worker['_DSV41_RELEASE_COUNT'] is False
