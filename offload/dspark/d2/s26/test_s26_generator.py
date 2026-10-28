"""S26 identity invariants, re-pinned to reviewed option-ii v2; no production state I/O."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from d2 import generate as G

REPO = Path(__file__).resolve().parents[4]
GIT = os.environ.get('S26_GIT_ROOT', str(Path(__file__).resolve().parents[4]))  # archived merge trees
HIST = '/opt/dsv41/data/d2/gen'


def reviewed(revision='f13a825'):
    """Read the historical request-body baseline or pinned option-ii generator from git."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    src = subprocess.run(['git', 'show', revision + ':offload/dspark/d2/generate.py'], cwd=GIT,
                         env=env, capture_output=True, check=True).stdout
    spec = importlib.util.spec_from_loader('d2.s26_generate_' + revision, loader=None)
    assert spec
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__['__package__'] = 'd2'
    sys.modules[spec.name] = mod
    exec(compile(src, 'generate_' + revision, 'exec'), mod.__dict__)
    return mod


def test_only_request_identity_and_revision_change() -> None:
    old = reviewed()
    for name in ('MAX_STREAMS', 'AUX_PRIORITY', 'DEFAULT_STREAMS', 'ORDER', 'SCHEMA'):
        assert getattr(old, name) == getattr(G, name)
    assert (old.GENERATOR_REVISION, G.GENERATOR_REVISION) == ('d2-priority20-v1', 'd2-decode-capture-option-ii-v2')
    for prompt in ([5] * 3000, [5] * 50):
        a = old.Stream(old.Server('http://127.0.0.1:9', 'm'), 'p0007', 2, prompt, 4096)
        b = G.Stream(G.Server('http://127.0.0.1:9', 'm'), 'p0007', 2, prompt, 4096)
        assert {k: v for k, v in a.body.items() if k != 'request_id'} == \
            {k: v for k, v in b.body.items() if k != 'request_id'}
        assert a.request_id == 'd2g.p0007.2' and b.request_id == f'd2c.p0007.2.{max(0, len(prompt) - 128)}.{len(prompt)}'
    # Option-ii deliberately added top20 state/checkpoint/error handling since
    # f13a825. Pin those methods to c1eaaed rather than claiming they never changed.
    # Keep the historical request-body/identity invariants above, and all current
    # driver methods (including Stream.__init__) against this reviewed revision.
    pinned = reviewed('c1eaaed')
    import inspect
    for cls in ('Driver', 'HoldPoller', 'Stream', 'Server'):
        for name, fn in inspect.getmembers(getattr(G, cls), inspect.isfunction):
            assert fn.__code__.co_code == getattr(getattr(pinned, cls), name).__code__.co_code, (cls, name)


@pytest.mark.parametrize('alias', ['link', 'dot', 'slash'])
def test_historical_gen_refused_through_aliases(tmp_path: Path, alias: str, monkeypatch) -> None:
    historical = tmp_path / 'historical'
    historical.mkdir()
    marker = historical / 'marker'
    marker.write_text('synthetic historical state')
    state = {'link': str(tmp_path / 'g'), 'dot': str(historical) + '/.',
             'slash': str(historical) + '/'}[alias]
    if alias == 'link':
        os.symlink(historical, state)
    realpath = os.path.realpath
    # Resolve real scratch aliases, then model the production canonical name.
    # No existence checks, directory listings or reads touch the production tree.
    monkeypatch.setattr(G.os.path, 'realpath', lambda p, **kw:
                        HIST if realpath(p, **kw) == str(historical) else realpath(p, **kw))
    with pytest.raises(SystemExit, match='NEW second-pass'):
        G.cmd_run(SimpleNamespace(state=state))
    assert list(historical.iterdir()) == [marker]
    assert marker.read_text() == 'synthetic historical state'
