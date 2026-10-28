"""Production K1 composition, including both DROP_MTP_BUFFER None paths."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.launch.windows import pp4_prepare as p

SOURCE = Path('/opt/dsv41/data/src-image')


def test_exact_k1_composition_and_removal_of_only_new_additions(tmp_path: Path) -> None:
    out = tmp_path/'prepared'
    receipt = p.prepare(SOURCE, out)
    original = p.K1_RUNNER.read_text()
    actual = (out/pp3.RUNNER).read_text()
    assert receipt == json.loads((p.ROOT/'offload/dspark/probabilistic/pp4/verifier-k1-receipt.json').read_text())
    assert receipt['production_k1_sha256'] == p.K1_SHA256
    assert hashlib.sha256(original.encode()).hexdigest() == p.K1_SHA256
    # Removing exactly PP4's import/decorators and PP3's added mask/ABI
    # reconstructs the entire reviewed K1 source, including its memory probe.
    actual = actual.removeprefix('from vllm.v1.worker.gpu import dsv41_pp4_round_trace as _pp4_round\n')
    actual = actual.replace('    @_pp4_round.execution\n', '').replace('    @_pp4_round.sampling\n', '')
    assert actual == pp3.runner(original)
    prefix = 'REMOTE_DRAFT_PROBS_ABI = ' + repr(pp3.ABI) + '\n'
    delta = pp3.runner(original).removeprefix(prefix)
    start = delta.index('        # PP3 failed/missing-q rounds are target-only.')
    end = delta.index('        sampler_output: SamplerOutput | None\n', start)
    assert delta[:start]+delta[end:] == original
    assert (out/pp3.UTILS).read_text() == pp3.utilities((SOURCE/pp3.UTILS).read_text())


@pytest.mark.parametrize('site', [0, 1])
@pytest.mark.parametrize('value', [None, 'mtp'])
def test_actual_none_return_branches_preserve_hidden_states(tmp_path: Path, site: int, value: str | None) -> None:
    out = tmp_path/'prepared'
    p.prepare(SOURCE, out)
    source = (out/pp3.RUNNER).read_text()
    blocks = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
              and n.test.func.id == 'hasattr' and len(n.test.args) > 1
              and isinstance(n.test.args[1], ast.Constant) and n.test.args[1].value == 'get_mtp_target_hidden_states']
    assert len(blocks) == 2
    fn = ast.parse('def probe(self, hidden_states, draft_hidden_states):\n    spec_hidden_states = hidden_states\n    return spec_hidden_states\n')
    definition = cast(ast.FunctionDef, fn.body[0])
    definition.body.insert(1, copy.deepcopy(blocks[site])); ast.fix_missing_locations(fn)
    ns: dict[str, object] = {}
    exec(compile(fn, 'actual-K1-branch', 'exec'), ns)  # noqa: S102 -- execute only hash-pinned K1 branch
    class Hidden:
        shape = (1, 1)
        def size(self, dim: int) -> int: return 1
        def __getitem__(self, index: object) -> str: return 'mtp-selected'
    hidden = Hidden()
    model = SimpleNamespace(model=SimpleNamespace(get_mtp_target_hidden_states=lambda: None if value is None else hidden))
    result = cast(Callable[[object, object, object], object], ns['probe'])(model, hidden, hidden)
    assert result is hidden if value is None else result == 'mtp-selected'


def test_k1_hash_drift_refuses_before_any_output(tmp_path: Path) -> None:
    changed = tmp_path/'changed.py'; changed.write_bytes(p.K1_RUNNER.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='pinned source mismatch'):
        p.prepare(SOURCE, tmp_path/'output', changed)
    assert not (tmp_path/'output').exists()


def test_off_failure_mask_does_not_change_original_invalid_drafts() -> None:
    original = p.K1_RUNNER.read_text()
    generated = pp3.runner(original)
    start = generated.index('        pp3_valid = ')
    end = generated.index('        sampler_output: SamplerOutput | None', start)
    block = generated[start:end]
    for sentinel in (None, object()):
        ns = {'self': SimpleNamespace(speculator=SimpleNamespace()), 'invalid_drafts': sentinel}
        exec(compile(__import__('textwrap').dedent(block), 'actual-OFF-mask', 'exec'), ns)  # noqa: S102 -- pinned mask branch
        assert ns['invalid_drafts'] is sentinel
