"""CPU checks of the PP4 #61161 verifier-RNG gate receipt (the real tests run under pp4_rng_gate --execute)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_rng_gate as g
from offload.prefill_maxlen.evidence import Json

HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip()
needs_image = pytest.mark.skipif(not (g.IMAGE_SRC / pp3.UTILS).is_file(), reason='image verifier source absent')


def junit(path: Path, counts: dict[str, int], *, fail: str = '') -> None:
    cases = ''.join(f'<testcase classname="offload.audit.verifier_rng.{module}" name="t{i}">'
                    + ('<failure/>' if module == fail and i == 0 else '') + '</testcase>'
                    for module, n in counts.items() for i in range(n))
    path.write_text(f'<testsuites><testsuite>{cases}</testsuite></testsuites>')


def fake_runner(counts: dict[str, int], rc: int = 0, fail: str = '') -> object:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[1:4] == ['-m', 'pytest', '-q'] and list(g.TESTS) == argv[-2:]
        env = cast(dict[str, str], kwargs['env'])
        assert env['TRITON_INTERPRET'] == '1' and env['CUDA_VISIBLE_DEVICES'] == ''
        junit(Path(next(a for a in argv if a.startswith('--junitxml=')).split('=', 1)[1]), counts, fail=fail)
        return subprocess.CompletedProcess(argv, rc, '', '')
    return run


@needs_image
def test_static_checks_on_the_generated_pp4_kernel() -> None:
    result = g.static_checks(g.pp4_kernel())
    assert result == {'acceptance_draws': 1, 'block_refusal': True, 'step_keyed': False, 'pass': True}
    broken = g.pp4_kernel().replace(g.BLOCK_REFUSAL, 'if False:')
    assert g.static_checks(broken)['pass'] is False


@pytest.mark.parametrize('spec,ok', [('{"method":"dspark","num_speculative_tokens":5}', True),
    ('{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}', True),
    ('{"method":"dspark","num_speculative_tokens":5,"rejection_sample_method":"block"}', False),
    ('{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}', False),
    ('{"method":"mtp","num_speculative_tokens":5}', False)])
def test_standard_spec(spec: str, ok: bool) -> None:
    if ok:
        g.standard_spec(spec)
    else:
        with pytest.raises(ValueError):
            g.standard_spec(spec)


@needs_image
@pytest.mark.parametrize('change', ['', 'commit', 'kernel', 'count', 'source', 'status'])
def test_receipt_binds_commit_tests_sources_and_admitted_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                change: str) -> None:
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kw: HEAD if argv[1] == 'rev-parse' else '')
    out = tmp_path / 'gate.json'
    receipt = g.run(out, tmp_path / 'base', runner=fake_runner(dict(g.EXPECTED)))  # type: ignore[arg-type]
    assert receipt['status'] == 'PASS' and receipt['commit'] == HEAD
    admission: dict[str, Json] = {'verifier_sha256': {pp3.UTILS: receipt['pp4_kernel_sha256']}}
    data = json.loads(out.read_text())
    head = HEAD
    if change == 'commit':
        head = 'f' * 40
    elif change == 'kernel':
        admission = {'verifier_sha256': {pp3.UTILS: 'e' * 64}}
    elif change == 'count':
        data['tests']['test_pp4_sampled_rng']['passed'] = 3
    elif change == 'source':
        data['source_sha256'][g.TESTS[0]] = '0' * 64
    elif change == 'status':
        data['status'] = 'FAIL'
    out.write_text(json.dumps(data))
    if change:
        with pytest.raises(ValueError):
            g.validate(out, head, admission)
    else:
        assert g.validate(out, head, admission)['status'] == 'PASS'


@needs_image
@pytest.mark.parametrize('counts,rc,fail', [({'test_verifier_rng': 6, 'test_pp4_sampled_rng': 3}, 0, ''),
                                            (dict(g.EXPECTED), 1, 'test_pp4_sampled_rng')])
def test_failed_or_short_run_is_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counts: dict[str, int],
                                     rc: int, fail: str) -> None:
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kw: HEAD if argv[1] == 'rev-parse' else '')
    receipt = g.run(tmp_path / 'gate.json', tmp_path / 'base', runner=fake_runner(counts, rc, fail))  # type: ignore[arg-type]
    assert receipt['status'] == 'FAIL'
    with pytest.raises(ValueError, match='did not PASS'):
        g.validate(tmp_path / 'gate.json', HEAD, {'verifier_sha256': {pp3.UTILS: receipt['pp4_kernel_sha256']}})


def test_dirty_tree_or_existing_receipt_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'gate.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh'):
        g.run(tmp_path / 'gate.json', tmp_path)
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kw: HEAD if argv[1] == 'rev-parse' else ' M x.py')
    with pytest.raises(ValueError, match='dirty'):
        g.run(tmp_path / 'new.json', tmp_path)
