from pathlib import Path

import pytest

from offload.dspark.d2.train.full_admission import verify_pilot_source


def original(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    import subprocess
    from offload.dspark.d2.train.tests.snapshot_git import git
    root = Path(__file__).resolve().parents[5]
    tmp_path.mkdir(parents=True, exist_ok=True)
    # The real staged snapshot is a whole-commit archive: its repo-root modules are bound too (S41 FIX 1).
    archive = git(root, 'archive', 'd1dd41a', 'offload/dspark', 'bench.py', 'functest.py', 'quality.py', 'report_tables.py', 'verify_hashes.py').stdout
    subprocess.run(['tar', '-x', '-C', str(tmp_path)], input=archive, check=True)
    from offload.dspark.d2.train.full_admission import PILOT_SOURCES
    return tmp_path, dict(PILOT_SOURCES)


def test_original_source_admitted_and_edits_refused(tmp_path: Path) -> None:
    root, mapping = original(tmp_path)
    verify_pilot_source(root, mapping)
    p = root/'offload/dspark/d2/train/loss.py'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='immutable pilot source changed'):
        verify_pilot_source(root, mapping)


def test_original_addition_and_unknown_lineage_refused(tmp_path: Path) -> None:
    root, mapping = original(tmp_path)
    (root/'offload/dspark/d2/train/unreviewed.py').write_text('')
    with pytest.raises(ValueError, match='file set changed'):
        verify_pilot_source(root, mapping)
    mapping['d2/train/loop.py'] = '0'*64
    with pytest.raises(ValueError, match='lineage'):
        verify_pilot_source(root, mapping)


@pytest.mark.parametrize('verdict,synthetic', [('PASS', True), ('FAIL', False), ('INCONCLUSIVE', False)])
def test_original_verifier_still_refuses_unqualified_receipts(tmp_path: Path, verdict: str, synthetic: bool) -> None:
    import json
    from offload.dspark.d2.train.full_admission import require_full_pilot
    root, mapping = original(tmp_path/'old-source')
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'schema': 'dsv41.d2.pilot-receipt.v1', 'passed': verdict == 'PASS',
        'synthetic': synthetic, 'verdict': verdict, 'plan': {'source_sha256': mapping}}))
    with pytest.raises(ValueError, match='PASSING real pilot receipt'):
        require_full_pilot(receipt, 'base', tmp_path/'manifest', tmp_path/'parity', tmp_path/'g1',
                           128, 8192, 32, root)


def test_real_whole_commit_snapshot_admitted(tmp_path: Path) -> None:
    """S41 FIX 1: pilot_stage stages `git archive d1dd41a` (whole commit); that exact tree must pass the source checks."""
    import subprocess
    from offload.dspark.d2.train.full_admission import PILOT_SOURCES
    from offload.dspark.d2.train.tests.snapshot_git import git
    archive = git(Path(__file__).resolve().parents[5], 'archive', 'd1dd41a').stdout
    subprocess.run(['tar', '-x', '-C', str(tmp_path)], input=archive, check=True)
    verify_pilot_source(tmp_path, dict(PILOT_SOURCES))


def test_unpinned_or_changed_root_module_refused(tmp_path: Path) -> None:
    root, mapping = original(tmp_path)
    (root/'quality.py').write_text((root/'quality.py').read_text()+'#\n')
    with pytest.raises(ValueError, match='package initializer changed'):
        verify_pilot_source(root, mapping)
    root2, _ = original(tmp_path/'second')
    (root2/'extra_root.py').write_text('')
    with pytest.raises(ValueError, match='unbound pilot import source'):
        verify_pilot_source(root2, mapping)


@pytest.mark.parametrize('module', ['offload/dspark/d2/train/fast_policy', 'offload/dspark/d2/g1', 'report_tables'])
def test_package_directory_shadowing_a_pinned_module_refused(tmp_path: Path, module: str) -> None:
    root, mapping = original(tmp_path)
    (root/module).mkdir()
    (root/module/'__init__.py').write_text('')
    with pytest.raises(ValueError, match='shadow'):
        verify_pilot_source(root, mapping)


@pytest.mark.parametrize('extra', ['unknown', 'changed'])
def test_full_tree_extras_must_be_exactly_the_pinned_diag_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 extra: str) -> None:
    """S41 FIX 2: the merged tree may add only pilot_diag.py, byte-pinned."""
    import offload.dspark.d2.train.pilot as pilot
    from offload.dspark.d2.train import full_admission as A
    root, mapping = original(tmp_path)
    verify_pilot_source(root, mapping)
    current = pilot.sources()
    assert set(current) - set(mapping) == set(A.FULL_EXTRAS)
    bad = dict(current)
    if extra == 'unknown':
        bad['d2/train/unreviewed.py'] = '0'*64
    else:
        bad['d2/train/pilot_diag.py'] = '0'*64
    monkeypatch.setattr(pilot, 'sources', lambda: bad)
    with pytest.raises(ValueError, match='file set differs'):
        verify_pilot_source(root, mapping)


def test_planted_offload_package_at_the_dspark_root_refused(tmp_path: Path) -> None:
    """S41 round 2 observation: offload/dspark/offload/__init__.py would take over the `offload` package in the child."""
    root, mapping = original(tmp_path)
    (root/'offload/dspark/offload').mkdir()
    (root/'offload/dspark/offload/__init__.py').write_text('')
    with pytest.raises(ValueError, match='shadow'):
        verify_pilot_source(root, mapping)
