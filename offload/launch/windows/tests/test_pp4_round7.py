"""S64 cross-admission transplant counterexample; CPU fixtures only."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_admission as a
from offload.launch.windows import pp4_budget as b
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.tests.test_pp4_admission import Fake, run
from offload.launch.windows.tests.test_pp4_window import WORKER
from offload.prefill_maxlen.evidence import Json, record


class OwnRun(Fake):
    def __init__(self, root: Path, tag: str, seconds: float) -> None:
        super().__init__(root)
        self.tag, self.seconds = tag, seconds

    def qualify_pair(self) -> dict[str, Json]:
        physical = super().qualify_pair()
        physical.update(run=self.tag, seconds=self.seconds)
        a.write(self.out/'physical-pair-qualification.json', physical)
        return physical


def transplant(root: Path) -> tuple[a.Session, dict[str, Json], Path]:
    old, new = root/'old', root/'new'
    old.mkdir(); new.mkdir()
    old_session, _ = run(OwnRun(old, 'PP4-admission-OLD', 120.))
    session, result = run(OwnRun(new, 'PP4-admission-NEW', 700.))
    assert old_session.data['verdict'] == session.data['verdict'] == 'QUALIFIED'
    receipt = new/'admission-receipt.json'
    with pytest.raises(ValueError, match='86.55min exceeds85min'):
        b.admitted(receipt, WORKER.commit)
    primary = new/'physical-pair-qualification.json'
    original = (old/primary.name).read_bytes()
    shutil.copyfile(old/primary.name, primary)
    # Update the same bookkeeping as S64. Preserve the NEW parent identity,
    # semantic gates, commit, restore/watch and the intact OLD primary bytes.
    session.data['physical_pair_evidence'] = copy.deepcopy(old_session.data['physical_pair_evidence'])
    record(session.data['physical_pair_evidence'])['evidence_path'] = str(primary.resolve())
    record(session.data['artifact_sha256'])[primary.name] = pf.digest(primary)
    session.save()
    result['physical_pair'] = ck.read(primary)
    assert primary.read_bytes() == original
    return session, result, receipt


def test_s64_intact_old_pair_cannot_price_new_admission(tmp_path: Path) -> None:
    session, _, receipt = transplant(tmp_path)
    with pytest.raises(ValueError, match='admission run'):
        b.admitted(receipt, WORKER.commit)
    assert session.data['admission_run'] == 'PP4-admission-NEW'


def test_s64_refinish_cannot_qualify_transplanted_pair(tmp_path: Path) -> None:
    session, result, _ = transplant(tmp_path)
    session.finish(result)
    assert session.data['verdict'] == 'NOT QUALIFIED'
    assert session.data['failing_gate'] == 'physical_pair'
    assert 'physical-peer receipt identity' in str(session.data['stop'])


def test_parent_identity_persisted_before_pair_and_seed_cannot_override(tmp_path: Path) -> None:
    fake = OwnRun(tmp_path, 'PP4-admission-NEW', 120.)
    fake.admission['admission_run'] = 'PP4-admission-OLD'
    session = a.Session(fake, lambda row: None)
    assert session.data['admission_run'] == fake.tag
    assert ck.read(tmp_path/'admission-receipt.json')['admission_run'] == fake.tag
    assert fake.events == [] and not (tmp_path/'physical-pair-qualification.json').exists()


@pytest.mark.parametrize('parent', [None, '', ' ', 42, 'PP4-admission-OLD'])
def test_missing_empty_or_foreign_parent_refuses(tmp_path: Path, parent: Json) -> None:
    session, _ = run(OwnRun(tmp_path, 'PP4-admission-NEW', 120.))
    if parent is None: session.data.pop('admission_run', None)
    else: session.data['admission_run'] = parent
    session.save()
    with pytest.raises(ValueError, match='admission run'):
        b.admitted(tmp_path/'admission-receipt.json', WORKER.commit)


@pytest.mark.parametrize('foreign', ['run', 'commit', 'directory', 'symlink'])
def test_same_run_foreign_commit_or_directory_refuses(tmp_path: Path, foreign: str) -> None:
    session, result = run(OwnRun(tmp_path, 'PP4-admission-NEW', 120.))
    primary = tmp_path/'physical-pair-qualification.json'
    if foreign in ('run', 'commit'):
        physical = record(result['physical_pair'])
        physical['source_commit' if foreign == 'commit' else 'run'] = 'f'*40 if foreign == 'commit' else 'PP4-admission-OLD'
        a.write(primary, physical)
    else:
        elsewhere = tmp_path/'other-directory'
        elsewhere.mkdir()
        moved = elsewhere/primary.name
        primary.rename(moved)
        if foreign == 'symlink': primary.symlink_to(moved)
        else: primary = moved
        record(session.data['physical_pair_evidence'])['evidence_path'] = str(primary.absolute())
    record(session.data['physical_pair_evidence'])['sha256'] = pf.digest(primary)
    record(session.data['artifact_sha256'])[primary.name] = pf.digest(primary)
    session.save()
    with pytest.raises(ValueError):
        b.admitted(tmp_path/'admission-receipt.json', WORKER.commit)
