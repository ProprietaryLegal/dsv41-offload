"""S36 probes of merge b6f79e9 (9ff3a51): the pilot root is only decode-d2pass2-r1-c2.

Static probes on the merged source, plus read-only checks of the x86host receipt copies in S36-inputs (skipped if absent).
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import hashlib
import re
import subprocess

import pytest

from offload.dspark.d2.train import pilot_stage as S
from offload.dspark.d2.train.tests.test_pilot_stage import settings

REPO = Path(__file__).resolve().parents[4]
D2 = REPO/'offload/dspark/d2'
C2 = '/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2'
OLD = re.compile(r'/opt/dsv41/x86host-data/d2/decode-d2pass2-r1(?!-c2)')
INPUTS = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/S36-inputs')
needs_real = pytest.mark.skipif(not (INPUTS/'ASSEMBLED.json').exists(), reason='x86host receipt copies absent')


def test_no_code_or_runbook_outside_tests_names_the_first_copy_assembly_root() -> None:
    hits = []
    for path in D2.rglob('*'):
        if path.suffix in ('.py', '.sh', '.md', '.json') and 'tests' not in path.parts and 's36' not in path.parts:
            for n, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                if OLD.search(line):
                    hits.append(f'{path.relative_to(REPO)}:{n}')
    # Only the historical copy runbook (decode_copy destination for the FIRST attempt) still names it.
    assert hits == ['offload/dspark/d2/DECODE-CLOSE.md:92']


def test_execute_admits_only_c2_and_every_rendered_path_derives_from_it(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    assert str(S.ASSEMBLY) == C2
    monkeypatch.setenv('D2_COORDINATOR', S.COORDINATOR)
    monkeypatch.setattr(S.socket, 'gethostname', lambda: 'x86host')
    for bad in ('/opt/dsv41/x86host-data/d2/decode-d2pass2-r1', '/opt/dsv41/x86host-data/d2/decode-d2pass2-r1/',
                C2 + '/../decode-d2pass2-r1', '/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2x'):
        with pytest.raises(ValueError, match='decode-d2pass2-r1-c2 assembly path'):
            S.execute(replace(settings(tmp_path), assembly=Path(bad)), 'stage')
    s = replace(settings(tmp_path), assembly=S.ASSEMBLY)
    assert s.manifest == Path(C2 + '/shards/MANIFEST.json')
    rendered = json.dumps(S.commands(s))
    assert not OLD.search(rendered) and rendered.count(C2 + '/shards/MANIFEST.json') >= 5


def test_cli_default_assembly_is_c2() -> None:
    source = (D2/'train/pilot_stage.py').read_text()
    assert "ap.add_argument('--assembly', type=Path, default=ASSEMBLY)" in source


@needs_real
def test_real_receipts_bind_only_c2() -> None:
    for name in ('ASSEMBLED.json', 'decode-plan/DECODE.json', 'shards/MANIFEST.json', 'dev.json',
                 'heldout_legal.json', 'heldout_general.json', 'live-reference.json', 'pilot-plan-header.json'):
        assert not OLD.search((INPUTS/name).read_text()), name
    assembled = json.loads((INPUTS/'ASSEMBLED.json').read_text())
    assert assembled['relocation']['root'] == C2 and assembled['g1a_pass'] is True and assembled['synthetic'] is False
    receipt = json.loads((INPUTS/'COPY_RECEIPT.json').read_text())
    assert {Path(f['file']).parts[5] for f in receipt['files']} == {'decode-d2pass2-r1-c2'} and len(receipt['files']) == 662
    # The only bare 'decode-d2pass2-r1' strings in the copy receipt are the sparkhead capture source directory.
    assert receipt['source'] == '/home/rank0user/dsv41-offload-run/d2-decode-d2pass2-r1'
    plan = json.loads((INPUTS/'pilot-plan-header.json').read_text())
    assert plan['manifest_sha256'] == assembled['outputs']['shards/MANIFEST.json']
    for module in ('g1.py', 'live_reference.py', 'train/pilot_stage.py', 'train/pilot.py', 'train/pilot_eval.py',
                   'train/pilot_verdict.py', 'train/pilot_sample.py'):
        blob = subprocess.run(['git', '-C', str(REPO), 'show', 'b6f79e9:offload/dspark/d2/' + module],
                              capture_output=True, check=True).stdout
        assert plan['source_sha256']['d2/' + module] == hashlib.sha256(blob).hexdigest(), module
