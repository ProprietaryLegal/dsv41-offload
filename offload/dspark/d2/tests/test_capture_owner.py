"""Sequence files take the run dir's owner at creation, so the host user can read a ledger before any flush."""
from __future__ import annotations

import os
from pathlib import Path

from capture.decode_capture import SequenceFiles


def test_new_sequence_files_take_the_run_dir_owner(tmp_path: Path) -> None:
    run = tmp_path / 'run'
    run.mkdir(mode=0o700)
    files = SequenceFiles(run, 'smoke0', 0)
    owner = run.stat()
    for p in (run / 'smoke0', run / 'smoke0' / 'CAPTURE.json', run / 'smoke0' / 'taps.bin',
              run / 'smoke0' / 'ids.bin', run / 'smoke0' / 'ledger.jsonl'):
        st = p.stat()
        assert (st.st_uid, st.st_gid) == (owner.st_uid, owner.st_gid)
    assert os.access(run / 'smoke0' / 'ledger.jsonl', os.R_OK)
    del files
