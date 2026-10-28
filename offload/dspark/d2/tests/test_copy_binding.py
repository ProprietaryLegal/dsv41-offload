"""Copy receipt verification against the reviewed fake-host transfer, CPU only."""
import json

import pytest

from d2.copy_binding import verify_copy, digest
from offload.dspark.d2.tests.test_decode_copy import fixture_tree, perform, no_network


def copied(tmp_path, monkeypatch):
    no_network.__wrapped__(monkeypatch)
    fixture_tree(tmp_path)
    perform(tmp_path, top20=str(tmp_path / 'state'))
    return tmp_path / 'new'


def test_copy_receipt_binds_all_files_accepts_additive_fields(tmp_path, monkeypatch):
    root = copied(tmp_path, monkeypatch)
    receipt = root / 'COPY_RECEIPT.json'
    raw = json.loads(receipt.read_text())
    raw['future_coordinator_field'] = {'extra': True}
    receipt.write_text(json.dumps(raw))
    assert verify_copy(root) == digest(receipt)


@pytest.mark.parametrize('bad', ['missing', 'moved', 'hash', 'extra', 'duplicate', 'traversal', 'inventory'])
def test_bad_copy_refused(tmp_path, monkeypatch, bad):
    root = copied(tmp_path, monkeypatch)
    receipt = root / 'COPY_RECEIPT.json'
    raw = json.loads(receipt.read_text())
    path = root / 'capture/s0/taps.bin'
    if bad == 'missing':
        receipt.unlink()
    elif bad == 'moved':
        path.rename(path.with_name('moved.bin'))
    elif bad == 'hash':
        path.write_bytes(b'y' * path.stat().st_size)
    elif bad == 'extra':
        (root / 'generator/extra').touch()
    else:
        if bad == 'duplicate':
            raw['files'].append(raw['files'][0])
        elif bad == 'traversal':
            raw['files'][0]['relative'] = '../escape'
        else:
            raw['inventories']['generator']['state.json']['sha256'] = '0' * 64
        receipt.write_text(json.dumps(raw))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_copy(root)
