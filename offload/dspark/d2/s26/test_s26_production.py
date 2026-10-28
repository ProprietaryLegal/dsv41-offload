"""S26 production-safety probes of the decode-capture overlay (CPU; fake engine classes, real overlay/remote text).

The overlay's decode section and the remote speculator's appended hook are executed verbatim in a throwaway package,
with fake DSparkSpeculator / BaseSpeculator classes whose propose signatures are checked against the real files.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
GIT = os.environ.get('S26_GIT_ROOT', str(Path(__file__).resolve().parents[4]))  # archived merge trees
OVERLAY = ROOT / 'capture/speculator_replay.py'
CAPTURE = ROOT / 'capture/decode_capture.py'
REMOTE = ROOT / 'spark/remote_dspark/speculator.py'
IMAGE = Path('/opt/dsv41/data/src-image/vllm/v1/worker/gpu/spec_decode')
REPLAY_MARK = b'# --- DSV41-OFFLOAD D2 replay overlay'
DECODE_MARK = '# --- D2 decode capture:'
REMOTE_MARK = '# Capacity mode does not call'
PROPOSE_ARGS = ['self', 'input_batch', 'attn_metadata', 'slot_mappings', 'last_hidden_states', 'aux_hidden_states',
                'num_sampled', 'num_rejected', 'last_sampled', 'next_prefill_tokens', 'temperature', 'seeds',
                'dp_sync', 'dummy_run', 'skip_attn_for_dummy_run', 'mm_inputs', 'is_profile']


def propose_args(path: Path, cls: str) -> list[str]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == 'propose':
                    return [a.arg for a in f.args.args]
    raise AssertionError(f'{cls}.propose not found in {path}')


def test_real_propose_signatures_match_the_wrapper_binding() -> None:
    # The wrapper binds by name: input_batch, aux_hidden_states, num_rejected, dummy_run, is_profile.
    assert propose_args(IMAGE / 'dflash/speculator.py', 'DFlashSpeculator') == PROPOSE_ARGS
    assert propose_args(REMOTE, 'RemoteDSparkCapacitySpeculator') == PROPOSE_ARGS
    assert propose_args(REMOTE, 'RemoteDSparkLocalSpeculator') == PROPOSE_ARGS
    # The image DSpark class inherits propose (so the overlay's wrap of DSparkSpeculator covers the local drafter).
    assert 'def propose' not in (IMAGE / 'dspark/speculator.py').read_text()


def test_overlay_image_prefix_is_byte_identical_to_the_actual_image_file() -> None:
    raw = OVERLAY.read_bytes()
    prefix = raw[:raw.index(REPLAY_MARK)].rstrip(b'\n') + b'\n'
    assert prefix == (IMAGE / 'dspark/speculator.py').read_bytes()


def test_remote_prefix_is_the_deployed_file_and_the_hook_is_appended_only() -> None:
    import subprocess
    base = subprocess.run(['git', 'show', 'f13a825:offload/dspark/spark/remote_dspark/speculator.py'],
                          cwd=GIT, capture_output=True, check=True).stdout
    raw = REMOTE.read_bytes()
    assert raw.startswith(base) and hashlib.sha256(base).hexdigest() == \
        '441be6b7d245114974c5dcb307c2efebd29516b958cd363bc70e24bcd73bdcf3'
    assert REMOTE_MARK.encode() in raw[len(base):]


class Poison:
    def __getattr__(self, name: str):
        raise AssertionError('untagged tensor access: ' + name)

    def __getitem__(self, key):
        raise AssertionError('untagged tensor indexing')


class FakeBase:
    calls: list[str]

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
        return 'drafts'


def build(tmp_path: Path, env: dict[str, str]):
    """Package `s26pkg` = overlay decode section over fake DSparkSpeculator + decode_capture; then the remote hook."""
    pkg = tmp_path / 'pkg' / 's26pkg'
    pkg.mkdir(parents=True)
    (pkg / '__init__.py').write_text('')
    shutil.copy(CAPTURE, pkg / 'decode_capture.py')
    text = OVERLAY.read_text()
    section = text[text.index(DECODE_MARK):]
    (pkg / 'speculator.py').write_text(
        'import os as _d2r_os, logging\nimport torch\n'
        'from d2.s26.test_s26_production import FakeBase\n'
        'logger = logging.getLogger("s26")\n_D2R_DIR = None\n'
        'class DSparkSpeculator(FakeBase):\n    pass\n' + section)
    for k in list(sys.modules):
        if k.startswith('s26pkg'):
            del sys.modules[k]
    sys.path.insert(0, str(tmp_path / 'pkg'))
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        spec_mod = importlib.import_module('s26pkg.speculator')
        sys.modules['vllm.v1.worker.gpu.spec_decode.dspark.speculator'] = spec_mod
        DS = spec_mod.DSparkSpeculator

        class RemoteDSparkCapacitySpeculator(FakeBase):          # BaseSpeculator subclass: own propose
            def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                        num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
                        dp_sync=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                        is_profile=False):
                return 'remote'

        class RemoteDSparkLocalSpeculator(DS):                   # DSparkSpeculator subclass: super().propose
            def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                        num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
                        dp_sync=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                        is_profile=False):
                args = (input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                        num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds)
                return super().propose(*args, dp_sync=dp_sync, dummy_run=dummy_run, mm_inputs=mm_inputs,
                                       is_profile=is_profile)

        ns = {'os': os, 'RemoteDSparkCapacitySpeculator': RemoteDSparkCapacitySpeculator,
              'RemoteDSparkLocalSpeculator': RemoteDSparkLocalSpeculator}
        rtext = REMOTE.read_text()
        exec(compile(rtext[rtext.index(REMOTE_MARK):], 'remote_hook', 'exec'), ns)
        return spec_mod, RemoteDSparkCapacitySpeculator, RemoteDSparkLocalSpeculator
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path.remove(str(tmp_path / 'pkg'))
        sys.modules.pop('vllm.v1.worker.gpu.spec_decode.dspark.speculator', None)


def batch(reqs: list[tuple[str, int, int]]):
    """reqs: (req_id, first_pos, n_rows)."""
    ids, pos, qsl = [], [], [0]
    for _, p0, n in reqs:
        pos += list(range(p0, p0 + n))
        ids += [3 + p for p in range(p0, p0 + n)]
        qsl.append(len(ids))
    aux = [torch.full((len(ids), 5120), float(L + 1), dtype=torch.bfloat16) for L in range(3)]
    ib = types.SimpleNamespace(req_ids=[r[0] for r in reqs], num_reqs=len(reqs), num_tokens=len(ids),
                               query_start_loc_np=np.array(qsl, dtype=np.int32),
                               input_ids=torch.tensor(ids, dtype=torch.int32),
                               positions=torch.tensor(pos, dtype=torch.int64))
    return ib, aux


def call(obj, ib, aux, rejected, **kw):
    return obj.propose(ib, None, None, None, aux, None, rejected, None, None, None, None, **kw)


def test_unset_variable_is_inert(tmp_path: Path) -> None:
    os.environ.pop('VLLM_DSPARK_DECODE_CAPTURE_DIR', None)
    before = {t.name for t in threading.enumerate()}
    mod, Cap, Loc = build(tmp_path, {})
    assert mod.DSparkSpeculator.propose is FakeBase.propose
    assert 's26pkg.decode_capture' not in sys.modules and not hasattr(mod, '_d2c_capture')
    assert Cap.propose.__qualname__.endswith('RemoteDSparkCapacitySpeculator.propose')
    assert Loc.propose.__qualname__.endswith('RemoteDSparkLocalSpeculator.propose')
    assert {t.name for t in threading.enumerate()} == before


def test_capacity_and_local_hooked_once_untagged_never_touched(tmp_path: Path) -> None:
    cap_dir = tmp_path / 'cap'
    cap_dir.mkdir(mode=0o700)
    mod, Cap, Loc = build(tmp_path, {'VLLM_DSPARK_DECODE_CAPTURE_DIR': str(cap_dir)})
    capture = mod._d2c_capture
    try:
        # Untagged production batch: tensors, rejected counts and aux are poison; only req_ids is read.
        ib = types.SimpleNamespace(req_ids=['cmpl-1b2c-0-deadbeef', 'chatcmpl-abc-12345678'], num_reqs=2,
                                   num_tokens=Poison(), query_start_loc_np=Poison(), input_ids=Poison(),
                                   positions=Poison())
        assert Cap().propose(ib, None, None, None, Poison(), None, Poison(), None, None, None, None) == 'remote'
        assert Loc().propose(ib, None, None, None, Poison(), None, Poison(), None, None, None, None) == 'drafts'
        assert capture.round == 0
        # Tagged rows through each production entry point: exactly one ledger row per round.
        ib, aux = batch([('cmpl-d2c.pC.0.0-0-1a2b3c4d', 0, 4), ('ndm-legal', 0, 3)])
        assert call(Cap(), ib, aux, torch.tensor([1, 0])) == 'remote'
        ib, aux = batch([('chatcmpl-d2c.pL.0.0-1a2b3c4d', 0, 4)])
        assert call(Loc(), ib, aux, torch.tensor([0])) == 'drafts'
        ib, aux = batch([('cmpl-d2c.pD.0.0-0-1a2b3c4d', 0, 4)])
        call(Cap(), ib, aux, torch.tensor([0]), dummy_run=True)
    finally:
        capture.close()
    assert sorted(p.name for p in cap_dir.iterdir()) == ['ERROR_COUNT.json', 'pC', 'pL']
    for sid, rows in (('pC', 3), ('pL', 4)):
        ledger = (cap_dir / sid / 'ledger.jsonl').read_text().splitlines()
        assert len(ledger) == 1 and json.loads(ledger[0])['rows'] == rows
    assert not (cap_dir / 'ndm-legal').exists() and not (cap_dir / 'ERRORS.jsonl').exists()


def test_untagged_hot_path_cost(tmp_path: Path) -> None:
    """MEASURED on Controller CPU: the whole per-round cost for 4 untagged requests (regex over req_ids)."""
    cap_dir = tmp_path / 'cap'
    cap_dir.mkdir(mode=0o700)
    mod, Cap, _ = build(tmp_path, {'VLLM_DSPARK_DECODE_CAPTURE_DIR': str(cap_dir)})
    try:
        ib = types.SimpleNamespace(req_ids=[f'cmpl-{i:032x}-0-{i:08x}' for i in range(4)], num_reqs=4)
        bare = Cap.__mro__[0].propose.__wrapped__
        obj = Cap()
        args = (ib, None, None, None, None, None, None, None, None, None, None)
        n = 20000
        t0 = time.perf_counter()
        for _ in range(n):
            obj.propose(*args)
        wrapped = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            bare(obj, *args)
        plain = (time.perf_counter() - t0) / n
    finally:
        mod._d2c_capture.close()
    print(f'S26 MEASURED untagged wrapper overhead: {1e6 * (wrapped - plain):.2f} us per round (4 requests)')
    assert wrapped - plain < 50e-6
