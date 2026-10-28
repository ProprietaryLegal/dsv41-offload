"""CPU checks of real admission seams and separately prepared production sources."""
from __future__ import annotations

import json
import struct
import sys
import time
import types
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.dspark.spark.remote_dspark import client as real_client
from offload.launch.windows import pp4_admission_engine as e
from offload.launch.windows import pp4_admission_prepare as p
from offload.launch.windows import pp4_prepare as base


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('VLLM_PP4_ADMISSION_RUN', 'owned-run')
    monkeypatch.setenv('VLLM_PP4_ROUND_DIR', str(tmp_path))
    e._seen.clear()
    return tmp_path


def fault(root: Path, mode: str) -> None:
    (root/'FAULT.json').write_text(json.dumps({'id': mode, 'mode': mode, 'run': 'owned-run', 'expires': time.time()+60}))


def test_no_env_means_no_fault_or_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('VLLM_PP4_ADMISSION_RUN', raising=False)
    e.client_before(object()); e.client_after(object(), bytearray()); e.invalidate(object(), object())
    assert e.marker() is None


@pytest.mark.parametrize('bad', ['owner', 'expiry', 'mode'])
def test_bad_markers_refuse(seam: Path, bad: str) -> None:
    fault(seam, 'timeout')
    data = json.loads((seam/'FAULT.json').read_text())
    data[{'owner': 'run', 'expiry': 'expires', 'mode': 'mode'}[bad]] = {'owner': 'foreign', 'expiry': 0, 'mode': 'unknown'}[bad]
    (seam/'FAULT.json').write_text(json.dumps(data))
    with pytest.raises(ValueError): e.marker()


@pytest.mark.parametrize('origin', [0, 1])
def test_both_ranks_invalidate_from_one_rank(seam: Path, monkeypatch: pytest.MonkeyPatch, origin: int) -> None:
    fault(seam, 'q-rank'+str(origin))
    for rank in (0, 1):
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda rank=rank: rank)
        def reduce(flag: torch.Tensor, op: object, rank: int = rank) -> None:
            assert int(flag.item()) == int(rank == origin)
            flag.fill_(1)
        monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
        valid = torch.ones(4, dtype=torch.bool)
        runner = types.SimpleNamespace(speculator=types.SimpleNamespace(sampled_valid=valid))
        e.invalidate(runner, types.SimpleNamespace(idx_mapping=torch.tensor([1, 3])))
        assert valid.tolist() == [True, False, True, False]


@pytest.mark.parametrize('corrupt', ['none', 'token', 'length', 'proposal'])
def test_exact_target_only_verifier_coupling(seam: Path, monkeypatch: pytest.MonkeyPatch, corrupt: str) -> None:
    fault(seam, 'q-rank0')
    monkeypatch.setenv('VLLM_DSPARK_SAMPLED_TOP32', '1')
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    def verify(draft_logits: object, draft_sampled: torch.Tensor, expanded_local_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = 8 if draft_logits is None or corrupt != 'token' else 7
        length = 2 if draft_logits is not None and corrupt == 'length' else 1
        return torch.tensor([[token, 0]]), torch.tensor([length])
    call = e.verify(verify)
    if corrupt == 'none':
        call(torch.zeros(1), torch.tensor([0, -1]), torch.tensor([0, 1]))
        rows = [json.loads(s) for f in seam.glob('*.jsonl') for s in f.read_text().splitlines()]
        assert rows[-1]['exact'] and rows[-1]['lengths_one']
    else:
        with pytest.raises((ValueError, AssertionError)):
            call(torch.zeros(1), torch.tensor([0, 1 if corrupt == 'proposal' else -1]), torch.tensor([0, 1]))
    assert __import__('os').environ['VLLM_DSPARK_SAMPLED_TOP32'] == '1'


def test_prepare_keeps_ordinary_receipt_unchanged_and_hooks_actual_client_try(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = p.prepare(Path('/opt/dsv41/data/src-image'), tmp_path/'prepared')
    ordinary = base.prepare(Path('/opt/dsv41/data/src-image'), tmp_path/'ordinary')
    assert ordinary == json.loads((base.ROOT/'offload/dspark/probabilistic/pp4/verifier-k1-receipt.json').read_text())
    assert receipt['admission_only'] is True
    runner = (tmp_path/'prepared'/pp3.RUNNER).read_text()
    assert runner.index('_pp4_admission.invalidate(self, input_batch)') < runner.index('pp3_valid =')
    for name in ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu'):
        module = types.ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu.dsv41_pp4_admission', e)
    monkeypatch.setitem(sys.modules, real_client.__package__+'.'+p.SEAM.removesuffix('.py'), e)  # R16 client seam
    # Execute actual composed client with the same imports as production. Fault
    # callback raises inside its real TransportTimeout handler, not a fake pass.
    module = types.ModuleType(real_client.__name__+'.admission_fixture')
    module.__package__ = real_client.__package__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile((tmp_path/'prepared/remote_dspark/client.py').read_text(), 'prepared-client', 'exec'), module.__dict__)  # noqa: S102 -- execute exact separately prepared pinned-source client
    assert callable(module.RemoteDraftClient)
    assert 'except TransportTimeout:' in (tmp_path/'prepared/remote_dspark/client.py').read_text()


def test_real_timeout_exception_and_late_sequence(seam: Path) -> None:
    client = object.__new__(real_client.RemoteDraftClient)
    client.session = 7; client.epoch = 1
    fault(seam, 'timeout')
    from offload.dspark.spark.remote_dspark.transport import TransportTimeout
    with pytest.raises(TransportTimeout): e.client_after(client, bytearray(32))
    fault(seam, 'late-reply')
    reply = bytearray(32); struct.pack_into('<Q', reply, 8, 41)
    e.client_after(client, reply)
    assert struct.unpack_from('<Q', reply, 8)[0] == 40
    e.client_after(client, reply)
    assert struct.unpack_from('<Q', reply, 8)[0] == 40  # once only


@pytest.mark.parametrize('mode,reason', [('timeout', 'timeout'), ('late-reply', 'bad_reply'), ('reset', 'not_ready'), ('reconnect', 'not_ready')])
def test_actual_prepared_client_steps_refuse_then_reconnect(seam: Path, monkeypatch: pytest.MonkeyPatch, mode: str, reason: str) -> None:
    import numpy as np

    from offload.dspark.spark.remote_dspark import wire
    from offload.dspark.spark.remote_dspark.config import RemoteConfig
    from offload.dspark.spark.remote_dspark.mock_worker import WorkerCore
    from offload.dspark.spark.remote_dspark.transport import RpcClient
    # Execute exact prepared client, with an in-memory real protocol worker.
    # No socket/listener, mock outcome, or external host call is involved.
    p.prepare(Path('/opt/dsv41/data/src-image'), seam/'prepared')
    for name in ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu'):
        module = types.ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu.dsv41_pp4_admission', e)
    monkeypatch.setitem(sys.modules, real_client.__package__+'.'+p.SEAM.removesuffix('.py'), e)  # R16 client seam
    module = types.ModuleType(real_client.__name__)
    module.__package__ = real_client.__package__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile((seam/'prepared/remote_dspark/client.py').read_text(), 'prepared-client', 'exec'), module.__dict__)  # noqa: S102 -- execute exact separately prepared pinned-source client
    identity = bytes(range(32))
    hello = wire.Hello(hidden=4, row_bytes=24, drafter_identity=identity)
    core = WorkerCore(hello)
    class Transport(RpcClient):
        def connect(self, timeout_s: float) -> None: pass
        def close(self) -> None: pass
        def request(self, msg: object, resp_buf: bytearray, timeout_s: float) -> int:
            response = core.handle(msg)
            assert response is not None
            resp_buf[:len(response)] = response
            return len(response)
    cfg = RemoteConfig.from_env({'VLLM_DSPARK_REMOTE': 'tcp://127.0.0.1:1', 'VLLM_DSPARK_REMOTE_IDENTITY': identity.hex()})
    cls = cast(type[real_client.RemoteDraftClient], module.RemoteDraftClient)
    client = cls(cfg, hello, background=False, transport_factory=lambda endpoint: Transport())
    try:
        assert client.connect_now() and client.epoch == 1
        fault(seam, mode)
        entries = [wire.ReqEntry(0, 1, 0, 2, 3, 2, flags=wire.RF_NEED_DRAFT | wire.RF_NEW_GEN)]
        rows = np.ones((2, 24), np.uint8)
        def step(seq: int) -> real_client.StepOutcome:
            offset = wire.APP_HEADER_SIZE+wire.rows_offset_for(1)
            buf = bytearray(offset+rows.nbytes); buf[offset:] = rows.tobytes()
            return client.step(buf, entries, 2, step_id=seq)
        rejected = step(1)
        assert not rejected.ok and rejected.reason == reason
        assert rejected.unknown_apply == (mode in ('timeout', 'late-reply'))
        assert not client.ready
        assert client.connect_now() and client.epoch == 2
        assert step(2).ok
    finally:
        client.close()


def test_eos_observer_sees_defined_tokens_only(seam: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (seam/'EOS.json').write_text(json.dumps({'run': 'owned-run', 'expires': time.time()+60, 'eos': 8}))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    output = types.SimpleNamespace(sampled_token_ids=torch.tensor([[7, 8]]), num_sampled=torch.tensor([1]))
    e.observe_sample(output)
    assert not list(seam.glob('*.jsonl'))
    output.num_sampled.fill_(2)
    e.observe_sample(output)
    evidence = json.loads(next(seam.glob('*.jsonl')).read_text())
    assert evidence['eos'] == 8 and evidence['observed'] is True and evidence['rank'] == 1


def test_import_in_cpu_proxy_does_not_import_torch() -> None:
    import subprocess
    subprocess.run([sys.executable, '-c', 'import sys; from offload.launch.windows import pp4_admission_engine; assert "torch" not in sys.modules'], check=True)
