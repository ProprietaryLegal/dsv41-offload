"""CPU startup and contract checks, with a recording GPU factory boundary."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worker import server
from worker.arm_selection import SERVING_ARMS, selected_arm


def test_arm_policy_fails_before_cuda(monkeypatch):
    monkeypatch.setenv("DSPARK_WORKER_ARM", "native-typo")
    with pytest.raises(ValueError, match="no fallback"):
        server.build_core_tp2("missing")
    monkeypatch.setenv("DSPARK_WORKER_ARM", "native-packed")
    with pytest.raises(ValueError, match="requires --tp 2"):
        server.build_core("missing")
    monkeypatch.delenv("DSPARK_WORKER_ARM")
    assert selected_arm() == "k2-tp2"


@pytest.fixture
def boundary(monkeypatch):
    import torch
    import worker.checkpoint
    import worker.identity
    import worker.tp2
    import offload.dspark.native.identity as identity

    model = SimpleNamespace(capture=Mock())
    native = Mock(return_value=model)
    control = Mock(return_value=model)
    ident = {"drafter_identity": "ab" * 32, "markov_identity": "cd" * 32,
             "arm": "native", "draft_head": "K6"}
    native_id = Mock(return_value=ident)
    monkeypatch.setattr(worker.checkpoint, "Checkpoint", lambda root: SimpleNamespace(manifest={}))
    monkeypatch.setattr(worker.identity, "compute", lambda *a: {**ident, "drafter_identity": "ef" * 32})
    monkeypatch.setattr(identity, "compute", native_id)
    monkeypatch.setattr(worker.tp2, "check_visible", Mock())
    monkeypatch.setattr(worker.tp2, "TPDrafter", control)
    monkeypatch.setattr(worker.tp2, "TP2Backend", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 0)
    monkeypatch.setitem(sys.modules, "offload.dspark.native.tp2_native",
                        SimpleNamespace(NATIVE_ROOT="/official", build=native))
    return SimpleNamespace(model=model, native=native, control=control, native_id=native_id)


@pytest.mark.parametrize("arm", SERVING_ARMS)
def test_worker_factory_identity_prewarm_and_bounds(arm, boundary, monkeypatch):
    monkeypatch.setenv("DSPARK_WORKER_ARM", arm)
    monkeypatch.setenv("DSPARK_WORKER_TP_EMULATE", "1")
    core, model = server.build_core_tp2("/checkpoint", max_model_len=1048576)
    assert core.expected.max_slots == 4 and len(core.slots) == 4
    assert core.expected.max_model_len == 1048576
    model.capture.assert_called_once_with(batch_sizes=(1, 2, 3, 4), ks=(5,),
        insert_rows=(8, 16, 32, 64, 128), n_outs=(1, 2, 3, 4, 5))
    if arm == "k2-tp2":
        boundary.native.assert_not_called()
        boundary.native_id.assert_not_called()
        assert core.worker_name == "e-dspark-3090x2-emulated"
        assert core.expected.drafter_identity == bytes.fromhex("ef" * 32)
    else:
        boundary.control.assert_not_called()
        assert boundary.native.call_args.args[0] == arm
        assert boundary.native.call_args.kwargs == dict(slots=8, max_model_len=1048576, native_root="/official")
        assert core.expected.drafter_identity == bytes.fromhex("ab" * 32)
        assert core.backend.arm_identity["draft_head"] == "K6"
        assert core.worker_name == f"{arm}-K6-emulated"


def test_missing_shards_or_prepack_never_builds_control(boundary, monkeypatch):
    monkeypatch.setenv("DSPARK_WORKER_ARM", "native-hybrid-kx2")
    boundary.native_id.side_effect = FileNotFoundError("missing official shard")
    with pytest.raises(FileNotFoundError, match="missing official shard"):
        server.build_core_tp2("/checkpoint")
    boundary.native.assert_not_called()
    boundary.native_id.side_effect = None
    boundary.native.side_effect = RuntimeError("prepack broke")
    with pytest.raises(RuntimeError, match="startup refused: prepack broke"):
        server.build_core_tp2("/checkpoint")
    boundary.control.assert_not_called()
    boundary.model.capture.assert_not_called()
