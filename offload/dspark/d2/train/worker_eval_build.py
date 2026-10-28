"""Isolated real-worker adapter constructor; no production selector changes."""
from __future__ import annotations

import json
from ...worker.server import DraftCore, expected_hello

def build(root: str, adapter: str, native_root: str, *,
                                   emulated: bool = True, graphs: bool = True,
                                   max_model_len: int = 327680) -> tuple[DraftCore, object]:
    """Isolated D2 evaluation entry point; never selected by the service CLI.

    Uses only the authorized solo card(s), native BF16 dense/head and native
    MXFP4 experts, the same TP2 backend, graph capture and wire state machine.
    """
    import hashlib
    from pathlib import Path
    from typing import cast
    import torch
    from offload.dspark.d2.train.model import Config
    from offload.dspark.d2.train.replay import read_checkpoint
    from offload.dspark.d2.train.worker_adapter import Worker, install, adapted_identity
    from offload.dspark.d2.train.worker_eval_lease import require_evaluation_lease
    from offload.dspark.native.identity import compute as native_identity
    from offload.dspark.native.ops import MarkovRowsFP32
    from offload.dspark.native.tp2_native import NativeTPDrafter
    from ...worker.checkpoint import Checkpoint
    from ...worker.identity import compute as base_identity
    from ...worker.tp2 import TP2Backend

    require_evaluation_lease(emulated)  # before any CUDA work
    state, adapter_hash = read_checkpoint(Path(adapter))
    if state.get("schema") != "dsv41.d2.train-checkpoint.v1" or state.get("model_config") is None:
        raise ValueError("adapter evaluation requires a full D2 checkpoint")
    config = Config(**state["model_config"])
    if config != Config():
        raise ValueError("real worker requires the exact released 1017 shape")
    ck = Checkpoint(root)
    profile = "native-hybrid-head-hot"
    # Verify the full native receipt through an already registered policy.
    # This is provenance validation, not a claim that its K6 head is used here.
    ident = native_identity("native-expanded", base_identity(root, ck.manifest), native_root=native_root)
    from dataclasses import asdict
    from offload.dspark.native.arms import ARMS
    ident["evaluation_base_config"] = asdict(ARMS[profile])
    ident["provenance_only_profile"] = ident.pop("arm")
    devs = [torch.device("cuda", 0), torch.device("cuda", 0 if emulated else 1)]
    model = NativeTPDrafter(ck, devs, profile, native_root=native_root, slots=8, max_model_len=max_model_len)
    frozen = hashlib.sha256(json.dumps(model.sh.manifest, sort_keys=True).encode()).hexdigest()
    if state.get("frozen_identity") != frozen:
        raise ValueError("adapter source differs from verified native weights")
    tokens_path = Path(root)/"markov_top_tokens.json"
    tokens = json.loads(tokens_path.read_text()) if tokens_path.exists() else []
    install(cast(Worker, model), state["trainable"], config, state["run"]["rank"], MarkovRowsFP32, tokens)
    ident = adapted_identity(ident, adapter_hash, config)
    exp = expected_hello(bytes.fromhex(str(ident["drafter_identity"])), bytes.fromhex(str(ident["markov_identity"])),
                         max_model_len=max_model_len)
    if graphs:
        model.capture(batch_sizes=(1,), ks=(5,), insert_rows=(8, 16, 32, 64, 128), n_outs=(5,))
    class EvaluationBackend(TP2Backend):
        arm_identity: dict[str, object]

    backend = EvaluationBackend(model, exp.max_rows_per_step, exp.row_bytes, graphs=graphs, max_slots=exp.max_slots)
    backend.arm_identity = ident
    core = DraftCore(exp, backend, worker_name="d2-adapted-native-evaluation"+("-emulated" if emulated else "-solo-tp2"))
    return core, model

