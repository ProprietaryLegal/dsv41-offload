from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

from offload.dspark.d2.train import worker_eval
from offload.dspark.d2.train.data import D2Blocks, SyntheticBlocks
from offload.dspark.d2.train.pilot_sample import RoundKey
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.worker.server import DraftCore, FakeBackend, expected_hello
from offload.dspark.d2.train.worker_eval_lease import require_evaluation_lease
from offload.dspark.d2.train.safety import GPU, SOLO_B


def test_same_sample_real_process_proxy_tcp_worker_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local test transport uses the repository's explicit loopback test switch.
    monkeypatch.setenv("DSPARK_WORKER_ALLOW_LOOPBACK", "1")
    config = model().c
    hello = expected_hello(b"a"*32, b"b"*32, hidden=config.dim, row_bytes=config.dim*6,
                           window=config.window, vocab_size=config.vocab, noise_token_id=config.noise)
    def draft(window: object, anchor: int, position: int, n: int, vocab: int) -> tuple[object, object]:
        return np.array([(anchor+i) % vocab for i in range(n)], dtype=np.int32), np.full(n, .5, dtype=np.float32)
    backend = FakeBackend(hello.window, hello.row_bytes, hello.max_slots, draft, hello.vocab_size)
    core = DraftCore(hello, backend)
    predictor = worker_eval.ProcessPredictor(core, "127.0.0.1")
    assert predictor.round.proxy is not None
    try:
        for index in (0, 2, 0):  # full-window reset/reuse with different request ids
            inp = SyntheticBlocks(config, 3)[index]["inputs"]
            actual = predictor(inp, f"test-{index}-{id(inp)}")
            expected = torch.arange(3, 8)[None]
            assert torch.equal(actual, expected)
    finally:
        predictor.close()
    assert predictor.region.closed


def test_comparison_uses_same_complete_keys_and_bonus_excluded_prefixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = model().eval()
    data = SyntheticBlocks(m.c, 2)
    keys: list[RoundKey] = [{"request": "synthetic-0", "P": 12}, {"request": "synthetic-1", "P": 13}]
    monkeypatch.setattr(worker_eval, "complete_rounds", lambda *_: list(enumerate(keys)))
    refs = worker_eval.reference_predictions(m, cast(D2Blocks, data), tmp_path/"k2", keys)
    calls: list[str] = []
    def same(inp: dict[str, Tensor], request: str) -> Tensor:
        calls.append(request)
        return refs[int(inp["P"][0])-12].clone()
    report = worker_eval.compare(cast(D2Blocks, data), tmp_path/"k2", keys, refs, same)
    assert report["rounds"] == report["exact_blocks"] == 2
    assert report["position_matches"] == [2]*5 and report["bonus_excluded"]
    assert report["exact_path_EA"] == report["worker_EA"]
    assert len(calls) == 2 and all("worker-eval/" in s for s in calls)
    with pytest.raises(ValueError, match="exact frozen pilot sample"):
        worker_eval.compare(cast(D2Blocks, data), tmp_path/"k2", list(reversed(keys)), refs, same)


def test_worker_eval_refuses_production_pair_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c4e7a371,GPU-7744b44b")
    monkeypatch.setenv("D2_ADAPTER_EVAL", "1")
    with pytest.raises(RuntimeError, match="only both named solos"):
        require_evaluation_lease(False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU+","+SOLO_B)
    monkeypatch.delenv("D2_ADAPTER_EVAL")
    with pytest.raises(RuntimeError, match="explicit evaluation flag"):
        require_evaluation_lease(False)
