"""Exercise the real service builder, with CUDA/model boundaries replaced on CPU."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
from offload.dspark.worker import checkpoint, identity, server, tp2
from offload.dspark.worker import protocol as P


@pytest.mark.parametrize('sampled', [False, True])
@pytest.mark.parametrize('graphs', [None, False])
def test_actual_builder_preserves_callers_graph_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    sampled: bool, graphs: bool | None,
) -> None:
    monkeypatch.setenv('DSPARK_SAMPLED_TOP32', '1' if sampled else '0')
    monkeypatch.setenv('DSPARK_WORKER_ARM', 'k2-tp2')
    monkeypatch.delenv('DSPARK_WORKER_TP_EMULATE', raising=False)
    monkeypatch.delenv('DSPARK_V1_DUMP_DIR', raising=False)
    captures: list[dict[str, object]] = []
    devices: list[bool] = []

    def capture(**options: object) -> None:
        captures.append(options)

    model = SimpleNamespace(capture=capture)

    def make_model(ck: object, devs: list[torch.device], **options: object) -> object:
        assert devs == [torch.device('cuda:0'), torch.device('cuda:1')]
        assert options == {'slots': 8, 'max_model_len': 327680}
        return model

    def backend(model_arg: object, rows: int, row_bytes: int, *, graphs: bool, max_slots: int) -> object:
        assert model_arg is model and (rows, row_bytes, max_slots) == (512, 30720, 4)
        return SimpleNamespace(graphs=graphs)

    monkeypatch.setattr(checkpoint, 'Checkpoint', lambda root: SimpleNamespace(manifest={}))
    monkeypatch.setattr(identity, 'compute', lambda *args: {'drafter_identity': 'ab'*32, 'markov_identity': 'cd'*32})
    monkeypatch.setattr(tp2, 'check_visible', lambda *, pair: devices.append(pair))
    monkeypatch.setattr(tp2, 'TPDrafter', make_model)
    monkeypatch.setattr(tp2, 'TP2Backend', backend)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda dev: 0)
    with caplog.at_level(logging.INFO, logger='dspark_worker'):
        if graphs is None:
            core, actual_model = server.build_core_tp2('unused-fake-root')
        else:
            core, actual_model = server.build_core_tp2('unused-fake-root', graphs=graphs)
    resolved = graphs is not False
    assert actual_model is model and core.backend.graphs is resolved
    assert devices == [True]
    assert core.expected.sample_mode == (P.SAMPLE_TOP32 if sampled else P.SAMPLE_GREEDY)
    assert captures == ([{'batch_sizes': (1, 2, 3, 4), 'ks': (5,),
                         'insert_rows': (8, 16, 32, 64, 128), 'n_outs': (1, 2, 3, 4, 5)}] if resolved else [])
    assert f'graphs={resolved}' in caplog.text


def test_service_entry_point_is_part_of_physical_probe_identity() -> None:
    assert 'offload/dspark/worker/server.py' in SOURCES
