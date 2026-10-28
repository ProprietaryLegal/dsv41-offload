from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from offload.tv1 import build_overlay
from offload.tv1.candidate_cases import (CACHE_OFFSET, EXCLUSIONS, LAYERS, MAX_RAW, STRIDE,
                                        case_matrix, validate_engine_receipts,
                                        validate_oracle_receipt)
from offload.tv1.candidate_metadata import consumer_layer, map_decode, score_decode


def receipt() -> dict[str, Any]:
    return {'type': 'tv1-installed-oracle-v2', 'classification': 'MEASURED SM121',
            'all_identical': True, 'capability': [12, 1], 'exclusions': EXCLUSIONS,
            'cases': [{**s, 'eager_identical': True, 'replay_identical': True} for s in case_matrix()]}


def engine_receipts() -> list[dict[str, Any]]:
    rows = [{'type': 'tv1-engine-oracle-v2', 'layer': layer, 'batch': batch,
             'raw_context': 300100 if batch == 4 else 150100, 'rows': batch * 6,
             'next_n': 1, 'stride': STRIDE, 'cache_offset': CACHE_OFFSET,
             'width': MAX_RAW, 'candidate_capacity': 2048, 'page': 64,
             'layout': 'flattened', 'backend': 'persistent', 'identical': True}
            for layer in LAYERS for batch in range(1, 5)]
    rows += [{**r, 'raw_context': 32000} for r in rows if r['batch'] == 4]
    return rows


def test_geometry_is_production_derived_not_raw_equals_compressed_guess() -> None:
    matrix = case_matrix()
    assert len(matrix) == 288
    assert len({s['id'] for s in matrix}) == 288
    assert {s['layer'] for s in matrix} == set(LAYERS)
    assert {s['batch'] for s in matrix} == {1, 2, 3, 4}
    assert {s['numeric'] for s in matrix} == {'random', 'tie', 'near-tie'}
    assert {s['layout'] for s in matrix} == {'single', 'native2', 'flat3', 'flat4', 'flat6', 'flat6-padded'}
    for s in matrix:
        assert s['width'] == MAX_RAW // s['compress_ratio']
        assert s['visible_width'] == s['raw_context'] // s['compress_ratio']
        assert s['high_page'] * s['stride'] > 2**31
        assert s['stride'] == 138240 // 3 and s['cache_offset'] == 37440
    validate_oracle_receipt(receipt())
    validate_engine_receipts(engine_receipts())


@pytest.mark.parametrize('change', ['old', 'missing', 'duplicate', 'stride', 'context', 'B3', 'false', 'tie', 'bool'])
def test_oracle_receipt_rejects_unqualified_coverage(change: str) -> None:
    r = receipt()
    cases = r['cases']
    if change == 'old':
        r['type'] = 'tv1-installed-oracle'
    elif change == 'missing':
        cases.pop()
    elif change == 'duplicate':
        cases[-1] = copy.deepcopy(cases[-2])
    elif change == 'stride':
        cases[0]['stride'] = 64 * 132 + 256
    elif change == 'context':
        cases[-1]['raw_context'] = 150000
    elif change == 'B3':
        cases[next(i for i, s in enumerate(cases) if s['batch'] == 3)]['batch'] = 4
    elif change == 'false':
        cases[0]['replay_identical'] = False
    elif change == 'tie':
        cases[0]['numeric'] = 'random-only'
    else:
        cases[0]['batch'] = True
    with pytest.raises(ValueError, match='TV1'):
        validate_oracle_receipt(r)


@pytest.mark.parametrize('change', ['missing-B3', 'short-B4', 'false', 'stride', 'layer', 'graph', 'backend'])
def test_engine_receipts_fail_closed(change: str) -> None:
    r = engine_receipts()
    if change == 'missing-B3':
        r = [s for s in r if s['batch'] != 3]
    elif change == 'short-B4':
        for s in r:
            if s['batch'] == 4:
                s['raw_context'] = 140000
    elif change == 'false':
        r[0]['identical'] = False
    elif change == 'stride':
        r[0]['stride'] += 256
    elif change == 'layer':
        r[0]['layer'] = 20
    elif change == 'graph':
        r[0]['layout'] = 'capture-only'
    else:
        r[0]['backend'] = 'per_row'
    with pytest.raises(ValueError, match='TV1'):
        validate_engine_receipts(r)


@pytest.mark.parametrize('batch,next_n,padding', [(1, 1, 0), (2, 2, 0), (3, 1, 0), (4, 2, 0), (24, 1, 1)])
def test_exact_patch_mapping_request_ownership(batch: int, next_n: int, padding: int) -> None:
    q = torch.arange(batch * next_n * 32 * 128).reshape(batch, next_n, 32, 128)
    lens = torch.arange(batch * next_n, dtype=torch.int32).reshape(batch, next_n)
    if padding:
        lens[-1] = 0
    pages = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4) + 47000
    candidates = torch.arange((batch * next_n + 3) * 8, dtype=torch.int32).reshape(-1, 8)
    fq, fl, fp, fc = map_decode(q, lens, pages, candidates)
    for row in range(batch * next_n):
        assert torch.equal(fq[row], q[row // next_n, row % next_n])
        assert fl[row] == lens[row // next_n, row % next_n]
        assert torch.equal(fp[row], pages[row // next_n])
        assert torch.equal(fc[row], candidates[row])


def test_mapping_refuses_unqualified_implicit_length_repeat_and_native_six() -> None:
    q = torch.zeros(3, 2, 32, 128)
    pages = torch.zeros(3, 2, dtype=torch.int32)
    candidates = torch.zeros(6, 2048, dtype=torch.int32)
    with pytest.raises(ValueError, match='metadata expansion'):
        map_decode(q, torch.ones(3, 1, dtype=torch.int32), pages, candidates)
    with pytest.raises(ValueError, match='metadata expansion'):
        map_decode(q.reshape(1, 6, 32, 128), torch.ones(1, 6, dtype=torch.int32), pages[:1], candidates)


def test_loaded_geometry_cannot_silently_use_dense_synthetic_stride() -> None:
    q = torch.zeros(1, 1, 32, 128).to(torch.float8_e4m3fn)
    cache = torch.zeros(2, 64, 1, 132, dtype=torch.uint8)
    with pytest.raises(ValueError, match='loaded K1 consumer geometry'):
        score_decode(q, cache, torch.zeros(1, 32), torch.ones(1, 1, dtype=torch.int32),
                     torch.zeros(1, 5120, dtype=torch.int32), torch.zeros(1, 2048, dtype=torch.int32), MAX_RAW, 8)


def test_layer_resolution_uses_actual_op_owner_not_shared_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    op = SimpleNamespace()
    forward = ModuleType('vllm.forward_context')
    cast(Any, forward).get_forward_context = lambda: SimpleNamespace(
        no_compile_layers={'model.layers.24.attn': SimpleNamespace(indexer=SimpleNamespace(indexer_op=op)),
                           'model.layers.20.attn.indexer.k_cache': SimpleNamespace()})
    monkeypatch.setitem(sys.modules, 'vllm.forward_context', forward)
    assert consumer_layer(op) == 24
    assert consumer_layer(op) == 24
    with pytest.raises(ValueError, match='uniquely'):
        consumer_layer(SimpleNamespace())


def test_overlay_executes_shared_adapter_and_explicit_dense_exclusions(tmp_path: Path) -> None:
    manifest = build_overlay.build(tmp_path / 'candidate', ('candidate',))
    assert len(manifest['files']) == 6
    source = (tmp_path / 'candidate/vllm/model_executor/layers/sparse_attn_indexer.py').read_text()
    assert 'logits = score_decode(' in source
    assert 'check_actual(' in source
    assert 'and not needs_padded_path and decode_metadata.indices is None' in source
    assert 'consumer_layer(self)' in source
    off = tmp_path / 'off'
    build_overlay.build(off)
    for name, baseline in build_overlay.BASES.items():
        assert (off / name).read_bytes() == baseline.read_bytes()


def test_actual_equality_mismatch_stops_and_bound_skips_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.tv1 import candidate_metadata as meta
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(meta, '_CHECKED', set())
    mask = ModuleType('vllm.model_executor.kernels.attention.dsa.candidate_blocks')
    cast(Any, mask).apply_candidate_mask = lambda *a: None
    deep = ModuleType('vllm.utils.deep_gemm')
    cast(Any, deep).fp8_fp4_paged_mqa_logits = lambda *a, **kw: torch.zeros(1, 8)
    top = ModuleType('vllm.model_executor.layers.indexer_topk')
    class Select:
        def resolve_backend(self, *args: object) -> str:
            return 'persistent'
        def __call__(self, logits: torch.Tensor, lens: torch.Tensor, next_n: int,
                     output: torch.Tensor, topk: int, maximum: int) -> None:
            output.fill_(int(logits[0, 0].item()))
    cast(Any, top).get_indexer_topk = lambda backend: Select()
    for module in (mask, deep, top):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    q = torch.zeros(1, 1, 32, 128)
    lens = torch.tensor([[150100]], dtype=torch.int32)
    pages = torch.zeros(1, 1, dtype=torch.int32)
    blocks = torch.zeros(1, 2048, dtype=torch.int32)
    args = (q, torch.zeros(1), torch.zeros(1, 32), lens, pages, blocks, torch.zeros(1),
            MAX_RAW, 150100, 24, 1, 512, 'auto')
    with pytest.raises(ValueError, match='mismatch: STOP'):
        meta.check_actual(torch.ones(1, 8), *args)
    assert not meta._CHECKED
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    meta.check_actual(torch.ones(1, 8), *args)
    assert not meta._CHECKED


def test_qualification_traffic_uses_existing_public_cuts_and_is_dry(capsys: pytest.CaptureFixture[str]) -> None:
    from offload.tv1 import qualification
    assert qualification.main([]) == 0
    assert '"executed": false' in capsys.readouterr().out
    assert [c.batch for c in qualification.CELLS] == [1, 2, 3, 4, 4]
    assert [c.prompts[0] for c in qualification.CELLS] == ['150k', '32k-0', '32k-0', '32k-0', '150k']
    assert all(not c.captures and c.warmup for c in qualification.CELLS)
    assert qualification.ESTIMATE_S == 735


def test_eager_qualification_returns_to_graphs_and_never_traps_short_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.tv1 import candidate_metadata as meta
    monkeypatch.setattr(meta, '_CHECKED', set())
    assert not meta.needs_eager_qualification(1, 1024)
    assert meta.needs_eager_qualification(3, 32000)
    for layer in LAYERS:
        meta._CHECKED.add((layer, 3, meta.context_bucket(32000)))
    assert not meta.needs_eager_qualification(3, 32000)
    assert meta.needs_eager_qualification(3, 150000)
    assert meta.needs_eager_qualification(4, 32000)
    with pytest.raises(ValueError, match='B1..B4'):
        meta.needs_eager_qualification(5, 32000)
