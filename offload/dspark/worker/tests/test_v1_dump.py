"""CPU tests for the opt-in dump's provenance and fail-closed comparisons."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worker.v1_dump import V1Dump, chosen_rounds
from offload.dspark.native.tools.n3.classify import verdict


def model():
    rank = SimpleNamespace(norm=torch.ones(4), stages=[SimpleNamespace(gate_w=torch.eye(4), gate_bias=torch.zeros(4)) for _ in range(3)],
                           ring_kv=torch.ones(9, 3, 128, 4))
    m = SimpleNamespace(s=SimpleNamespace(window=128, n_stages=3, hc_mult=4, top_k=3), ranks=[rank, rank],
                        ALL=(0,1), eps=1e-20, insert_graphs={8: {}}, dev=torch.device('cpu'))
    m._main_x = lambda rk, taps: taps
    m.insert_taps_graphed = lambda *a: None
    m.insert_eager = lambda *a: None
    m.insert_mx_eager = lambda *a: None
    m.reset_slots = lambda *a: None
    ids, conf = torch.arange(5).view(1,5), torch.ones(1,5)*.5
    def draft(host, N, n_out, margin, trace):
        trace['head_hidden'] = torch.ones(5,4)
        trace['base_logits'] = torch.ones(1,5,10)
        for i in range(3):
            for name in ('attn_out','ffn_in','ffn_out'):
                trace[f'stage{i}.{name}'] = torch.ones(5,4)
            trace[f'stage{i}.stream'] = torch.ones(5,4,4)
            trace[f'stage{i}.experts'] = torch.tensor([[0,1,2]]*5)
            trace[f'stage{i}.route_w'] = torch.ones(5,3)*.5
        return ids, conf, None, ids.clone()
    m.draft = draft
    return m, ids.numpy(), conf.numpy()


def test_dump_uses_actual_insert_graph_mainx_and_live_window(tmp_path, monkeypatch):
    import worker.v1_dump as D
    monkeypatch.setattr(D.FU, 'rms_norm', lambda x,*a: x)
    m, ids, conf = model()
    sink = V1Dump(m, {'arm':'native-packed','drafter_identity':'ab'*32}, str(tmp_path/'dumps'))
    mx = torch.arange(32).view(8,4).float()
    m.insert_graphs[8]['dump_main_x'] = {0: mx}
    m.insert_taps_graphed(torch.tensor([0,0]),torch.tensor([0,1]),torch.zeros(2,4))
    sink.emit(torch.tensor([[0],[7],[1],[0]]),5,5,False,(ids,conf,None))
    record = torch.load(sink.files[0], weights_only=False)
    assert torch.equal(record['main_x_P'],mx[1])
    assert record['kv_window'].shape == (2,3,4)
    assert all(record[f's{i}']['attn_out'].shape == (5,4) for i in range(3))
    assert record['window'] == dict(n=2,taps_rows=2,engine_mx_rows=0,P_source='taps',first=0,last=1)
    assert record['meta']['trace'].startswith('eager replay checked')
    m.reset_slots([0])
    assert not sink.main_x


def test_dump_graph_eager_mismatch_refuses_before_writing(tmp_path):
    m, ids, conf = model()
    sink = V1Dump(m, {'drafter_identity':'a'*64}, str(tmp_path/'dumps'))
    with pytest.raises(RuntimeError,match='differs from the actual reply'):
        sink.emit(torch.tensor([[0],[7],[1],[0]]),5,5,False,(ids+1,conf,None))
    assert not list((tmp_path/'dumps').glob('*.pt'))


def test_chosen_rounds_and_private_output_guard():
    assert chosen_rounds('1,5,40') == frozenset((1,5,40))
    for value in ('', '0', '-2', '1,a'):
        with pytest.raises(ValueError):
            chosen_rounds(value)
    m, _, _ = model()
    with pytest.raises(ValueError, match='outside the repository'):
        V1Dump(m, {}, str(__import__('pathlib').Path(__file__).resolve().parent/'private'))


def test_all_classification_gates_are_required():
    r = {'classes':{},'samples':12,'median_excess_over_band_median':{'s0.attn_out':2},
         'kl_ref_arm_final_logits':{'median':.002}}
    assert verdict(r,.001,12)['pass']
    for key in ('state_provenance_defect','agree_but_state_flag','unexplained'):
        assert not verdict({**r,'classes':{key:1}},.001,12)['pass']
    assert not verdict(r,.0001,12)['pass']
    assert not verdict(r,.001,13)['pass']
    r['median_excess_over_band_median']['s0.attn_out']=3
    assert not verdict(r,.001,12)['pass']
