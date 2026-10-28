"""Execute ported Python dispatch and Triton kernel source using CPU memory; no CUDA calls."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
import types
from typing import Any

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]/'overlay'


def extracted(path: str, names: set[str], scope: dict[str,Any]) -> dict[str,Any]:
    parsed = ast.parse((TREE/path).read_text())
    nodes = [node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        if isinstance(node,ast.FunctionDef):
            node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),path,'exec'),scope)
    return scope


def test_smoe_band_and_exact_row_destination() -> None:
    scope = extracted('vllm_exl3_moe/forward.py',{'split_rows','smoe_split_ok','smoe_split'},
                      {'torch':torch,'_SPLIT_MAX':63,'_TF_PROMPT':False,'_SPLIT_LOGGED':[False],'logging':logging})
    ranges, eligible, split = (scope[k] for k in ('split_rows','smoe_split_ok','smoe_split'))
    coop = types.SimpleNamespace(rows_max=32)
    calls: list[tuple[torch.Tensor,torch.Tensor]] = []
    def into(out: torch.Tensor,x: torch.Tensor,*args: object) -> None:
        out.copy_(x)
        calls.append((out,x))
    smoe = types.SimpleNamespace(max_rows=lambda:42,into=into)
    for m in (1,31,32,33,42,43,63,64,129):
        x = torch.arange(m*8).reshape(m,8).to(torch.bfloat16)
        assert eligible(m,x,'coop',coop,smoe) == (32 < m <= 63)
        assert not eligible(m,x,'fused',coop,smoe)
        assert not eligible(m,x.float(),'coop',coop,smoe)
        assert not eligible(m,x,'coop',coop,None)
        for cap in (1,36,42):
            spans = ranges(m,cap)
            assert [r for a,b in spans for r in range(a,b)] == list(range(m))
            assert all(0 < b-a <= cap for a,b in spans)
        if 32 < m <= 63:
            calls.clear()
            out = split(smoe,x,torch.zeros(m,6),torch.ones(m,6),torch.arange(1),0.0)
            assert torch.equal(out,x)
            assert sum(a.shape[0] for a,_ in calls) == m
            assert all(a.untyped_storage().data_ptr()==out.untyped_storage().data_ptr() for a,_ in calls)
    scope['_SPLIT_MAX'] = 0
    assert not eligible(43,torch.zeros(43,8,dtype=torch.bfloat16),'coop',coop,smoe)
    for args in ((0,42),(10,0)):
        with pytest.raises(ValueError):
            ranges(*args)


class Pointer:
    def __init__(self,values: np.ndarray,offset: Any = 0):
        self.values = values.reshape(-1)
        self.offset = offset
    def __add__(self,other: Any) -> Pointer:
        return Pointer(self.values,self.offset+other)
    __radd__ = __add__


class TL:
    int8, int32, int64 = np.int8,np.int32,np.int64
    constexpr = object
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)
    arange = staticmethod(np.arange)
    def __init__(self) -> None:
        self.programs = (0,0)
        self.memory: dict[int,Pointer] = {}
    def program_id(self,axis: int) -> int:
        return self.programs[axis]
    def pointer_type(self,dtype: Any) -> str:
        return 'pointer'
    def cast(self,value: Any,dtype: Any) -> Any:
        if dtype == 'pointer':
            return self.memory[int(value)]
        return np.asarray(value).astype(dtype)
    def load(self,p: Pointer,mask: Any = True,other: Any = 0) -> Any:
        indices = np.asarray(p.offset)
        active = np.broadcast_to(mask,indices.shape)
        # Refuse even partially active out-of-bounds accesses, like an illegal kernel load.
        assert np.all((indices[active]>=0)&(indices[active]<len(p.values)))
        return np.where(active,p.values[np.where(active,indices,0)],other)
    def store(self,p: Pointer,value: Any,mask: Any = True) -> None:
        indices, values, active = np.broadcast_arrays(p.offset,value,mask)
        assert np.all((indices[active]>=0)&(indices[active]<len(p.values)))
        p.values[indices[active]] = values[active]


class Kernel:
    def __init__(self,fn: Any,tl: TL):
        self.fn,self.tl = fn,tl
    def __getitem__(self,grid: tuple[int,int]) -> Any:
        def launch(*args: Any,**kwargs: Any) -> None:
            bound = [Pointer(t.numpy()) if isinstance(t,torch.Tensor) else t for t in args]
            for group in range(grid[1]):
                for pid in range(grid[0]):
                    self.tl.programs = pid,group
                    self.fn(*bound,**kwargs)
        return launch


def swa_scope() -> tuple[dict[str,Any],TL]:
    tl = TL()
    names = {'_swa_decode_groups_kernel','_same_view','swa_plan_entries','SWADecodePlan','ReplayStartMirror'}
    scope = extracted('vllm/models/deepseek_v41/common/ops/swa_fused.py',names,
        {'torch':torch,'np':np,'Any':Any,'tl':tl,'triton':types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),
         'logger':logging.getLogger('up1-test'),'_BLOCK':1024,'_COLS':8,
         '_SLOT':0,'_VALID':1,'_TABLE':2,'_INDICES':3,'_LENS':4,'_TABLE_STRIDE':5,'_BLOCK_SIZE':6})
    scope['_swa_decode_groups_kernel'] = Kernel(scope['_swa_decode_groups_kernel'],tl)
    return scope,tl


def test_swa_source_kernel_against_independent_slot_reference() -> None:
    scope,tl = swa_scope()
    rng = np.random.default_rng(670075)
    for case in range(24):
        nreq = case%4+1
        q = np.full(nreq,case%6+1,dtype=np.int32)
        qsl = torch.from_numpy(np.r_[np.int32(0),q.cumsum(dtype=np.int32)])
        n = int(q.sum())
        seq = torch.from_numpy(rng.integers(128,327680,nreq,dtype=np.int32))
        token_req = torch.repeat_interleave(torch.arange(nreq,dtype=torch.int32),torch.from_numpy(q))
        replay = torch.clamp(seq-64,min=0) if case%2 else torch.zeros(nreq,dtype=torch.int32)
        entries = []
        for group in range(12):  # synthetic group count, deliberately different from donor local-drafter15
            table = torch.from_numpy(rng.integers(1,10000,(nreq,5122),dtype=np.int32))
            slot = torch.from_numpy(rng.integers(-1,2,32,dtype=np.int64))
            builder = types.SimpleNamespace(window_size=128,block_size=64,
                decode_swa_indices=torch.full((32,192),-99,dtype=torch.int32),
                decode_swa_lens=torch.full((48,),-99,dtype=torch.int32),
                is_valid_token=torch.zeros(32,dtype=torch.bool),no_replay_start=torch.zeros(nreq,dtype=torch.int32))
            for tensor in (slot,table,builder.decode_swa_indices,builder.decode_swa_lens,builder.is_valid_token):
                tl.memory[tensor.data_ptr()] = Pointer(tensor.numpy())
            entries.append((builder,table,slot))
        plan = scope['SWADecodePlan']()
        plan.arm(entries)
        for builder,table,slot in entries:
            cm = types.SimpleNamespace(block_table_tensor=table,slot_mapping=slot,causal=True,
                                       query_start_loc=qsl,seq_lens=seq)
            assert plan.claim(builder,cm,n,0,token_req,replay)
            assert torch.equal(builder.is_valid_token,slot>=0)
            assert torch.equal(builder.decode_swa_lens[n:],torch.zeros(48-n,dtype=torch.int32))
            for i in range(n):
                if slot[i] < 0:
                    expected = [-1]*192
                    length = 0
                else:
                    req = int(token_req[i]);position = int(seq[req]-q[req]+i-qsl[req])
                    start = max(position-127,0,int(replay[req]));length=position+1-start
                    expected = [int(table[req,p//64])*64+p%64 for p in range(start,position+1)]
                    expected += [-1]*(192-len(expected))
                assert int(builder.decode_swa_lens[i]) == length
                assert builder.decode_swa_indices[i].tolist() == expected
        assert plan.launches == 1
        plan.disarm()
        assert not plan.claim(entries[0][0],cm,n,0,token_req,replay)


def test_swa_fallbacks_and_replay_mirror() -> None:
    scope,_ = swa_scope()
    mirror = scope['ReplayStartMirror'](4)
    assert mirror.holds(np.array([0,0],dtype=np.int32))
    assert not mirror.holds(np.array([4,0],dtype=np.int32))
    mirror.wrote(np.array([4,8],dtype=np.int32))
    assert mirror.holds(np.array([4],dtype=np.int32))
    assert not mirror.holds(np.array([4,0],dtype=np.int32))
    plan = scope['SWADecodePlan']()
    plan.arm([])
    assert not plan.claim(None,None,0,0,None,None)
    # Early mixed/noncausal/empty checks must not dereference any group buffers.
    for n,prefill,causal in ((1,1,True),(0,0,True),(1,0,False)):
        assert not plan._eligible(types.SimpleNamespace(causal=causal),n,prefill,None,None)
