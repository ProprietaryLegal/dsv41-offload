from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / 'offload/spark_leads'
spec = importlib.util.spec_from_file_location('x1_lookahead', PKG / 'lookahead_bridge.py')
assert spec is not None and spec.loader is not None
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)


def test_remote_patch_bases_and_syntax(tmp_path: Path) -> None:
    pins = json.loads((PKG / 'patches/0078-bases.json').read_text())['bases']
    for name, sha in pins.items():
        if sha is None:
            continue
        src = (ROOT if name.startswith('offload/') else Path('/opt/dsv41/data/src-image')) / name
        assert hashlib.sha256(src.read_bytes()).hexdigest() == sha
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src,dst)
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-d',str(tmp_path),'-i',str(PKG/'patches/0078-remote.patch')],
                   check=True,capture_output=True)
    for name in pins:
        if name.endswith('.py'):
            compile((tmp_path/name).read_text(),name,'exec')
    cpp = (tmp_path/'offload/engram/spark/engram_remote.cpp').read_text()
    assert cpp.index('if (spin_) RemoteProxy::get().remove(this);') < cpp.index('stop_.store(true);')
    assert 'bounce_ = bounce_ || la_on_' in cpp
    assert 'if (!busy)' in cpp and 'std::min<int64_t>(timeout_ns_, 2000000LL)' in cpp


def tables() -> object:
    return la.HashTables(np.arange(32),[[11,13,17,19],[23,29,31,37]],
                         np.array([[[101,103],[107,109],[113,127]]]*2),
                         [[0,101,204,311,420,533]]*2,0,31)


def test_hash_matches_independent_scalar_reference() -> None:
    h=tables()
    tokens=np.array([3,4,31,6,7,8,9,10,11,12])
    for start,end in [(0,6),(1,9),(4,10)]:
        got=la.host_hashes(h,tokens[la.segment_start(h,start):end],start,end)
        for r,p in enumerate(range(start,end)):
            for layer in range(2):
                rolling=0; blocked=False
                for shift in range(4):
                    q=p-shift
                    blocked |= q < 0 or (q >= 0 and tokens[q]==31)
                    value=0 if blocked else int(tokens[q])
                    rolling ^= value*int(h.multipliers[layer,shift])
                    if shift:
                        for head in range(2):
                            col=(shift-1)*2+head
                            assert got[r,layer,col] == rolling%h.primes[layer,shift-1,head]+h.offsets[layer,col]


def test_snapshot_cancel_and_row_bound(monkeypatch) -> None:
    monkeypatch.setattr(la,'TOKENS',512)
    records=[]
    backend=types.SimpleNamespace(replace_lookahead=lambda ids,prio: records.append((ids,prio)))
    emb=types.SimpleNamespace(_remote_endpoint='rdma://direct',dp_size=1,head_start=0,part_n_hash_cols=6,
                              n_hash_cols=6,vocab_start_idx=0,part_num_embeddings=1000,_disk=backend)
    mods=[types.SimpleNamespace(_x1_host=tables()),types.SimpleNamespace(layer_hash_index=0,embed_tokens=emb)]
    model=types.SimpleNamespace(modules=lambda:iter(mods))
    states=types.SimpleNamespace(all_token_ids=types.SimpleNamespace(_uva_buf=types.SimpleNamespace(np=np.tile(np.arange(20),(1,1)))))
    batch=types.SimpleNamespace(num_reqs=1,is_prefilling_np=[True],idx_mapping_np=[0],
                                num_computed_prefill_tokens_np=[0],num_scheduled_tokens=[4],prefill_len_np=[20])
    la.after_forward(model,batch,states)
    assert records[-1][0] and len(records[-1][0])<=8192
    batch.is_prefilling_np=[False]
    la.after_forward(model,batch,states)
    assert records[-1]==([],0)


def test_native_cache_core(tmp_path: Path) -> None:
    binary=tmp_path/'row-test'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(PKG/'include'),
                    str(PKG/'tests/row_lookahead_test.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
