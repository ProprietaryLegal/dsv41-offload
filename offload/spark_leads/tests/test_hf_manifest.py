from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import struct
import types
import pytest

spec=importlib.util.spec_from_file_location('x1_hf',Path(__file__).resolve().parents[1]/'hf_manifest.py')
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Response:
    def __init__(self, status: int, content_range: str, payload: bytes):
        self.status=status; self.headers={'Content-Range':content_range}; self.payload=payload; self.reads=[]
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,n: int) -> bytes:
        self.reads.append(n); return self.payload[:n]


def test_200_range_response_closed_without_consuming_weights(monkeypatch) -> None:
    response=Response(200,'',b'weights-must-not-be-read')
    monkeypatch.setattr(m,'urlopen',lambda *a,**k:response)
    with pytest.raises(ValueError,match='body NOT consumed'): m.ranged('https://huggingface.co/test',0,7)
    assert response.reads==[]


def test_header_reads_only_prefix_and_json(monkeypatch) -> None:
    data=json.dumps({'mtp.0.weight':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]}}).encode()
    calls=[]
    def ranged(url: str,start: int,end: int):
        calls.append((start,end))
        return (struct.pack('<Q',len(data)) if start==0 else data),8+len(data)+8
    monkeypatch.setattr(m,'ranged',ranged)
    h,info=m.header('https://huggingface.co/pinned')
    assert list(h)==['mtp.0.weight']
    assert calls==[(0,7),(8,7+len(data))] and info['range_bytes']==len(data)+8


@pytest.mark.parametrize('status,content_range',[(206,'bytes 0-99/1000'),(206,'bytes 0-7/7'),(200,''),(206,'invalid')])
def test_wrong_ranges_refused(status: int,content_range: str) -> None:
    with pytest.raises(ValueError): m.validate_range(status,content_range,0,7)


def test_saved_public_metadata_proves_representation_without_tensor_reads() -> None:
    folder=Path(__file__).resolve().parents[1]/'metadata'
    packs=[json.loads((folder/f'sudoingx-{v}.json').read_text()) for v in ['27','28']]
    for pack in packs:
        assert pack['tensor_payload_bytes_consumed']==0 and not pack['header_failures']
        experts=pack['headers']['model-exl3-mtp-experts.safetensors']['tensors']
        trellis=[v for k,v in experts.items() if k.endswith('.trellis')]
        assert len(trellis)==3*128*3
        assert all(t['dtype']=='I16' and t['shape'][-1]==48 for t in trellis)
        assert pack['headers']['model-00044-of-00046.safetensors']['tensors']['mtp.0.main_proj.weight']['dtype']=='F8_E4M3'
    def identity(pack):
        return next(f['lfs']['sha256'] for f in pack['files'] if f['rfilename']=='model-exl3-mtp-experts.safetensors')
    assert identity(packs[0])==identity(packs[1])
    assert packs[1]['headers']['model-exl3-mtp-head.safetensors']['tensors']['mtp.head.trellis']['shape']==[320,8080,64]
    official=json.loads((folder/'official-1017.json').read_text())
    assert len(official['mtp_weight_map'])==2401
    assert len(official['headers'])==3
    expert=official['headers']['model-00044-of-00048.safetensors']['tensors']
    assert expert['mtp.0.ffn.experts.0.w1.weight']['shape']==[2304,2560]
    assert expert['mtp.0.ffn.experts.0.w1.scale']['dtype']=='F8_E8M0'
