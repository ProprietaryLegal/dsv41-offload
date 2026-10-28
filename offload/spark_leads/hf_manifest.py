#!/usr/bin/env python3
"""Hugging Face metadata and safetensors HEADERS only. Refuses full-file HTTP responses to Range requests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Mapping, cast
from urllib.parse import quote
from urllib.request import Request, urlopen

MAX_METADATA = 64 * 1024 * 1024
MAX_HEADER = 8 * 1024 * 1024
AGENT = 'DSV41-X1-metadata-only/1.0'


def metadata(url: str) -> bytes:
    with urlopen(Request(url,headers={'User-Agent':AGENT}),timeout=60) as response:
        data=response.read(MAX_METADATA+1)
        if len(data)>MAX_METADATA:
            raise ValueError(f'metadata exceeds 64 MiB: {url}')
        return data


def validate_range(status: int, content_range: str | None, start: int, end: int) -> int:
    if status!=206:
        raise ValueError(f'Range refused: HTTP {status}; body NOT consumed')
    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range or '')
    if match is None or int(match[1])!=start or int(match[2])!=end or int(match[3])<=end:
        raise ValueError(f'wrong Content-Range {content_range!r}; body NOT consumed')
    return int(match[3])


def ranged(url: str, start: int, end: int) -> tuple[bytes,int]:
    if start<0 or end<start or end-start+1>MAX_HEADER:
        raise ValueError('invalid header-only range')
    with urlopen(Request(url,headers={'User-Agent':AGENT,'Range':f'bytes={start}-{end}',
                                     'Accept-Encoding':'identity'}),timeout=60) as response:
        total=validate_range(response.status,response.headers.get('Content-Range'),start,end)
        # Exact range only; no +1 read beyond the safetensors header into tensor data.
        data=response.read(end-start+1)
        if len(data)!=end-start+1:
            raise ValueError('short header Range response')
        return data,total


def header(url: str) -> tuple[dict[str,object],dict[str,object]]:
    prefix,total=ranged(url,0,7)
    length=struct.unpack('<Q',prefix)[0]
    if not 2<=length<=MAX_HEADER or length+8>total:
        raise ValueError(f'unsafe safetensors header length {length}')
    data,total2=ranged(url,8,7+length)
    if total!=total2:
        raise ValueError('file length changed between pinned header ranges')
    obj: object=json.loads(data)
    if not isinstance(obj,dict):
        raise ValueError('safetensors header is not object')
    return cast(dict[str,object],obj),{'range_bytes':length+8,'file_bytes':total,
                                      'header_sha256':hashlib.sha256(prefix+data).hexdigest()}


def object_json(data: bytes) -> dict[str,object]:
    obj: object=json.loads(data)
    if not isinstance(obj,dict):
        raise ValueError('expected metadata JSON object')
    return cast(dict[str,object],obj)


def inspect(repo: str, sha: str, *, with_headers: bool = True) -> dict[str,object]:
    if re.fullmatch('[0-9a-f]{40}',sha) is None:
        raise ValueError('full pinned repository sha required')
    api_url=f'https://huggingface.co/api/models/{repo}/revision/{sha}?blobs=true'
    raw=metadata(api_url); api=object_json(raw)
    if api.get('sha')!=sha:
        raise ValueError('HF revision response differs from pinned SHA')
    base=f'https://huggingface.co/{repo}/resolve/{sha}/'
    siblings=cast(list[dict[str,object]],api.get('siblings',[]))
    files={cast(str,f['rfilename']):f for f in siblings}
    index: dict[str,object]={}; config: dict[str,object]={}; plan: dict[str,object]={}
    metadata_files: dict[str,object]={}
    for name in ['README.md','config.json','exl3_moe_plan.json','model.safetensors.index.json','MANIFEST.sha256']:
        if name not in files:
            continue
        data=metadata(base+quote(name,safe='/'))
        metadata_files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if name=='config.json': config=object_json(data)
        elif name=='exl3_moe_plan.json': plan=object_json(data)
        elif name=='model.safetensors.index.json': index=object_json(data)
    weight_map=cast(dict[str,str],index.get('weight_map',{}))
    mtp={name:filename for name,filename in weight_map.items() if name.startswith('mtp.')}
    selected={f for f in files if f.startswith('model-exl3-mtp') and f.endswith('.safetensors')}
    selected.update(mtp.values())
    # GGUF: list metadata/provenance only; never request any GGUF payload or tensor bytes.
    headers: dict[str,object]={}; failures: dict[str,str]={}
    for name in sorted(selected):
        if not with_headers:
            continue
        try:
            hdr,info=header(base+quote(name,safe='/'))
            tensors={k:v for k,v in hdr.items() if k.startswith('mtp.') or (name.startswith('model-exl3-mtp') and k!='__metadata__')}
            headers[name]={'transport':info,'tensors':tensors}
        except (ValueError,OSError) as exc:
            failures[name]=str(exc)  # explicit artifact failure; caller returns nonzero
    return {'schema':'dsv41.hf-header-inventory.v1','repo':repo,'revision':sha,'api_url':api_url,
            'api_sha256':hashlib.sha256(raw).hexdigest(),'license':cast(dict[str,object],api.get('cardData',{})).get('license'),
            'files':siblings,'metadata_files':metadata_files,'config':config,'plan':plan,'mtp_weight_map':mtp,
            'headers':headers,'header_failures':failures,'tensor_payload_bytes_consumed':0}


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('repo'); ap.add_argument('sha'); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--no-headers',action='store_true')
    args=ap.parse_args()
    result=inspect(args.repo,args.sha,with_headers=not args.no_headers)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'repo':args.repo,'revision':args.sha,'header_files':len(cast(dict[str,object],result['headers'])),
                      'failures':result['header_failures'],'tensor_payload_bytes_consumed':0}))
    return 2 if result['header_failures'] else 0


if __name__=='__main__':
    raise SystemExit(main())
