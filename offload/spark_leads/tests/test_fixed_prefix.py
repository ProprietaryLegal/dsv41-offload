from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('x1_fixed',Path(__file__).resolve().parents[1]/'fixed_prefix.py')
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def capture(root: Path, label: str, probs: list[float], **row_changes: object) -> Path:
    folder=root/label; folder.mkdir()
    arr=folder/'row.npy'; np.save(arr,np.log(np.array(probs,dtype=np.float64)))
    row=dict(case_id='legal-hash',position=128,forced_next_id=1,prefix_token_sha256='a'*64,
             batch_token_sha256='b'*64,schedule_sha256='c'*64,array_sha256=hashlib.sha256(arr.read_bytes()).hexdigest(),
             shape={'n_requests':1,'query_lens':[129],'graph_mode':'NONE','graph_rows':129,
                    'spec_depths':[0],'n_out':5,'chunk_tokens':4096},logprobs_file='row.npy')
    row.update(row_changes)
    path=folder/'manifest.json'
    path.write_text(json.dumps(dict(schema='dsv41.fixed-prefix.v1',vocab_size=len(probs),vocab_order='token_id',
        target_identity='target',tokenizer_identity='tokenizer',numerical_config='fp8-baseline',
        run_id=label,build_id=label,rows=[row])))
    return path


def test_full_kl_known_and_identity() -> None:
    a=np.log(np.array([.25,.75])); b=np.log(np.array([.5,.5]))
    r=m.full_metrics(a,b)
    assert r['kl_a_b']==pytest.approx(.25*np.log(.5)+.75*np.log(1.5))
    assert r['total_variation']==pytest.approx(.25)
    assert m.full_metrics(a,a)['kl_a_b']==0


def test_zero_support_is_reported_not_nan() -> None:
    a=np.array([0.,-np.inf]); b=np.array([-np.inf,0.])
    r=m.full_metrics(a,b)
    assert r['kl_a_b']=='infinite' and r['kl_b_a']=='infinite' and r['support_mismatch']
    assert m.full_metrics(a,a)['kl_a_b']==0


@pytest.mark.parametrize('x',[[np.nan,0.],[np.inf,0.],[-2.,-3.]])
def test_truncated_or_invalid_distributions_refused(x: list[float]) -> None:
    with pytest.raises(ValueError): m.full_metrics(np.array(x),np.log(np.array([.5,.5])))


def test_aligned_capture_and_shape_refusal(tmp_path: Path) -> None:
    a=capture(tmp_path,'a',[.25,.75]); b=capture(tmp_path,'b',[.5,.5])
    assert len(m.compare(a,b)['rows'])==1
    obj=json.loads(b.read_text()); obj['rows'][0]['shape']['graph_rows']=256; b.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='shape'): m.compare(a,b)


def test_token_prefix_mismatch_and_array_tamper(tmp_path: Path) -> None:
    a=capture(tmp_path,'a',[.25,.75]); b=capture(tmp_path,'b',[.5,.5],prefix_token_sha256='f'*64)
    with pytest.raises(ValueError,match='prefix'): m.compare(a,b)
    obj=json.loads(b.read_text()); obj['rows'][0]['prefix_token_sha256']='a'*64; b.write_text(json.dumps(obj))
    np.save(b.parent/'row.npy',np.log(np.array([.3,.7])))
    with pytest.raises(ValueError,match='sha256'): m.compare(a,b)
