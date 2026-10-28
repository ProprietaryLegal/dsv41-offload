from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from offload.dspark.probabilistic.pp3.estimate import estimate
from offload.dspark.probabilistic.pp3.prepare_verifier import prepare
from offload.dspark.probabilistic.reference import Categorical, acceptance, residual

ROOT=Path(__file__).parents[1]
REPO=ROOT.parents[3]


def test_actual_gpu_q_has_exact_pp1_target_law() -> None:
    evidence=json.loads((ROOT/'oracle-bridge.json').read_text())
    p=Categorical(64,[40,41,42,43],[1,2,2,3])
    for row in evidence['cases']:
        assert sum(row['counts']) == 2**24
        q=Categorical(64,row['support_ids'],row['counts'])
        accepted=[q.prob(i)*acceptance(p,q,i) if q.prob(i) else Fraction() for i in range(64)]
        rejected=1-sum(accepted)
        correction=residual(p,q) if rejected else None
        output=[accepted[i]+(rejected*correction.prob(i) if correction is not None else 0) for i in range(64)]
        assert output == [p.prob(i) for i in range(64)]
        assert row['exact_law']


def test_measurements_bind_final_producer_and_oracle_sources() -> None:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    producer=json.loads((ROOT/'producer-final.json').read_text())
    qualifier=json.loads((ROOT/'gpu-qualification.json').read_text())
    bridge=json.loads((ROOT/'oracle-bridge.json').read_text())
    assert producer['complete'] and qualifier['complete']
    assert producer['producer_sha256'] == qualifier['producer_sha256'] == digest(REPO/'offload/dspark/worker/sampled.py')
    assert producer['source_sha256'] == digest(REPO/'offload/dspark/worker/tools/pp3_probe.py')
    assert qualifier['source_sha256'] == digest(ROOT/'gpu_qualify.py')
    assert bridge['source_sha256'] == digest(ROOT/'oracle_bridge.py')
    assert bridge['offload.dspark.worker.sampled_sha256'] == producer['producer_sha256']
    assert bridge['offload.dspark.probabilistic.reference_sha256'] == digest(ROOT.parent/'reference.py')
    expected=json.loads((ROOT/'cost-estimate.json').read_text())
    assert estimate(producer,qualifier,json.loads((ROOT/'pp2-frozen.json').read_text())) == expected


def test_overlay_is_pinned_complete_and_cannot_overwrite(tmp_path: Path) -> None:
    source=Path('/opt/dsv41/data/src-image')
    out=tmp_path/'prepared'
    receipt=prepare(source,out)
    assert receipt == json.loads((ROOT/'verifier-receipt.json').read_text())
    with pytest.raises(ValueError,match='must not exist'):
        prepare(source,out)
    bad=tmp_path/'bad'
    bad_source=tmp_path/'source'
    path=bad_source/'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py'
    path.parent.mkdir(parents=True)
    path.write_text('wrong revision')
    with pytest.raises(ValueError,match='differs from pinned'):
        prepare(bad_source,bad)
    assert not bad.exists()
