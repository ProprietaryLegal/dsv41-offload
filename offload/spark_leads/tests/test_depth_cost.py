from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('x1_depth',Path(__file__).resolve().parents[1]/'depth_cost.py')
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)


def sample(i: int, part: str='fit', **kwargs: object) -> dict[str,object]:
    out: dict[str,object]=dict(schema='dsv41.depth-round.v1',sample_id=f's{i}',group_id=f'g{i}',partition=part,
        build_id='build',n_requests=1,depth_vector=[4],planned_verify_rows=5,graph_rows=8,
        context_tokens_max=32000,mode='remote_async',cache='steady',verify_ms=30,prep_ms=4,
        overlap_ms=1,queue_ms=0,round_ms=36,committed_tokens=2)
    out.update(kwargs); return out


def test_fit_uses_disjoint_groups_and_direct_rounds() -> None:
    data=[m.Round.parse(sample(i,'fit' if i<3 else 'heldout',round_ms=36+i)) for i in range(5)]
    result=m.fit(data)
    assert result['qualified_coverage']
    entry=result['cells'][0]
    assert entry['round_mean_ms']==37 and entry['verify_mean_ms']==30 and entry['prep_mean_ms']==4
    assert result['heldout_errors'][0]['error_ms']==2


def test_leakage_and_duplicates_are_refused() -> None:
    a=m.Round.parse(sample(0)); b=m.Round.parse(sample(1,'heldout',group_id='g0'))
    with pytest.raises(ValueError,match='leakage'): m.fit([a,b])
    with pytest.raises(ValueError,match='duplicate'): m.fit([a,a])


@pytest.mark.parametrize('change',[{'round_ms':float('nan')},{'overlap_ms':5},{'depth_vector':[6]},
                                   {'planned_verify_rows':4},{'mode':'unknown'},{'committed_tokens':7}])
def test_bad_samples_refused(change: dict[str,object]) -> None:
    with pytest.raises(ValueError): m.Round.parse(sample(0,**change))


def test_unsupported_coverage_and_build_drift() -> None:
    result=m.fit([m.Round.parse(sample(0))])
    assert not result['qualified_coverage'] and result['unqualified_cells']
    with pytest.raises(ValueError,match='mixed builds'):
        m.fit([m.Round.parse(sample(0)),m.Round.parse(sample(1,build_id='changed'))])
    assert not m.fit([m.Round.parse(sample(0,'heldout'))])['cells']


def test_score_does_not_add_prep_twice() -> None:
    assert m.score(2,40,44,.5)==2/42
    with pytest.raises(ValueError): m.score(2,0,0,0)
