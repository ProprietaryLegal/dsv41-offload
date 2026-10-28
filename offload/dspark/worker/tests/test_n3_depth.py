"""Exercise the protocol runner and coordinator-only refusal without CUDA."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from worker.server import DraftCore, FakeBackend, expected_hello
from offload.dspark.native.tools.n3.soak import rounds


def test_depth_export_rejects_mixed_or_sparse_costs_and_subtracts_overlap():
    from offload.dspark.native.tools.n3.depth_config import export
    import pytest
    rows=[]
    for d in range(6):
        for g in range(5):
            rows.append(dict(schema='dsv41.depth-round.v1',build_id='native:identity:revision',mode='remote_async',
                n_requests=1,depth_vector=[d],context_tokens_max=32768,cache='steady',group_id=f'g{g}',
                partition='fit' if g<3 else 'heldout',verify_ms=30+d,prep_ms=4,overlap_ms=3,queue_ms=0,round_ms=31+d))
    kw=dict(build='native:identity:revision',mode='remote_async',requests=1,context=32768,overlap_audited=True)
    result=export(rows,**kw)
    assert result['env_lines']==['VLLM_DSPARK_REMOTE_DRAFT_MS=1','VLLM_DSPARK_REMOTE_ROUND_MS=30,31,32,33,34,35']
    assert result['heldout_round_relative_error_abs_max']==0
    with pytest.raises(ValueError,match='overlap must be audited'):
        export(rows,**{**kw,'overlap_audited':False})
    with pytest.raises(ValueError,match='depth 5'):
        export(rows[:-5],**kw)
    with pytest.raises(ValueError,match='mixed schema'):
        export([dict(rows[0],build_id='K2')]+rows[1:],**kw)


    # V1 joins simultaneous task-family IDs with '+'. Composite string IDs
    # must not hide a family that appears in both fit and held-out batches.
    with pytest.raises(ValueError,match='depth 0'):
        export([dict(rows[0],group_id='g0+g3')]+rows[1:],**kw)
