"""One-second memory samples must be attributed to the actual phase boundaries."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from offload.dspark.native.tools.n3.collect import sampled_peaks


def test_sampled_peaks_use_load_and_serve_boundaries(tmp_path):
    start=dt.datetime(2026,10,23,tzinfo=dt.timezone.utc).timestamp()
    log=tmp_path/'phase.jsonl'
    log.write_text(''.join(json.dumps({'phase':phase,'time':start+t})+'\n'
                           for phase,t in [('load',0),('serve',2),('done',4)]))
    smi=tmp_path/'smi.csv'
    smi.write_text('\n'.join(f'2026/10/23 00:00:0{sec}.000, GPU-test, {memory} MiB, 123 W'
                              for sec,memory in [(0,100),(1,120),(2,90),(3,95),(4,130)]))
    assert sampled_peaks(smi,log)=={'load':{'GPU-test':120},'serve':{'GPU-test':95}}
    log.write_text(json.dumps({'phase':'load','time':start})+'\n')
    with pytest.raises(ValueError,match='incomplete'):
        sampled_peaks(smi,log)
