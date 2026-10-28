from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'offload/dspark/spark'))
from remote_dspark.state import WindowRange, SlotTable


def test_backtracking_cannot_recover_evicted_raw_rows() -> None:
    ring=WindowRange(128)
    ring.apply(872,128,1000)
    ring.apply(0,0,995)
    assert ring.rows_for_anchor(995)==123  # five older positions are NOT in rank-0 backup either
    assert ring.expected_rows(995)==128
    # A RESYNC of only the retained rows cannot invent the missing five.
    copy=WindowRange(128)
    copy.apply(ring.lo,ring.hi-ring.lo+1,995)
    assert copy.rows_for_anchor(995)==123
    # Fresh target replay of the needed suffix restores all 128.
    copy.apply(867,128,995)
    assert copy.rows_for_anchor(995)==128


def test_slot_reuse_and_explicit_reset_require_real_refill() -> None:
    table=SlotTable(1,128)
    table.begin_request(0,'a'); g=table[0].generation
    table[0].rng.apply(0,128,128)
    assert table.begin_request(0,'b') and table[0].generation != g
    assert table[0].rng.rows_for_anchor(128)==0
    table[0].rng.apply(500,128,628)
    table[0].rng.clear()
    assert table[0].rng.rows_for_anchor(628)==0


def test_three_context_layers_are_row_local_in_the_actual_worker_source() -> None:
    import ast
    module=ast.parse((ROOT/'offload/dspark/worker/model.py').read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='DSparkDrafter')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='context_kv')
    loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For))
    # Context KV is independently projected from mx at every stage, no propagation of draft context hidden states.
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='wkv']
    assert len(calls)==1 and isinstance(calls[0].args[0],ast.Name) and calls[0].args[0].id=='mx'
    assert not any(isinstance(n,(ast.Assign,ast.AugAssign)) and any(isinstance(t,ast.Name) and t.id=='mx'
                   for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) for n in ast.walk(loop))
