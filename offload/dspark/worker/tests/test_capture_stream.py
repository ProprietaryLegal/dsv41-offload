"""Isolated real-checkpoint regression: model.py before TP2 with a foreign capture/current stream.

Run under p2p/run_owned.sh solo or pair. A failing CUDA capture poisons only the child process.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.environ.get("DSPARK_ROOT", "/opt/dsv41/x86host-data/dspark")


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(ROOT), reason="GPU + checkpoint")
def test_one_card_before_tp2_foreign_capture_stream() -> None:
    child = r'''
import torch
from worker.checkpoint import Checkpoint
from worker.model import DSparkDrafter
from worker.tp2 import TPDrafter

d0 = torch.device("cuda", 0)
d1 = torch.device("cuda", min(1, torch.cuda.device_count() - 1))
ck = Checkpoint(ROOT)
# Preserve the failing construction order; TP2's first capture must follow one-card capture.
with torch.cuda.device(d1):
    one = DSparkDrafter(ck, d1, slots=1, markov_cache_rows=256)
tp = TPDrafter(ck, [d0, d1], slots=1, markov_cache_rows=256)
foreign = torch.cuda.Stream(d0)
torch.cuda.graphs.graph.default_capture_stream = foreign
with torch.cuda.stream(foreign):
    one.capture(batch_sizes=(1,), ks=(5,), n_outs=(5,), insert_rows=(8,),
                margins=True, pack_reply=True)
tp.capture(batch_sizes=(1,), ks=(5,), n_outs=(5,), insert_rows=(8,), margins=False)
inp = torch.tensor([[0], [123], [0], [0]], dtype=torch.int64).pin_memory()
eager = tp.draft(inp, 5)
want = [x.cpu().clone() for x in eager[:2]]
got = tp.draft_graphed(inp, 5)
assert all(torch.equal(a, b.cpu()) for a, b in zip(want, got[:2]))
# The one-card drafter launches on the current device, as in the one-card worker (one visible card); with two visible
# cards its calls must run under its own device or Triton refuses the cuda:1 pointers.
with torch.cuda.device(d1):
    args = (torch.tensor([0], device=d1), torch.tensor([123], device=d1),
            torch.tensor([0], device=d1), 5)
    eager_one = one.draft(*args)
    want_one = [x.cpu().clone() for x in eager_one[:2]]
    for margin, packed in ((False, False), (True, False), (False, True), (True, True)):
        result = one.draft_graphed(*args, margin=margin, packed_reply=packed)
        if packed:
            assert torch.equal(result[0][:, :5].cpu(), want_one[0].double())
            assert torch.equal(result[0][:, 5:10].cpu(), want_one[1].double())
        else:
            assert all(torch.equal(a, b.cpu()) for a, b in zip(want_one, result[:2]))
for d in {d0, d1}:
    torch.cuda.synchronize(d)
print("FOREIGN_CAPTURE_PASS", flush=True)
'''
    result = subprocess.run([sys.executable, "-u", "-c", f"ROOT = {ROOT!r}\n" + child],
                            capture_output=True, text=True, timeout=780, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FOREIGN_CAPTURE_PASS" in result.stdout
