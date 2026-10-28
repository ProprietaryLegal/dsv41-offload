"""Actual physical-peer selector qualification; invoked only by fenced pair driver.

Both ranks execute real peer kernels. Same frozen CPU fixture is also computed
with two logical shards on device0, producing a fresh solo receipt of IDs/q.
No prompt, private tensor or selected token ID is written, only aggregate hashes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .pair_diagnostics import phase, progress, start_watchdog, stop_watchdog

if __name__ == '__main__':
    start_watchdog()  # imports, extension load and CUDA initialization can stall

with phase('imports torch / selector / pair contract'):
    import torch

    from ...worker.p2p import ext, wait_idle
    from ...worker.sampled_fused import FusedContext
    from .pair_contract import CASES, checked_output, guard


@dataclass
class Markov:
    w1: torch.Tensor
    w2: torch.Tensor
    slot_of_token: torch.Tensor
    cache: torch.Tensor


@dataclass
class Rank:
    r: int
    dev: torch.device
    v0: int
    v1: int
    markov: Markov


@dataclass
class Model:
    ALL: tuple[int, int]
    devs: list[torch.device]
    ranks: list[Rank]
    emulated: bool


def fixture(emulated: bool) -> Model:
    generator = torch.Generator().manual_seed(4104)
    vocab, width = 129280, 256
    w1 = (torch.randn(vocab, width, generator=generator)/8).bfloat16()
    slots = torch.full((vocab,), -1, dtype=torch.int64); slots[:4] = torch.arange(4)
    ranks = []
    for r, (lo, hi) in enumerate(((0, 64768), (64768, vocab))):
        dev = torch.device('cuda:0' if emulated else f'cuda:{r}')
        with phase(f'fixture emulated={emulated} rank={r} CPU tensors / device copies'):
            w2 = (torch.randn(hi-lo, width, generator=generator)/8).bfloat16()
            cache = (torch.randn(4, hi-lo, generator=generator)/4).bfloat16()
            ranks.append(Rank(r, dev, lo, hi, Markov(w1.to(dev), w2.to(dev), slots.to(dev), cache.to(dev))))
    return Model((0, 1), [r.dev for r in ranks], ranks, emulated)


def tensors(model: Model, batch: int, depth: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    gen = torch.Generator().manual_seed(4104+batch*10+depth)
    logits = {r: torch.randn(batch, depth, rk.v1-rk.v0, generator=gen).bfloat16().to(rk.dev)
              for r, rk in enumerate(model.ranks)}
    anchors = {r: torch.tensor([0 if j % 2 else 7 for j in range(batch)], device=model.devs[r]) for r in model.ALL}
    outputs = {r: torch.zeros(batch, depth, dtype=torch.int64, device=model.devs[r]) for r in model.ALL}
    return logits, anchors, outputs


def run(ctx: FusedContext, logits: dict[int, torch.Tensor], anchors: dict[int, torch.Tensor],
        outputs: dict[int, torch.Tensor], depth: int, ranks: tuple[int, ...]) -> None:
    for pos in range(depth):
        with phase(f'select emulated={ctx.model.emulated} ranks={ranks} position={pos}'):
            ctx.select(ranks, logits, anchors if pos == 0 else {r: outputs[r][:, pos-1] for r in (0, 1)}, outputs, pos)


def digest(parts: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture_pair(ctx: FusedContext, logits: dict[int, torch.Tensor], anchors: dict[int, torch.Tensor],
                 outputs: dict[int, torch.Tensor], depth: int) -> dict[int, torch.cuda.CUDAGraph]:
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    # torch.cuda.graph's class-level default stream can belong to rank0.
    # Capture each peer on a stream explicitly allocated on its own device.
    # PyTorch's Stream constructor has an untyped extension boundary.
    stream_factory = cast(Callable[..., torch.cuda.Stream], torch.cuda.Stream)
    capture_streams = {}
    for rank in (0, 1):
        with phase(f'capture stream create rank={rank}'):
            capture_streams[rank] = stream_factory(device=ctx.model.devs[rank])
    for rank in (0, 1):
        with phase(f'capture rank={rank}'), torch.cuda.device(ctx.model.devs[rank]):
            graph = torch.cuda.CUDAGraph()
            with ctx.capturing((rank,)), torch.cuda.graph(graph, stream=capture_streams[rank]):
                run(ctx, logits, anchors, outputs, depth, (rank,))
            graphs[rank] = graph
    return graphs


def qualify() -> list[dict[str, object]]:
    with phase('fixture physical'):
        peer = fixture(False)
    with phase('fixture solo reference'):
        solo = fixture(True)
    rows: list[dict[str, object]] = []
    for batch, depth, mode in CASES:
        case = f'case B={batch} depth={depth} mode={mode}'
        progress(f'BEGIN {case}')
        with phase('allocate physical / solo contexts and tensors'):
            real, reference = FusedContext(peer, batch, depth), FusedContext(solo, batch, depth)
            logits, anchors, outputs = tensors(peer, batch, depth)
            ref_logits, ref_anchors, ref_outputs = tensors(solo, batch, depth)
        mask = torch.tensor([mode == 'sampled' or mode == 'mixed-even' and j % 2 == 0
                             or mode == 'mixed-odd' and j % 2 == 1 for j in range(batch)], dtype=torch.bool)
        keys = torch.tensor([[4104+j, 1, 32768] for j in range(batch)], dtype=torch.int64)
        with phase('initial keys / mask update'):
            real.update(keys, mask); reference.update(keys, mask)
        for warmup in range(2):
            with phase(f'eager warmup={warmup} ranks=(0, 1)'):
                run(real, logits, anchors, outputs, depth, (0, 1))
        with phase('wait_idle physical after eager warmup'):
            wait_idle(peer.devs, 5)
        graphs = capture_pair(real, logits, anchors, outputs, depth)
        hashes: list[str] = []
        for replay in range(10):
            progress(f'BEGIN replay={replay} {case}')
            keys[:, 1].fill_(replay+7)
            with phase('replay keys / mask update'):
                real.update(keys, mask); reference.update(keys, mask)
            # Values, previous-token anchors and RNG counters all change at
            # fixed addresses; stale-output replay cannot pass this comparison.
            with phase('replay change logits / anchors'):
                for rank in (0, 1):
                    with torch.cuda.device(peer.devs[rank]):
                        logits[rank].add_(.125)
                        anchors[rank].fill_(replay % 4 if replay % 2 else 7)
                    ref_logits[rank].add_(.125); ref_anchors[rank].fill_(replay % 4 if replay % 2 else 7)
            for rank in (1, 0):
                with phase(f'graph replay rank={rank}'), torch.cuda.device(peer.devs[rank]):
                    graphs[rank].replay()
            with phase('wait_idle physical after replay'):
                wait_idle(peer.devs, 5)
            with phase('solo reference run / wait_idle'):
                run(reference, ref_logits, ref_anchors, ref_outputs, depth, (0, 1))
                wait_idle(solo.devs, 5)
            with phase('physical / solo result flags'):
                real.result(); reference.result()
            with phase('exact q / IDs comparison and digest'):
                for rank in (0, 1):
                    for value, expected in ((outputs[rank], ref_outputs[0]),
                                            (real.state[rank]['ids'], reference.state[0]['ids']),
                                            (real.state[rank]['q'], reference.state[0]['q'])):
                        torch.testing.assert_close(value.cpu(), expected.cpu(), rtol=0, atol=0)
                hashes.append(digest([ref_outputs[0], reference.state[0]['ids'], reference.state[0]['q']]))
            progress(f'END replay={replay} {case}')
        rows.append({'batch': batch, 'depth': depth, 'mode': mode, 'rank0_exact': True, 'rank1_exact': True,
                     'changed_input_graph_replays': 10, 'solo_sha256': hashlib.sha256(''.join(hashes).encode()).hexdigest()})
        with phase('case graph / context teardown and gc'):
            del graphs, real, reference
            gc.collect()
        progress(f'END {case}')
    return rows


def timeout_probe(rank: int) -> dict[str, object]:
    with phase(f'timeout fixture / state missing_rank={1-rank}'):
        model = fixture(False)
        ctx = FusedContext(model, 1, 4)
        logits, anchors, outputs = tensors(model, 1, 4)
        ctx.update(torch.tensor([[4104, 1, 32768]]), torch.tensor([True]))
    with phase('timeout probe prime both ranks / wait_idle / result'):
        ctx.select((0, 1), logits, anchors, outputs, 0)  # prime both specializations
        wait_idle(model.devs, 5); ctx.result()
    with phase(f'timeout deliberate absent peer launch active_rank={rank} / wait_idle'):
        ctx.select((rank,), logits, anchors, outputs, 0)
        wait_idle(model.devs, 5)
    try:
        ctx.result()
    except RuntimeError:
        pass
    else:
        raise AssertionError('missing physical peer was accepted')
    with phase('timeout sticky retry both ranks / wait_idle'):
        ctx.select((0, 1), logits, anchors, outputs, 0)
        wait_idle(model.devs, 5)
    try:
        ctx.result()
    except RuntimeError:
        pass
    else:
        raise AssertionError('sticky peer error disappeared on retry')
    if not any(bool(s['sticky'].any().item()) for s in ctx.state.values()):
        raise AssertionError('timeout did not leave sticky state')
    # Never reset the poison or reuse this worker. Driver waits for process
    # exit before any other GPU subprocess or restoration.
    return {'missing_rank': 1-rank, 'timeout_refused': True, 'sticky_retry_refused': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('correctness', 'timeout0', 'timeout1'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with phase(f'output / coordinator UUID lock guard mode={args.mode}'):
        checked_output(args.out); guard()
    with phase('torch thread limits'):
        torch.set_num_threads(8); torch.set_num_interop_threads(1)
    with phase('P2P extension load'):
        extension = ext()
    for rank in (0, 1):
        with phase(f'P2P enable rank={rank} peer={1-rank}'):
            extension.enable_peer(rank, 1-rank)
    with phase(f'probe mode={args.mode}'):
        result = {'cases': qualify()} if args.mode == 'correctness' else timeout_probe(int(args.mode[-1]))
    with phase('write complete result'):
        args.out.write_text(json.dumps(result, allow_nan=False)+'\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        stop_watchdog()
