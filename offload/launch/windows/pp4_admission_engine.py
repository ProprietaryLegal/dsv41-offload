"""Owned experiment seams; installed ONLY in separately prepared admission overlays.

No torch import in the CPU proxy. Every seam needs the explicit admission env,
run-bound expiring marker, and private writable experiment mount. Never deployed
by ordinary PP4 preparation. Evidence contains counters/hashes, never text/logits.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

_seen: set[tuple[int, str]] = set()


class Client(Protocol):
    session: int
    epoch: int
    def new_session(self, reason: str) -> None: ...
    def mark_failed(self, reason: str, *, backoff: bool = True) -> None: ...


class Speculator(Protocol):
    sampled_valid: torch.Tensor


class Runner(Protocol):
    speculator: Speculator


class Batch(Protocol):
    idx_mapping: torch.Tensor


class Sample(Protocol):
    sampled_token_ids: torch.Tensor
    num_sampled: torch.Tensor


def marker() -> dict[str, object] | None:
    run = os.environ.get('VLLM_PP4_ADMISSION_RUN')
    if not run:
        return None
    root = Path(os.environ['VLLM_PP4_ROUND_DIR'])
    path = root/'FAULT.json'
    if not path.exists():
        return None
    data = cast(dict[str, object], json.loads(path.read_text()))
    if data.get('run') != run or not isinstance(data.get('id'), str) or float(str(data['expires'])) <= time.time():
        raise ValueError('PP4 admission marker owner/lease mismatch')
    if data.get('mode') not in ('q-rank0', 'q-rank1', 'reset', 'reconnect', 'timeout', 'late-reply'):
        raise ValueError('PP4 admission unknown fault')
    return data


def event(kind: str, **values: object) -> None:
    run = os.environ.get('VLLM_PP4_ADMISSION_RUN')
    if not run:
        raise ValueError('admission event outside owned experiment')
    root = Path(os.environ['VLLM_PP4_ROUND_DIR'])
    with (root/f'admission-{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps({'kind': kind, 'run': run, 'time': time.time(), **values}, allow_nan=False)+'\n')


def client_before(value: object) -> None:
    data = marker()
    if data is None or data['mode'] not in ('reset', 'reconnect'):
        return
    key = id(value), str(data['id'])
    if key in _seen:
        return
    _seen.add(key)
    client = cast(Client, value)
    before = client.session
    if data['mode'] == 'reset':
        client.new_session('PP4 owned admission reset')
    else:
        client.mark_failed('PP4 owned admission reconnect', backoff=False)
    event('client_fault', fault=data['mode'], id=data['id'], session_before=before,
          session_after=client.session, epoch=client.epoch)


def client_after(value: object, reply: bytearray) -> None:
    data = marker()
    client = cast(Client, value)
    if data is None:
        return
    key = id(value), str(data['id'])
    if data['mode'] in ('timeout', 'late-reply') and key not in _seen:
        _seen.add(key)
        event('client_fault', fault=data['mode'], id=data['id'], session_before=client.session,
              session_after=client.session, epoch=client.epoch)
        if data['mode'] == 'timeout':
            # Worker has applied a real STEP; its reply is deliberately lost.
            # Raise INSIDE the production client's transport try/except. This
            # exercises unknown_apply, slot resync and reconnect, with no unit kill.
            module = importlib.import_module(type(value).__module__.rsplit('.', 1)[0]+'.transport')
            error = cast(Callable[[str], Exception], module.TransportTimeout)
            raise error('PP4 owned lost reply after worker application')
        # Real previous-sequence reply on this session; actual client echo check
        # must reject it, close transport and resync. No fake successful outcome.
        seq = struct.unpack_from('<Q', reply, 8)[0]
        struct.pack_into('<Q', reply, 8, seq-1)
    else:
        event('client_recovered', id=data['id'], epoch=client.epoch, session=client.session)


def invalidate(runner: object, batch: object) -> None:
    data = marker()
    if data is None or data['mode'] not in ('q-rank0', 'q-rank1'):
        return
    import torch
    rank = torch.distributed.get_rank()
    valid = cast(Runner, runner).speculator.sampled_valid
    mapping = cast(Batch, batch).idx_mapping.long()
    flag = torch.tensor([int(rank == int(str(data['mode'])[-1]))], device=valid.device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    if int(flag.item()) != 1:
        raise ValueError('PP4 q failure not agreed by both ranks')
    valid.index_fill_(0, mapping, False)
    event('q_invalidated', id=data['id'], rank=rank, origin=int(str(data['mode'])[-1]),
          slots=int(mapping.numel()), all_invalid=bool((~valid[mapping]).all().item()))


def verify(fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Same real processed target logits/positions/seeds, independently target-only.

    Exact equality of every emitted token/length (not a histogram tolerance).
    This proves coupled target-only behavior on each observed engine invocation;
    it does not claim exhaustive equality for unobserved logits.
    """
    def call(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        data = marker()
        actual = fn(*args, **kwargs)
        if data is None or data['mode'] not in ('q-rank0', 'q-rank1'):
            return actual
        import torch
        bound = __import__('inspect').signature(fn).bind(*args, **kwargs)
        drafts = cast(torch.Tensor, bound.arguments['draft_sampled'])
        local = cast(torch.Tensor, bound.arguments['expanded_local_pos'])
        if not bool((local > 0).any().item()):
            return actual
        if not bool((drafts[local > 0] == -1).all().item()):
            raise ValueError('PP4 failed q reached verifier as a proposal')
        bound.arguments['draft_logits'] = None
        previous = os.environ['VLLM_DSPARK_SAMPLED_TOP32']
        try:
            os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '0'
            expected = fn(*bound.args, **bound.kwargs)
        finally:
            os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = previous
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        if not bool((actual[1] == 1).all().item()):
            raise ValueError('failed q accepted drafts')
        torch.testing.assert_close(actual[0][:, 0], expected[0][:, 0], rtol=0, atol=0)
        digest = hashlib.sha256(actual[0][:, 0].cpu().numpy().tobytes()).hexdigest()
        event('target_only_exact', id=data['id'], rank=torch.distributed.get_rank(),
              requests=int(actual[1].numel()), emitted_sha256=digest, lengths_one=True, exact=True)
        return actual
    return call


def observe_sample(output: object) -> None:
    """Prove model EOS from actual sampler IDs even when API stop_reason is null."""
    run = os.environ.get('VLLM_PP4_ADMISSION_RUN')
    if not run:
        return
    path = Path(os.environ['VLLM_PP4_ROUND_DIR'])/'EOS.json'
    if not path.exists():
        return
    data = cast(dict[str, object], json.loads(path.read_text()))
    if data.get('run') != run or float(str(data['expires'])) <= time.time():
        raise ValueError('EOS observation owner/lease mismatch')
    import torch
    sampled = cast(Sample, output).sampled_token_ids
    lengths = cast(Sample, output).num_sampled
    valid = torch.arange(sampled.shape[1], device=sampled.device)[None, :] < lengths[:, None]
    eos = int(str(data['eos']))
    seen = bool(((sampled == eos) & valid).any().item())
    if seen:
        event('model_eos_sampled', rank=torch.distributed.get_rank(), eos=eos, observed=True)
