"""0067 installed stock/fused metadata checks with the loaded target's groups.

Only metadata is allocated: no second model, KV cache or draft model. All mutable
builder buffers are cloned; physical IDs are used as metadata, never dereferenced.
Run on an idle worker after startup warmup, before accepting timed work.
"""
from __future__ import annotations

import copy
import importlib
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Iterator, cast

import numpy as np
import torch

from offload.prefill_maxlen.evidence import Json
from offload.up3.engine_common import Cases, attr, invoke, passed, replaced, require, runner
from offload.up3.engine_split import equal_bytes

BUFFERS = ('decode_swa_indices', 'decode_swa_lens', 'is_valid_token', 'no_replay_start', 'token_to_req_indices')


def tensor(obj: object, name: str) -> torch.Tensor:
    return cast(torch.Tensor, attr(obj, name))


@contextmanager
def forbid_compile() -> Iterator[None]:
    """A compiler entry on the request-shaped second pass is a hard failure."""
    def refuse(*args: object, **kwargs: object) -> object:
        raise ValueError('UP3 request-time Triton compilation')
    with ExitStack() as stack:
        for module_name in ('triton', 'triton.compiler', 'triton.compiler.compiler'):
            module = importlib.import_module(module_name)
            stack.enter_context(replaced(module, 'compile', refuse))
        yield


def loaded_builders(r: object, builder_cls: type) -> list[object]:
    result = []
    for groups in cast(Sequence[Sequence[object]], attr(r, 'attn_groups')):
        if len(groups) != 1:
            continue
        builder = invoke(attr(groups[0], 'get_metadata_builder'), 0)
        if isinstance(builder, builder_cls):
            result.append(builder)
    require(bool(result), 'no actual target SWA groups')
    return result


def clone_builder(builder: object) -> object:
    result = copy.copy(builder)
    for name in BUFFERS:
        setattr(result, name, tensor(builder, name).clone())
    setattr(result, 'dsv41_swa_plan', None)
    return result


def stock(builder: object, module: object, cm: object, mapping: torch.Tensor,
          replay: torch.Tensor, n: int, *, noncausal: bool = False) -> None:
    indices, lens, valid = (tensor(builder, k) for k in BUFFERS[:3])
    slot = tensor(cm, 'slot_mapping')
    torch.ge(slot, 0, out=valid[:len(slot)])
    lens[n:] = 0
    if noncausal:
        indices = tensor(builder, 'decode_swa_indices_noncausal')
        invoke(attr(module, '_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL'), indices, lens,
            attr(builder, 'window_size'), attr(builder, 'noncausal_index_width'),
            tensor(cm, 'query_start_loc'), tensor(cm, 'seq_lens'), mapping, valid,
            tensor(cm, 'block_table_tensor'), attr(builder, 'block_size'), num_tokens=n, token_offset=0)
    else:
        invoke(attr(module, '_COMPUTE_SWA_INDICES_AND_LENS_KERNEL'), indices, lens,
            attr(builder, 'window_size'), indices.shape[-1], lens, lens,
            tensor(cm, 'query_start_loc'), tensor(cm, 'seq_lens'), mapping, valid,
            tensor(cm, 'block_table_tensor'), attr(builder, 'block_size'), replay,
            num_tokens=n, token_offset=0)


def metadata_inputs(builder: object, batch: int, width: int, context: int,
                    replaying: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor, int]:
    dev = tensor(builder, 'decode_swa_lens').device
    n, capacity = batch*width, len(tensor(builder, 'is_valid_token'))
    require(n+8 <= capacity, 'loaded metadata capacity insufficient for padding')
    block = cast(int, attr(builder, 'block_size'))
    # Pitched tables and nonzero/high physical IDs exercise actual block geometry.
    cols = (327680+block-1)//block
    backing = torch.arange(batch*(cols+17), dtype=torch.int32, device=dev).reshape(batch, cols+17)
    table = (backing+100003)[:, :cols]
    qsl = torch.arange(batch+1, dtype=torch.int32, device=dev)*width
    seq = torch.full((batch,), max(context, width), dtype=torch.int32, device=dev)
    mapping = torch.arange(batch, dtype=torch.int32, device=dev).repeat_interleave(width)
    replay = torch.clamp(seq-width-31, min=0) if replaying else torch.zeros_like(seq)
    slots = torch.arange(n+8, dtype=torch.int64, device=dev)+17
    slots[::7] = -1
    return table, slots, qsl, seq, mapping, replay, n


def check_groups(builders: list[object], fused: object, sparse: object) -> Cases:
    require(bool(builders), 'no groups')
    plan = invoke(attr(fused, 'SWADecodePlan'))
    observations, bytes_count = 0, 0
    exclusions = {name: 0 for name in ('mixed-stock', 'noncausal-stock', 'capture-stock')}
    for batch in range(1, 5):
        for width in (1, 5, 6):
            for context in (128, 32000, 150000, 327680):
                for replaying in (False, True):
                    candidates, references, entries, cms = [], [], [], []
                    for actual in builders:
                        candidate, ref = clone_builder(actual), clone_builder(actual)
                        table, slots, qsl, seq, mapping, replay, n = metadata_inputs(actual, batch, width, context, replaying)
                        for builder in (candidate, ref):
                            tensor(builder, 'decode_swa_indices').fill_(-123)
                            tensor(builder, 'decode_swa_lens').fill_(-123)
                            tensor(builder, 'is_valid_token').fill_(False)
                        cm = SimpleNamespace(block_table_tensor=table, slot_mapping=slots,
                            causal=True, query_start_loc=qsl, seq_lens=seq)
                        candidates.append(candidate)
                        references.append(ref)
                        entries.append((candidate, table, slots))
                        cms.append(cm)
                    # The real plan's claim key requires shared step vectors across groups.
                    for cm in cms:
                        cm.query_start_loc, cm.seq_lens = cms[0].query_start_loc, cms[0].seq_lens
                    invoke(attr(plan, 'arm'), entries)
                    before = cast(int, attr(plan, 'launches'))
                    for index, (candidate, ref, cm) in enumerate(zip(candidates, references, cms)):
                        require(invoke(attr(plan, 'claim'), candidate, cm, n, 0, mapping, replay) is True,
                                'eligible target group did not use fused path')
                        stock(ref, sparse, cm, mapping, replay, n)
                        for name in BUFFERS[:3]:
                            bytes_count += equal_bytes(tensor(candidate, name), tensor(ref, name), f'0067 {name} group{index}')
                    require(attr(plan, 'launches') == before+1 and attr(plan, 'disabled') is False,
                            '0067 did not launch exactly once, or exception-disabled')
                    # Same signatures, warmed once; compile failure must not silently disable.
                    invoke(attr(plan, 'disarm'))
                    with forbid_compile():
                        invoke(attr(plan, 'arm'), entries)
                        require(invoke(attr(plan, 'claim'), candidates[0], cms[0], n, 0, mapping, replay) is True,
                                'second eligible launch failed/compiled')
                    require(attr(plan, 'disabled') is False, 'compile exception silently disabled fusion')
                    for name, prefills, causal in (('mixed-stock', 1, True), ('noncausal-stock', 0, False)):
                        cms[0].causal = causal
                        invoke(attr(plan, 'arm'), entries)
                        count = attr(plan, 'launches')
                        require(invoke(attr(plan, 'claim'), candidates[0], cms[0], n, prefills, mapping, replay) is False
                            and attr(plan, 'launches') == count, name+' took fused path')
                        exclusions[name] += 1
                    cms[0].causal = True
                    invoke(attr(plan, 'arm'), entries)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = invoke(attr(plan, 'claim'), candidates[0], cms[0], n, 0, mapping, replay)
                    require(captured is False, 'capture took fused path')
                    exclusions['capture-stock'] += 1
                    graph.replay()
                    invoke(attr(plan, 'disarm'))
                    observations += 1
    # Real mirror, real device bytes, copy counts from actual changed-value decisions.
    mirror = invoke(attr(fused, 'ReplayStartMirror'), 4)
    device = tensor(builders[0], 'no_replay_start').device
    target = torch.zeros(4, dtype=torch.int32, device=device)
    copies = 0
    for values in (np.zeros(4, dtype=np.int32), np.array([0, 100, 12, 0], dtype=np.int32),
                   np.array([0, 100, 12, 0], dtype=np.int32), np.zeros(4, dtype=np.int32)):
        same = invoke(attr(mirror, 'holds'), values)
        if same is False:
            target.copy_(torch.from_numpy(values).to(device))
            invoke(attr(mirror, 'wrote'), values)
            copies += 1
        equal_bytes(target, torch.from_numpy(values).to(device), '0067 replay mirror')
    require(copies == 2 and attr(mirror, 'skipped') == 2, '0067 replay copy count differs')
    result: Cases = {name: passed(observations, comparison='every-byte-and-index', bytes_compared=bytes_count)
        for name in ('valid', 'lens', 'indices', 'padded', 'invalid', 'replay', 'ctx327680')}
    result.update({name: passed(count) for name, count in exclusions.items()})
    result['actual-groups'] = passed(len(builders), count=len(builders), capacity_mode=True,
        derived_from_loaded_engine=True, geometry=[{'window': cast(int, attr(v, 'window_size')),
        'block': cast(int, attr(v, 'block_size')), 'indices_shape': list(tensor(v, 'decode_swa_indices').shape),
        'stride': list(tensor(v, 'decode_swa_indices').stride())} for v in builders])
    result['launch'] = passed(observations, once_per_eligible_step=True, request_time_compile=False,
        exception_disabled=False, replay_copy_only_on_change=True)
    return result


def state_exclusions(state: object) -> Json:
    """Execute the installed model-state routing, with an inert stock delegate.

    This isolates the capture/ubatch selection itself. The delegate cannot write
    live builder/KV state. Mixed/noncausal stock kernels are checked separately.
    """
    cls = type(state)
    parent = next(base for base in cls.__mro__[1:] if 'prepare_attn' in base.__dict__)
    cloned = copy.copy(state)
    setattr(cloned, '_replay', (0, None))
    setattr(cloned, 'decoder_replay_layers', None)
    calls = 0
    def stock_delegate(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        plan = attr(cloned, '_swa_plan')
        require(attr(plan, 'armed') is False, 'model-state exclusion armed fusion')
        return {}
    with replaced(parent, 'prepare_attn', stock_delegate):
        for capture, ubatch in ((True, 0), (False, 1)):
            require(invoke(attr(cloned, 'prepare_attn'), None, None, (), None, [], None,
                for_capture=capture, ubatch_idx=ubatch) == {}, 'stock delegation changed')
    require(calls == 2, 'model-state exclusions did not delegate')
    return passed(calls, installed_state_routing=True)


def run(worker: object) -> Cases:
    r = runner(worker)
    require(type(attr(r, 'speculator')).__name__ == 'RemoteDSparkCapacitySpeculator', 'capacity speculator required')
    sparse = importlib.import_module('vllm.v1.attention.backends.mla.sparse_swa')
    fused = importlib.import_module('vllm.models.deepseek_v41.common.ops.swa_fused')
    require(attr(fused, 'SWA_FUSED') is True and attr(fused, 'REPLAY_COPY_SKIP') is True, '0067 both switches required')
    actual_plan = attr(attr(r, 'model_state'), '_swa_plan')
    require(actual_plan is not None and attr(actual_plan, 'disabled') is False
        and attr(actual_plan, '_warm') is True, 'actual startup SWA plan was not warmed or was exception-disabled')
    builders = loaded_builders(r, cast(type, attr(sparse, 'DeepseekSparseSWAMetadataBuilder')))
    result = check_groups(builders, fused, sparse)
    result['microbatch-stock'] = state_exclusions(attr(r, 'model_state'))
    torch.cuda.synchronize(tensor(builders[0], 'decode_swa_lens').device)
    return result
