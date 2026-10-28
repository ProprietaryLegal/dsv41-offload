"""K5 qualification: actual proposals/24-row target observations and native noncausal oracle.

The observer must span real public qualification requests. Dummy/warmup proposals
are deliberately ignored: capacity mode returns zeros without contacting its
proxy for those calls. A configured depth or allocated tensor cannot pass this gate.
"""
from __future__ import annotations

import hashlib
import importlib
from contextlib import ExitStack
from typing import cast

import torch

from offload.prefill_maxlen.evidence import Json
from offload.up3.engine_common import Cases, attr, invoke, passed, replaced, require, runner
from offload.up3.engine_split import equal_bytes
from offload.up3.engine_vor import clone_builder, loaded_builders, metadata_inputs, stock, tensor


def noncausal(builders: list[object], sparse: object) -> Json:
    compared = 0
    samples = 0
    for original in builders:
        builder = clone_builder(original)
        width = cast(int, attr(builder, 'noncausal_index_width'))
        require(width >= cast(int, attr(builder, 'window_size'))+6, 'noncausal block truncated')
        shape = list(tensor(builder, 'decode_swa_indices').shape)
        shape[-1] = width
        out = torch.full(shape, -123, dtype=torch.int32, device=tensor(builder, 'decode_swa_lens').device)
        setattr(builder, 'decode_swa_indices_noncausal', out)
        for context in (128, 32000, 150000, 327680):
            table, slots, qsl, seq, mapping, replay, n = metadata_inputs(builder, 4, 6, context, False)
            # Invalid rows are checked too, while row1 of each request sees future rows.
            from types import SimpleNamespace
            cm = SimpleNamespace(slot_mapping=slots, block_table_tensor=table, query_start_loc=qsl, seq_lens=seq)
            stock(builder, sparse, cm, mapping, replay, n, noncausal=True)
            expected = torch.full_like(out, -123)
            expected_lens = torch.zeros_like(tensor(builder, 'decode_swa_lens'))
            for t in range(n):
                if int(slots[t].item()) < 0:
                    expected[t] = -1
                    continue
                req = t//6
                end = int(seq[req].item())
                start = max(end-6-cast(int, attr(builder, 'window_size')), 0)
                positions = torch.arange(start, start+width, device=out.device)
                block = cast(int, attr(builder, 'block_size'))
                # Clamp masked columns, matching the kernel without invalid address arithmetic.
                safe = positions.clamp(max=end-1)
                indices = table[req, safe//block]*block+safe%block
                expected[t] = torch.where(positions < end, indices, -1).reshape_as(expected[t])
                expected_lens[t] = end-start
            compared += equal_bytes(out, expected, 'K5 full future-visible noncausal block')
            compared += equal_bytes(tensor(builder, 'decode_swa_lens'), expected_lens, 'K5 noncausal lens')
            samples += 1
    return passed(samples, byte_identical=True, bytes_compared=compared, future_queries_visible=True)


class Observer:
    """Attach to the real capacity proposer and target metadata builder; remove before timing."""
    def __init__(self, worker: object):
        self.r = runner(worker)
        self.spec = attr(self.r, 'speculator')
        require(type(self.spec).__name__ == 'RemoteDSparkCapacitySpeculator', 'actual capacity proposer required')
        require(attr(self.spec, 'num_speculative_steps') == 5, 'five configured proposal slots required')
        schedule = attr(attr(attr(worker, 'vllm_config'), 'speculative_config'), 'num_speculative_tokens_per_batch_size')
        require(schedule == [(1, 4, 5)] or schedule == [[1, 4, 5]], 'isolated K5 schedule changed')
        self.stack = ExitStack()
        self.proposals: dict[int, int] = {}
        self.rows = 0
        self.hashes: list[str] = []
        self.remote_ok = 0

    def install(self) -> None:
        require(not self.proposals and self.rows == 0, 'observer cannot be reused')
        remote = attr(self.spec, 'remote')
        old_run, old_propose = attr(remote, 'run'), attr(self.spec, 'propose')
        state = attr(self.r, 'model_state')
        old_prepare = attr(state, 'prepare_attn')
        def remote_run(*args: object, **kwargs: object) -> object:
            result = invoke(old_run, *args, **kwargs)
            require(invoke(attr(result, 'read_status')) == 0
                and bool((cast(torch.Tensor, attr(result, 'ok_mask')) == 1).all()), 'remote fallback/failed proposal')
            self.remote_ok += 1
            return result
        def propose(*args: object, **kwargs: object) -> object:
            before = self.remote_ok
            result = invoke(old_propose, *args, **kwargs)
            # Native positional dummy_run is argument12; normal V2 runner passes keywords.
            dummy = kwargs.get('dummy_run', args[12] if len(args) > 12 else False)
            batch = kwargs.get('input_batch', args[0] if args else None)
            if dummy is True or batch is None:
                return result
            n = cast(int, attr(batch, 'num_reqs'))
            require(self.remote_ok == before+1, 'proposal returned without actual remote work')
            values = cast(torch.Tensor, result)
            require(tuple(values.shape) == (n, 5) and bool((values >= 0).all()), 'actual five proposals unavailable')
            confidence = tensor(self.spec, 'draft_token_confidence_probs')[:n]
            require(bool(torch.isfinite(confidence).all()), 'invalid proposal confidence')
            self.proposals[n] = self.proposals.get(n, 0)+1
            self.hashes.append(hashlib.sha256(values.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
            return result
        def prepare(*args: object, **kwargs: object) -> object:
            result = invoke(old_prepare, *args, **kwargs)
            batch = kwargs.get('input_batch', args[0] if args else None)
            if batch is not None and attr(batch, 'num_reqs') == 4 and attr(batch, 'num_tokens') == 24:
                if kwargs.get('for_capture', False) is True or bool(tensor(batch, 'is_padding').all()):
                    return result
                qsl = tensor(batch, 'query_start_loc').cpu()
                require(qsl.tolist() == [0, 6, 12, 18, 24], 'B4 target rows are not 4x6')
                require(attr(batch, 'has_prefill') is False, '24 rows were prefill, not target verification')
                self.rows += 1
            return result
        self.stack.enter_context(replaced(remote, 'run', remote_run))
        self.stack.enter_context(replaced(self.spec, 'propose', propose))
        self.stack.enter_context(replaced(state, 'prepare_attn', prepare))

    def close(self) -> None:
        self.stack.close()

    def collect(self, noncausal_evidence: Json) -> Cases:
        require(all(self.proposals.get(n, 0) > 0 for n in (3, 4)) and self.rows > 0,
                'real B3/B4 proposals and B4 24-row target step required; dummy evidence refuses')
        return {'five-proposals': passed(sum(self.proposals.values()), proposals=5,
                    b3_samples=self.proposals[3], b4_samples=self.proposals[4],
                    proposal_sha256s=cast(list[Json], self.hashes)),
                'b4-24-rows': passed(self.rows, rows=24), 'noncausal-stock': noncausal_evidence}


def native_gate(worker: object) -> Json:
    r = runner(worker)
    sparse = importlib.import_module('vllm.v1.attention.backends.mla.sparse_swa')
    return noncausal(loaded_builders(r, cast(type, attr(sparse, 'DeepseekSparseSWAMetadataBuilder'))), sparse)
