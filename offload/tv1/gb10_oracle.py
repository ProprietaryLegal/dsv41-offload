"""Coordinator-only installed SM121 oracle; production-derived geometry, no model data.

The pipe entry point imports the SAME metadata adapter used by candidate.patch.
Both ranks must pass the full receipt validator. No lane may execute this remotely.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def main() -> None:
    import torch
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask  # type: ignore[import-not-found]
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk  # type: ignore[import-not-found]
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata  # type: ignore[import-not-found]
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder  # type: ignore[import-not-found]
    from vllm.v1.worker.workspace import init_workspace_manager  # type: ignore[import-not-found]
    from vllm_exl3_moe.tv1_candidate_cases import EXCLUSIONS, case_matrix, validate_oracle_receipt  # type: ignore[import-not-found]
    from vllm_exl3_moe.tv1_candidate_metadata import score_decode  # type: ignore[import-not-found]
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError('TV1 qualification requires actual SM121')
    torch.cuda.set_per_process_memory_fraction(.025)
    torch.manual_seed(41)
    init_workspace_manager(torch.device('cuda'), num_ubatches=1)
    select = get_indexer_topk('auto')
    results: list[dict[str, Any]] = []
    # Reuse one high-offset storage allocation across all cases. Only the actual
    # high physical pages are initialized/read; the gap makes int32 addressing fail.
    matrix = case_matrix()
    width, page, stride, offset, high = (matrix[0][k] for k in ('width', 'page', 'stride', 'cache_offset', 'high_page'))
    count = (width + page - 1) // page
    physical_count = high + count + 1
    storage_bytes = (physical_count - 1) * stride + offset + page * 132
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < storage_bytes + 512 * 1024**2:
        raise ValueError(f'TV1 isolated oracle needs {storage_bytes} contiguous virtual CUDA bytes '
                         f'plus 512MiB scratch; free={free_bytes}, total={total_bytes}; STOP')
    try:
        storage = torch.empty(storage_bytes, device='cuda', dtype=torch.uint8)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as exc:
        raise ValueError(f'TV1 high-offset CUDA allocation failed: {storage_bytes} bytes; '
                         'MemAvailable is insufficient evidence of allocatability; STOP') from exc
    cache = torch.as_strided(storage, (physical_count, page, 1, 132), (stride, 132, 132, 1), offset)
    live_pages = torch.as_strided(storage, (count, page * 132), (stride, 1), offset + high * stride)
    for spec in matrix:
        rows, next_n, visible = spec['rows'], spec['next_n'], spec['visible_width']
        metadata_batch = rows // next_n
        if rows % next_n:
            raise ValueError('TV1 matrix metadata rows do not divide')
        numeric = spec['numeric']
        if numeric == 'random':
            keys = torch.randn(count, page, 128, device='cuda').to(torch.float8_e4m3fn)
            scales = torch.exp2(torch.randint(-3, 3, (count, page), device='cuda').float())
            q = torch.randn(metadata_batch, next_n, 32, 128, device='cuda').to(torch.float8_e4m3fn)
            weights = torch.randn(rows, 32, device='cuda') / 64
        else:
            keys = torch.ones(count, page, 128, device='cuda').to(torch.float8_e4m3fn)
            scales = torch.ones(count, page, device='cuda')
            if numeric == 'near-tie':
                # Adjacent representable fp32 scale values at the top-k boundary;
                # deliberately avoid tolerance/overlap and a wide separation.
                scales.view(-1)[1::2] += torch.finfo(torch.float32).eps
            q = torch.full((metadata_batch, next_n, 32, 128), .0625, device='cuda').to(torch.float8_e4m3fn)
            weights = torch.full((rows, 32), 1 / 32, device='cuda')
        live_pages[:, :page * 128].copy_(keys.view(torch.uint8).reshape(count, -1))
        live_pages[:, page * 128:page * 132].copy_(scales.view(torch.uint8).reshape(count, -1))
        # The same request's expanded rows share its page table; permutations
        # differ per request and operate near/above 2 GiB physical byte offsets.
        base = torch.randperm(count, device='cuda').int() + high
        request_pages = torch.stack([base.roll(i * 17) for i in range(spec['batch'])])
        request_lens = torch.full((spec['batch'],), visible, device='cuda', dtype=torch.int32)
        decode_lens_cpu = torch.full((spec['batch'],), spec['expansion'], dtype=torch.int32)
        decode_lens = decode_lens_cpu.to('cuda')
        starts = torch.arange(spec['batch'], device='cuda', dtype=torch.int32) * spec['expansion']
        builder = SimpleNamespace(
            vllm_config=SimpleNamespace(speculative_config=None), supports_varlen=False,
            decode_seq_lens_buffer=torch.zeros(rows + 8, device='cuda', dtype=torch.int32),
            expanded_block_table_buffer=torch.zeros(rows + 8, count, device='cuda', dtype=torch.int32),
            decode_lens_buffer=torch.zeros(rows + 8, device='cuda', dtype=torch.int32),
            arange_buffer=torch.arange(rows + 8, device='cuda', dtype=torch.int32),
            offsets_buffer=torch.arange(8, device='cuda', dtype=torch.int32))
        # Execute the INSTALLED production builder, including its uniform Triton
        # expansion or its ragged/graph-tail expansion. Then execute our patch's
        # identical adapter. No independently reimplemented metadata oracle.
        lens, pages, _, built_batch, requires_padding = DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors(
            builder, seq_lens=request_lens, block_table=request_pages,
            decode_lens=decode_lens, decode_lens_cpu=decode_lens_cpu,
            query_start_loc=starts, num_decodes=spec['batch'], num_decode_tokens=rows,
            use_native=spec['layout'] in ('single', 'native2'),
            next_n=spec['expansion'], max_decode_len=spec['expansion'])
        if requires_padding or built_batch != metadata_batch:
            raise ValueError('TV1 installed metadata expansion differs from pinned case')
        if lens.ndim == 1:
            lens = lens.unsqueeze(-1)
        candidates = torch.full((rows, 2048), -1, device='cuda', dtype=torch.int32)
        nblocks = (visible + 7) // 8
        nc = min(2047, nblocks)
        choices = torch.randperm(nblocks, device='cuda')[:nc].int()
        for row in range(rows):
            candidates[row, :nc] = choices.roll(row)
        candidates[:, -2] = (visible - 1) // 8  # exact tail block; -1 pad at end
        if rows > 1:
            candidates[1].fill_(-1)  # empty candidates with occupied context
        schedule = get_paged_mqa_logits_metadata(lens, page,
            torch.cuda.get_device_properties(0).multi_processor_count)
        a = torch.full((rows, 512), -1, device='cuda', dtype=torch.int32)
        b = torch.full_like(a, -1)

        def run() -> None:
            dense = fp8_fp4_paged_mqa_logits((q, None), cache, weights, lens, pages,
                                           schedule, max_model_len=width, clean_logits=False, indices=None)
            apply_candidate_mask(dense, None, lens.reshape(-1), candidates, 8, 1)
            sparse = score_decode(q, cache, weights, lens, pages, candidates, width, 8)
            if select.resolve_backend(dense, 512, rows) != spec['backend']:
                raise ValueError('TV1 installed top-k backend differs from matrix')
            # Installed top-k scatters valid indices only. Match the engine's
            # -1 initialization for empty/tail rows, including graph replay.
            a.fill_(-1)
            b.fill_(-1)
            select(dense, lens, next_n, a, 512, visible)
            select(sparse, lens, next_n, b, 512, visible)

        run()
        if not torch.equal(a, b):
            raise ValueError(f'TV1 eager selected-index mismatch: {spec["id"]}: STOP')
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        q.copy_((-q.float()).to(q.dtype) if numeric == 'random' else (q.float() * 2).to(q.dtype))
        lens.sub_(visible // 2).clamp_min_(0)
        pages.copy_(pages.flip(1))
        candidates.copy_(candidates.roll(1, 1))
        if rows > 1:
            lens[-1].zero_()
        schedule.copy_(get_paged_mqa_logits_metadata(lens, page,
            torch.cuda.get_device_properties(0).multi_processor_count))
        a.fill_(-17)
        b.fill_(-19)
        graph.replay()
        if not torch.equal(a, b):
            raise ValueError(f'TV1 changed replay selected-index mismatch: {spec["id"]}: STOP')
        results.append({**spec, 'eager_identical': True, 'replay_identical': True})
        # Drop graph before it pins this case's allocations into the next case.
        del graph, builder, request_pages, request_lens, decode_lens, decode_lens_cpu, starts
        del keys, scales, q, weights, lens, pages, candidates, schedule, a, b
    receipt = {'type': 'tv1-installed-oracle-v2', 'classification': 'MEASURED SM121',
               'capability': [12, 1], 'exclusions': EXCLUSIONS, 'cases': results, 'all_identical': True}
    validate_oracle_receipt(receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
