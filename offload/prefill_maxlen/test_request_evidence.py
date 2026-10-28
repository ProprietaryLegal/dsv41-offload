from __future__ import annotations

from offload.prefill_maxlen.request_evidence import bench, needle, prefix


def test_stream_timing_is_not_prefill_and_content_is_excluded() -> None:
    row = bench({'usage': {'prompt_tokens': 500000, 'prompt_tokens_details': {'cached_tokens': 400000}},
                 'total_s': 340, 'ttft_s': 310, 'first_by_kind_s': {'reasoning': 310},
                 'last_by_kind_s': {'reasoning': 339}, 'answer_text': 'PRIVATE', 'error': None})
    assert row['uncached_prompt_tokens'] == 100000
    assert row['after_first_token_s'] == 30
    assert row['reasoning_delivery_span_s'] == 29
    assert row['prefill_gpu_s'] is None
    assert 'PRIVATE' not in str(row)
    assert bench({'usage': {}, 'error': 'PRIVATE'})['uncached_prompt_tokens'] is None


def test_nonstreaming_missing_boundaries_stay_unknown() -> None:
    row = needle({'usage': {'prompt_tokens': 498245, 'completion_tokens': 43,
                           'prompt_tokens_details': {'cached_tokens': 0},
                           'completion_tokens_details': {'reasoning_tokens': 0}},
                  'ctx': 500000, 'rep': 0, 'slot': 0, 'concurrent': 1, 'elapsed_s': 319.94, 'ok': True})
    assert row['uncached_prompt_tokens'] == 498245
    assert row['ttft_s'] is None
    assert row['answer_and_control_tokens'] == 43
    partial = prefix({'kind': 'partial', 'n': 500000, 'cached': 399744, 'common_tokens': 399994})[0]
    assert partial['uncached_prompt_tokens'] is None


def test_memprobe_preserves_reader_times_and_only_consecutive_host_spacing() -> None:
    from offload.prefill_maxlen.request_evidence import memory_timing
    from offload.prefill_maxlen.evidence import record
    rows = memory_timing([
        {'ev': 'phase', 'layer': 2, 'tag': 'ffn', 'toks': 4096, 't': 1.1},
        {'ev': 'step', 'n': 1, 'kind': 'prefill', 'toks': 4096, 'reqs': 1, 't': 2,
         'eng': [[3, 4.5, 2.5, 0]]},
        {'ev': 'step', 'n': 2, 'kind': 'prefill', 'toks': 4096, 'reqs': 1, 't': 4},
        {'ev': 'step', 'n': 512, 'kind': 'decode', 'toks': 6, 'reqs': 1, 't': 50},
    ])
    assert record(rows[2])['host_end_to_end_s'] == 2
    assert record(rows[3])['host_end_to_end_s'] is None
    assert record(rows[1])['engram_reader_timing'] == [
        {'table': 0, 'reads_since_previous_record': 3, 'reader_service_sum_ms': 4.5,
         'reader_service_running_max_ms': 2.5, 'errors': 0}]
    assert all(record(row)['gpu_prefill_s'] is None for row in rows)
