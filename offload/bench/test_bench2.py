"""Focused parser/accounting/scheduling tests; never contact the serving model."""
from __future__ import annotations

import argparse
import io
import json
import os
import runpy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import bench2 as b
import report

# Public release: the workload text is not shipped (data boundary).
NEEDS_SPEC = 'needs the workload spec named by DSV41_BENCH_SPEC_FILE (not shipped)'


class NullTelemetry:
    def __init__(self, *args: object) -> None:
        self.samples: list[b.JSON] = []
        self.errors: list[b.JSON] = []
    def start(self) -> None:
        pass
    def finish(self) -> None:
        pass
    def sample(self) -> None:
        pass


def event(delta: b.Object, finish: str | None = None) -> str:
    return 'data: ' + json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]}) + '\n\n'


class BenchTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DSV41_BENCH_SPEC_FILE'), NEEDS_SPEC)
    def test_original_workloads_and_record_builder_preserved(self) -> None:
        original = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'bench.py'))
        self.assertEqual(b.spec()['workloads'], original['spec']()['workloads'])
        self.assertEqual(b.build_record(3000, 42), original['build_record'](3000, 42))

    def test_split_simultaneous_fields_and_server_usage(self) -> None:
        wire = event({'reasoning_content': 'why', 'content': 'yes'})
        wire += event({'content': ' indeed'}, 'stop')
        wire += 'data: ' + json.dumps({'usage': {'prompt_tokens': 11, 'completion_tokens': 7,
                                              'completion_tokens_details': {'reasoning_tokens': 2}}}) + '\n\n'
        wire += 'data: [DONE]\n\n'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(wire.encode())), patch.object(b, 'tok_count', return_value=2):
            row = b.stream_request({})
        self.assertEqual(row['reasoning_text'], 'why')
        self.assertEqual(row['answer_text'], 'yes indeed')
        self.assertEqual(row['reasoning_tokens'], 2)
        self.assertEqual(row['answer_tokens'], 5)
        self.assertIsNotNone(row['first_visible_s'])
        self.assertEqual(row['completion_tokens'], 7)

    def test_channel_retokenization_is_explicit(self) -> None:
        wire = event({'reasoning': 'two chunks'}) + event({}, 'length')
        wire += 'data: {"usage":{"prompt_tokens":1,"completion_tokens":9}}\n\ndata: [DONE]\n'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(wire.encode())), patch.object(b, 'tok_count', return_value=3):
            row = b.stream_request({})
        self.assertEqual(row['reasoning_tokens'], 3)
        self.assertEqual(row['reasoning_token_source'], 'retokenized_text')
        self.assertIsNone(row['first_visible_s'])
        self.assertEqual(row['completion_tokens'], 9)

    def test_truncated_stream_is_not_success(self) -> None:
        with patch('urllib.request.urlopen', return_value=io.BytesIO(event({'content': 'partial'}).encode())), self.assertRaisesRegex(RuntimeError, 'incomplete stream'):
            b.stream_request({})

    def test_metric_deltas_labels_and_reset(self) -> None:
        before = b.parse_metrics('vllm:spec_decode_num_drafts_total{engine="0"} 10\nvllm:spec_decode_num_draft_tokens_total 40\nvllm:spec_decode_num_accepted_tokens_total 20\n')
        after = b.parse_metrics('vllm:spec_decode_num_drafts_total{engine="0"} 12\nvllm:spec_decode_num_draft_tokens_total 48\nvllm:spec_decode_num_accepted_tokens_total 25\nvllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",engine="0"} 2\n')
        row = b.spec_delta(before, after)
        self.assertEqual(row['mean_accepted_per_round'], 2.5)
        self.assertEqual(row['acceptance_per_position_per_round'], {'0': 1.0})
        self.assertEqual(row['proposed_tokens'], 8)
        self.assertEqual(row['accepted_tokens'], 5)
        with self.assertRaisesRegex(RuntimeError, 'reset/disappeared'):
            b.spec_delta(after, before)
        self.assertIsNone(b.spec_delta({}, {})['rounds'])

    def test_idle_fail_closed(self) -> None:
        cases: tuple[dict[str, float], ...] = ({}, {'vllm:num_requests_running': 0}, {'vllm:num_requests_running': 1, 'vllm:num_requests_waiting': 0})
        for metrics in cases:
            with self.assertRaises(RuntimeError):
                b.require_idle(metrics)
        b.require_idle({'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 0})

    @unittest.skipUnless(os.environ.get('DSV41_BENCH_SPEC_FILE'), NEEDS_SPEC)
    def test_prompt_counts_include_template_and_instruction(self) -> None:
        with patch.object(b, 'build_real', return_value='x' * 200), patch.object(b, 'prompt_tokens', side_effect=lambda p, t: len(p)):
            prompt, count = b.make_prompt(300, 'extract', 'real', 1, {})
        self.assertEqual(count, len(prompt))
        self.assertLessEqual(count, 300)
        self.assertIn(b.spec()['workloads']['extract'], prompt)

    def test_settings_priority_and_salts(self) -> None:
        args = argparse.Namespace(max_tokens=10, temperature=1, top_p=.9, seed=42, thinking=True, reasoning_effort='high', priority=10)
        row = b.request_body('input', args, 'salt')
        self.assertEqual(row['cache_salt'], 'salt')
        self.assertEqual(row['priority'], 10)
        self.assertEqual(row['chat_template_kwargs'], {'thinking': True, 'reasoning_effort': 'high'})

    def test_concurrent_and_mixed_overlap_and_jsonl(self) -> None:
        lock = threading.Lock()
        active = 0
        peak = 0
        def stream(body: b.Object, started: threading.Event | None = None) -> b.Object:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            if started:
                started.set()
            time.sleep(.1)
            with lock:
                active -= 1
            return {'completion_tokens': 10, 'total_s': .1, 'decode_tps': 100, 'ttft_s': .01, 'first_visible_s': .02, 'finish': 'stop'}
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(label='A', max_tokens=10, temperature=1, top_p=1, seed=42, thinking=True,
                                      reasoning_effort='high', priority=10, keep_text=False, spark_hosts=[], x86host_host='unused',
                                      process_name=[], extra_metrics=[], x86host_nic=[], out=str(Path(tmp)/'rows.jsonl'),
                                      mixed_timeout=1, mixed_delay=0, metrics_settle=0, cache_mode='cold')
            with patch.object(b, 'Telemetry', NullTelemetry), patch.object(b, 'scrape', return_value='vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'), patch.object(b, 'stream_request', side_effect=stream):
                for mixed in (False, True):
                    row = b.run_group(args, {}, 32000, 'legal', 3, 0, [('input', 5)]*3, ['s']*3, mixed)
                    self.assertEqual(row['delivered_tokens'], 30)
                    self.assertEqual(row['request_count'], 3)
                    self.assertIsInstance(row['aggregate_tps'], float)
            self.assertEqual(peak, 3)
            rows = [json.loads(x) for x in Path(args.out).read_text().splitlines()]
            self.assertEqual(sum(r['type']=='run' for r in rows), 2)
            self.assertEqual(sum(r['type']=='request' for r in rows), 6)
            rendered = report.table([Path(args.out)])
            self.assertIn('1/1', rendered)
            self.assertIn('±', rendered)

    def test_dry_run_never_calls_inference_or_tokenize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'dry.jsonl'
            with patch('sys.argv', ['bench2', '--dry-run', '--out', str(path)]), patch.object(b, 'models', return_value={'data':[{'id':'fixture'}]}), patch.object(b, 'scrape', return_value=''), patch.object(b, 'Telemetry', NullTelemetry), patch.object(b, 'post') as post, patch.object(b, 'stream_request') as stream:
                b.main()
            post.assert_not_called()
            stream.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['generation_requests'], 0)

    def test_report_duplicate_detection_and_repeat_spread(self) -> None:
        self.assertEqual(report.spread([1,3]), '2.00 ± 1.41')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.jsonl'
            row = {'type':'request', 'run_id':'same', 'request_index':0}
            path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                report.table([path])


if __name__ == '__main__':
    unittest.main()
