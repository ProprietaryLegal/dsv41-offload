"""CPU-only request/early-stop and no-output-disclosure regressions."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("warmup", Path(__file__).parents[1] / "warmup.py")
assert SPEC and SPEC.loader
warmup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warmup)


def reply(count: int = 32, reason: str = "length") -> str:
    return json.dumps({"usage": {"prompt_tokens": 5041, "completion_tokens": count},
                       "choices": [{"text": "DO NOT DISCLOSE", "finish_reason": reason,
                                    "stop_reason": None}]})


class WarmupTest(unittest.TestCase):
    def test_internal_budget(self) -> None:
        body = warmup.payload('quoted"model', "salt", 1, 0)
        self.assertEqual(json.loads(json.dumps(body)), body)
        self.assertEqual(body["min_tokens"], body["max_tokens"])
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(body["priority"], 20)
        self.assertNotIn("messages", body)

    def test_one_token_and_wrong_reason_fail(self) -> None:
        for raw in (reply(1, "stop"), reply(32, "stop"), '{"error": "DO NOT DISCLOSE"}'):
            with self.assertRaises(ValueError) as err:
                warmup.summary(raw)
            self.assertNotIn("DO NOT DISCLOSE", str(err.exception))

    def test_missing_metrics_are_unknown(self) -> None:
        record = warmup.summary(reply())
        self.assertIsNone(record["speculative_metrics"])
        self.assertIsNone(record["reasoning_tokens"])
        self.assertNotIn("DO NOT DISCLOSE", json.dumps(record))

    def test_decode_waves_do_not_serialize_long_prefills(self) -> None:
        long = warmup.payload("test", "salt", 1, 0)
        short = warmup.payload("test", "salt", 4, 0)
        self.assertGreater(len(str(long["prompt"])), 20000)
        self.assertLess(len(str(short["prompt"])), 200)

    def test_report_actual_spec_counts_without_text(self) -> None:
        data = json.loads(reply())
        data["metrics"] = {"speculative_decoding": {"num_spec_steps": 12,
                           "num_accepted_draft_tokens": 19, "num_draft_tokens": 60}}
        data["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
        result = warmup.summary(json.dumps(data))
        self.assertEqual(result["speculative_metrics"]["num_spec_steps"], 12)
        self.assertEqual(result["cached_tokens"], 0)

    def test_idle_fails_closed(self) -> None:
        for raw in ("", "vllm:num_requests_running 0", "vllm:num_requests_running NaN\nvllm:num_requests_waiting 0",
                    "vllm:num_requests_running 1\nvllm:num_requests_waiting 0"):
            with self.assertRaises(ValueError):
                warmup.idle(raw)

    def test_all_shapes_without_network(self) -> None:
        bodies: list[dict[str, object]] = []

        def read(path: str, body: dict[str, object] | None) -> str:
            if body is None:
                self.assertEqual(path, "/metrics")
                return 'vllm:num_requests_running{model_name="test"} 0\nvllm:num_requests_waiting 0'
            bodies.append(body)
            return reply()

        output = io.StringIO()
        with patch.object(warmup, "urlopen", side_effect=AssertionError("network forbidden")), redirect_stdout(output):
            warmup.run("test", read)
        self.assertEqual(len(bodies), 10)
        self.assertEqual(len({str(body["prompt"]) for body in bodies}), 10)
        self.assertEqual([json.loads(line)["requested_concurrency"] for line in output.getvalue().splitlines()], [1, 2, 3, 4])
        self.assertNotIn("DO NOT DISCLOSE", output.getvalue())

    def test_foreign_work_stops_before_post(self) -> None:
        def read(path: str, body: dict[str, object] | None) -> str:
            self.assertIsNone(body)
            return "vllm:num_requests_running 1\nvllm:num_requests_waiting 0"
        with self.assertRaises(ValueError):
            warmup.run("test", read)

    def test_long_prefill_requires_real_uncached_full_chunk(self) -> None:
        for prompt_tokens, cached_tokens in ((4096, 0), (5041, 64)):
            def read(path: str, body: dict[str, object] | None) -> str:
                if body is None:
                    return "vllm:num_requests_running 0\nvllm:num_requests_waiting 0"
                data = json.loads(reply())
                data["usage"]["prompt_tokens"] = prompt_tokens
                data["usage"]["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
                return json.dumps(data)
            with self.assertRaises(ValueError):
                warmup.run("test", read)


    def test_deadline_stops_before_the_next_wave_and_bounds_requests(self) -> None:
        posts: list[dict[str, object]] = []

        def read(path: str, body: dict[str, object] | None) -> str:
            if body is None:
                return "vllm:num_requests_running 0\nvllm:num_requests_waiting 0"
            posts.append(body)
            return reply()

        with self.assertRaisesRegex(ValueError, "deadline"):
            warmup.run("test", read, warmup.time.monotonic() - 1)      # deadline already past: nothing is sent
        self.assertEqual(posts, [])
        with self.assertRaisesRegex(ValueError, "deadline"):
            warmup.request_timeout(warmup.time.monotonic() - 0.1)
        self.assertEqual(warmup.request_timeout(warmup.time.monotonic() + 1000), warmup.TIMEOUT)
        self.assertLessEqual(warmup.request_timeout(warmup.time.monotonic() + 5), 5)
        self.assertLess(warmup.DEADLINE_S + 432, 600 - 60)   # last MEASURED K1 start + budget keeps a minute of guard margin

    def test_late_final_wave_fails_and_timeout_floor_is_the_time_left(self) -> None:
        clock = [0.0]

        def read(path: str, body: dict[str, object] | None) -> str:
            if body is None:
                return "vllm:num_requests_running 0\nvllm:num_requests_waiting 0"
            clock[0] += 30.0                           # every request "takes" 30 s on a fake clock
            return reply()

        with patch.object(warmup.time, "monotonic", side_effect=lambda: clock[0]), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "before the concurrency-3 wave"):
                warmup.run("test", read, 89.0)             # waves end at 30, 90 s: wave 3 never starts
        clock[0] = 0.0
        with patch.object(warmup.time, "monotonic", side_effect=lambda: clock[0]), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "finished after"):
                warmup.run("test", read, 200.0)            # wave 4 starts at 180 s, ends at 300 s: late success fails
        with patch.object(warmup.time, "monotonic", return_value=100.0):
            self.assertAlmostEqual(warmup.request_timeout(100.01), 0.01, places=3)

    def test_serve_wrapper_kills_a_hung_helper(self) -> None:
        import os
        import subprocess
        import tempfile
        root = Path(__file__).parents[3]
        with tempfile.TemporaryDirectory() as fake:
            shim = Path(fake) / "python3"
            shim.write_text("#!/bin/sh\ntrap '' TERM\nwhile :; do sleep 1; done\n")    # ignores TERM: KILL must end it
            shim.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "WARMUP_HARD_S": "1", "SERVED": "x"}
            start = warmup.time.monotonic()
            done = subprocess.run(["bash", str(root / "offload/launch/warmup_serve.sh"), "8000", "x"], env=env,
                                  cwd=root, capture_output=True, timeout=30)
            self.assertNotEqual(done.returncode, 0)
            self.assertLess(warmup.time.monotonic() - start, 15)

if __name__ == "__main__":
    unittest.main()
