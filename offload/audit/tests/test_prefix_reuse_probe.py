"""AUD1 item 2: prefix_reuse_probe.py against a loopback fake vLLM server (CPU only, no real server).

  python3 -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pytest offload/audit/tests/test_prefix_reuse_probe.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prefix_reuse_probe as probe

GENERATED = "FAKE-GENERATED-TEXT-7f3a"
BLOCK = 384


@dataclass
class FakeState:
    cache: bool = True
    busy: bool = False
    extra_queries: int = 0  # simulated other traffic between the probe's reads
    fail_first_post: bool = False
    hit_offset: int = 0  # metrics hits minus usage cached (the replay-window gap)
    paths: list[tuple[str, str]] = field(default_factory=list)
    seen: dict[str, list[str]] = field(default_factory=dict)
    queries: float = 0.0
    hits: float = 0.0
    prompt: float = 0.0

    def tokens(self, content: str) -> int:
        return len(content) // 4 + 6  # fake tokenizer + chat framing

    def cached(self, salt: str, content: str) -> int:
        if not self.cache:
            return 0
        best = 0
        for old in self.seen.get(salt, []):
            n = 0
            for x, y in zip(old, content):
                if x != y:
                    break
                n += 1
            best = max(best, n)
        return (best // 4 // BLOCK) * BLOCK


def make_handler(state: FakeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:
            state.paths.append(("GET", self.path))
            if self.path != "/metrics":
                self.send_error(404)
                return
            state.queries += state.extra_queries
            busy = 1 if state.busy else 0
            body = "\n".join([
                "# HELP vllm:prefix_cache_queries x",
                "# TYPE vllm:prefix_cache_queries counter",
                f'vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {state.queries}',
                f'vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {state.hits}',
                f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {state.prompt}',
                f'vllm:num_requests_running{{engine="0",model_name="m"}} {busy}',
                'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0',
                'vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 1.5',
                'vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 3.0',
            ]) + "\n"
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            state.paths.append(("POST", self.path))
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            if state.fail_first_post and sum(1 for m, _ in state.paths if m == "POST") == 1:
                self.send_error(500)
                return
            content = body["messages"][0]["content"]
            salt = body.get("cache_salt", "")
            pt = state.tokens(content)
            cached = min(state.cached(salt, content), pt - 1)
            state.seen.setdefault(salt, []).append(content)
            state.queries += pt
            state.hits += cached + (state.hit_offset if cached else 0)
            state.prompt += pt
            time.sleep((pt - cached) / 400_000)  # prefill time ~ uncached tokens
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": GENERATED}}]},
                {"choices": [{"index": 0, "delta": {"content": " more"}}]},
                {"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": 2,
                                          "prompt_tokens_details": {"cached_tokens": cached}}},
            ]
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


@pytest.fixture
def server() -> Iterator[tuple[str, FakeState]]:
    state = FakeState()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    finally:
        httpd.shutdown()
        httpd.server_close()


DOC_MARK = "UNIQUE-DOCUMENT-MARKER-91c2"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w") as f:
        for i in range(40):
            text = f"{DOC_MARK} record {i}. " + ("The court held that the statute applies. " * 60)
            f.write(json.dumps({"title": f"Case {i}", "court": "Ct", "date": "2025-01-01", "text": text}) + "\n")
    return path


def _run(argv: list[str]) -> int:
    return probe.main(argv)


def _args(url: str, corpus: Path, out: Path) -> list[str]:
    return ["--execute", "--base-url", url, "--out", str(out), "--data", str(corpus), "--no-data-sha-check",
            "--target-tokens", "20000"]


def _no_text(out: Path) -> None:
    text = out.read_text()
    assert DOC_MARK not in text and GENERATED not in text and "statute" not in text
    for q in probe.QUESTIONS:
        assert q not in text


def test_dry_run_sends_nothing(server: tuple[str, FakeState], corpus: Path, tmp_path: Path,
                               capsys: pytest.CaptureFixture[str]) -> None:
    _url, state = server
    assert _run(["--data", str(corpus), "--no-data-sha-check", "--target-tokens", "20000"]) == 0
    printed = capsys.readouterr().out
    assert state.paths == []
    plan = json.loads(printed)
    assert plan["executed"] is False and plan["plan"]["requests_to_send"] == 2
    assert DOC_MARK not in printed


def test_reuse_works_two_posts_metrics_only(server: tuple[str, FakeState], corpus: Path, tmp_path: Path) -> None:
    url, state = server
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 0
    posts = [p for m, p in state.paths if m == "POST"]
    gets = [p for m, p in state.paths if m == "GET"]
    assert posts == ["/v1/chat/completions"] * 2
    assert gets == ["/metrics"] * 4
    res = json.loads(out.read_text())
    assert res["result"]["verdict"] == "RESIDENT_REUSE_WORKS", res["result"]
    a, b = res["requests"]
    assert a["cached_tokens"] == 0 and b["cached_tokens"] > 0 and b["cached_tokens"] % BLOCK == 0
    assert res["result"]["second_cached_aligned_to_384"] is True
    assert res["metrics_delta"][1]["vllm:prefix_cache_hits"] == b["cached_tokens"]
    assert len(a["generated_sha256"]) == 64
    _no_text(out)


def test_replay_gap_within_window_is_not_contamination(server: tuple[str, FakeState], corpus: Path,
                                                       tmp_path: Path) -> None:
    url, state = server
    state.hit_offset = 128
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 0
    res = json.loads(out.read_text())["result"]
    assert res["verdict"] == "RESIDENT_REUSE_WORKS" and res["metrics_hits_minus_usage_cached"][1] == 128


def test_no_cache_means_reuse_fails(server: tuple[str, FakeState], corpus: Path, tmp_path: Path) -> None:
    url, state = server
    state.cache = False
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 0
    assert json.loads(out.read_text())["result"]["verdict"] == "RESIDENT_REUSE_FAILS"
    _no_text(out)


def test_busy_server_refused_before_any_post(server: tuple[str, FakeState], corpus: Path, tmp_path: Path) -> None:
    url, state = server
    state.busy = True
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 2
    assert [m for m, _ in state.paths] == ["GET"]
    assert not out.exists()


def test_other_traffic_is_contaminated(server: tuple[str, FakeState], corpus: Path, tmp_path: Path) -> None:
    url, state = server
    state.extra_queries = 1000
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 0
    assert json.loads(out.read_text())["result"]["verdict"] == "CONTAMINATED"


def test_first_request_failure_stops_before_second(server: tuple[str, FakeState], corpus: Path,
                                                   tmp_path: Path) -> None:
    url, state = server
    state.fail_first_post = True
    out = tmp_path / "r.json"
    assert _run(_args(url, corpus, out)) == 2
    assert sum(1 for m, _ in state.paths if m == "POST") == 1
    res = json.loads(out.read_text())
    assert res["result"]["verdict"] == "INCOMPLETE" and "error" in res


def test_corpus_pin_enforced(corpus: Path, tmp_path: Path) -> None:
    assert _run(["--data", str(corpus)]) == 2


def test_existing_out_refused(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "exists.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        _run(["--execute", "--base-url", "http://127.0.0.1:9", "--out", str(out), "--data", str(corpus)])


def test_execute_requires_url_and_out(corpus: Path) -> None:
    with pytest.raises(SystemExit):
        _run(["--execute", "--data", str(corpus)])


def test_parse_metrics_total_suffix_labels_and_sum() -> None:
    text = ('# TYPE vllm:prefix_cache_hits counter\n'
            'vllm:prefix_cache_hits_total{engine="0"} 10\n'
            'vllm:prefix_cache_hits_total{engine="1"} 5\n'
            'vllm:num_requests_running{engine="0"} 0\n'
            'vllm:unrelated_total 7\n')
    m = probe.parse_metrics(text)
    assert m["vllm:prefix_cache_hits"] == 15 and m["vllm:num_requests_running"] == 0
    assert "vllm:unrelated" not in m


def test_pinned_public_corpus_builds_about_150k_tokens() -> None:
    if not probe.DEFAULT_DATA.is_file():
        pytest.skip("public bench corpus not on this host")
    doc = probe.build_document(probe.DEFAULT_DATA, probe.TARGET_TOKENS)
    assert 400_000 < len(doc) <= int(probe.TARGET_TOKENS * probe.CHARS_PER_TOKEN)
    assert probe.build_document(probe.DEFAULT_DATA, probe.TARGET_TOKENS) == doc  # deterministic
