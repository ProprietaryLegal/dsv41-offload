"""Runner behaviour with an injected transport (no network) and one loopback HTTP round trip."""
from __future__ import annotations

import http.server
import json
from pathlib import Path
import threading

import pytest

from offload.lsuite import compare as K
from offload.lsuite import run_suite as R
from offload.lsuite.conftest import BOUNDARY_TEXT_KEYS, SENT, FakeServer, oracle_reply, run_args as args
from offload.lsuite.corpus import Corpus, sha256_text
from offload.lsuite.tasks import Materialized, Suite

def outputs(out: Path) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    return (json.loads((out / "run.json").read_text(encoding="utf-8")), rows,
            json.loads((out / "summary.json").read_text(encoding="utf-8")))


def test_oracle_run_passes_and_stores_no_reply_text(synthetic: tuple[Path, Path, list[Materialized]],
                                                    tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "out"), **fake.io()) == 0
    header, rows, summary = outputs(tmp_path / "out")
    assert [row["outcome"] for row in rows] == ["pass", "pass", "pass"]
    assert summary["complete"] is True and summary["pass"] == 3
    assert header["served"] == {"id": R.SERVED, "root": "/srv/dsv41/legal-v2a", "max_model_len": 327680}
    for path in (tmp_path / "out").iterdir():
        data = path.read_text(encoding="utf-8")
        assert SENT not in data and "NOT IN RECORD" not in data and "4,250" not in data, path.name
    for row in rows:
        assert not BOUNDARY_TEXT_KEYS.intersection(row)
        assert row["span_chars"] == len(oracle_reply(next(i for i in items if i.task_id == row["task"]).key))
        assert row["request_sha256"] == sha256_text(next(i for i in items if i.task_id == row["task"]).message)


def test_request_bodies_are_greedy_cold_and_background(synthetic: tuple[Path, Path, list[Materialized]],
                                                       tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: "NOT IN RECORD")
    assert R.main(args(bench, tasks, tmp_path / "out"), **fake.io()) == 0
    salts = set()
    for body, item in zip(fake.bodies, items, strict=True):
        assert body["temperature"] == 0 and body["max_tokens"] == item.max_tokens and body["priority"] == 10
        assert body["model"] == R.SERVED
        salts.add(body["cache_salt"])
    assert len(salts) == len(items)
    _, rows, _ = outputs(tmp_path / "out")
    assert [row["outcome"] for row in rows] == ["fail", "fail", "pass"]   # the marker passes only the probe
    assert [row["reason"] for row in rows] == ["refused", "refused", "marker"]
    fake2 = FakeServer(items, lambda item: "x")
    assert R.main(args(bench, tasks, tmp_path / "out2", "--priority", "0"), **fake2.io()) == 0
    assert all("priority" not in body for body in fake2.bodies)


def test_request_error_is_recorded_and_exits_1(synthetic: tuple[Path, Path, list[Materialized]],
                                               tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    fake.fail.add("figure-01")
    assert R.main(args(bench, tasks, tmp_path / "out"), **fake.io()) == 1
    _, rows, summary = outputs(tmp_path / "out")
    assert [(row["outcome"], row["reason"]) for row in rows] == [("pass", "verbatim"), ("error", "http-500"),
                                                                 ("pass", "marker")]
    assert summary["complete"] is False and summary["errors"] == 1


def test_refusals_before_any_request(synthetic: tuple[Path, Path, list[Materialized]], tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: "x")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "old.txt").write_text("x", encoding="utf-8")
    assert R.main(args(bench, tasks, busy), **fake.io()) == 2
    inside = Path(R.__file__).resolve().parent / "never-created"
    assert R.main(args(bench, tasks, inside), **fake.io()) == 2
    assert not inside.exists()
    assert R.main(args(bench, tasks, tmp_path / "o3", "--model", "other"), **fake.io()) == 2
    assert R.main(args(bench, tasks, tmp_path / "o4", "--tasks", "quote-99"), **fake.io()) == 2
    no_out = [a for a in args(bench, tasks, tmp_path / "o5") if a not in ("--out", str(tmp_path / "o5"))]
    assert R.main(no_out, **fake.io()) == 2
    assert fake.bodies == []


def test_subset_run_is_marked_partial(synthetic: tuple[Path, Path, list[Materialized]], tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "out", "--tasks", "quote-01"), **fake.io()) == 0
    header, rows, summary = outputs(tmp_path / "out")
    assert len(rows) == 1 and summary["complete"] is False and header["planned"] == 3


def test_dry_run_sends_nothing(synthetic: tuple[Path, Path, list[Materialized]], capsys: pytest.CaptureFixture[str]
                               ) -> None:
    bench, tasks, _ = synthetic

    def refuse(*_: object) -> dict[str, object]:
        raise AssertionError("dry run must not send")

    assert R.main(["--bench-data", str(bench), "--tasks-file", str(tasks), "--arm", "X"], post=refuse, get=refuse) == 0
    assert "ESTIMATED" in capsys.readouterr().out


def test_loopback_http_round_trip(synthetic: tuple[Path, Path, list[Materialized]], tmp_path: Path) -> None:
    """The real urllib path against a local fake server (127.0.0.1 only)."""
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/metrics":
                data = fake.metrics(self.path, 1.0).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._send(fake.get(self.path, 1.0))

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(fake.post(self.path, body, 1.0))

        def log_message(self, *_: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        argv = args(bench, tasks, tmp_path / "out")
        argv[argv.index("--url") + 1] = url
        assert R.main(argv) == 0
    finally:
        server.shutdown()
    _, rows, _ = outputs(tmp_path / "out")
    assert [row["outcome"] for row in rows] == ["pass", "pass", "pass"]


def test_real_suite_oracle_and_refuser(real_corpus: Corpus, tmp_path: Path) -> None:
    items = Suite.load().materialize(real_corpus)
    bench = str(real_corpus.root)
    oracle = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(["--bench-data", bench, "--arm", "C", "--url", "http://fake", "--out", str(tmp_path / "c"),
                   "--execute"], **oracle.io()) == 0
    _, rows, summary = outputs(tmp_path / "c")
    assert summary["pass"] == 46 and summary["complete"] is True
    for body in oracle.bodies:
        assert (body["temperature"], body["top_p"], body["seed"]) == (0, 1, 0)
        assert (body["repetition_penalty"], body["presence_penalty"], body["frequency_penalty"]) == (1.0, 0.0, 0.0)
        assert body["chat_template_kwargs"] == {"thinking": False} and body["priority"] == 10
    refuser = FakeServer(items, lambda item: "NOT IN RECORD")
    assert R.main(["--bench-data", bench, "--arm", "X", "--url", "http://fake", "--out", str(tmp_path / "x"),
                   "--execute"], **refuser.io()) == 0
    _, rows, summary = outputs(tmp_path / "x")
    assert summary["pass"] == 4 and {row["task"] for row in rows if row["outcome"] == "pass"} == {
        "refusal-01", "refusal-02", "refusal-03", "refusal-04"}


def test_rows_fingerprint_sent_settings_and_code(synthetic: tuple[Path, Path, list[Materialized]],
                                                 tmp_path: Path) -> None:
    """S101 FIX 6: two runs with the same settings share every body_sha256 (the salt counts as its per-task policy)
    and record the code_sha256 of the sources that scored them."""
    bench, tasks, items = synthetic
    for name in ("A", "B"):
        fake = FakeServer(items, lambda item: oracle_reply(item.key))
        assert R.main(args(bench, tasks, tmp_path / name), **fake.io()) == 0
    header_a, rows_a, _ = outputs(tmp_path / "A")
    header_b, rows_b, _ = outputs(tmp_path / "B")
    assert header_a["run_id"] != header_b["run_id"]
    assert header_a["code_sha256"] == header_b["code_sha256"] == R.code_sha256()
    assert [row["body_sha256"] for row in rows_a] == [row["body_sha256"] for row in rows_b]
    assert len({row["body_sha256"] for row in rows_a}) == len(rows_a)      # max_tokens differ by family here


def test_body_fingerprint_counts_salt_policy_and_every_setting() -> None:
    body: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 9,
                               "cache_salt": "lsuite-r1-quote-01", "temperature": 0, "priority": 10}
    same = R.body_fingerprint(body | {"cache_salt": "lsuite-r2-quote-01", "model": "other"}, "r2", "quote-01")
    assert R.body_fingerprint(body, "r1", "quote-01") == same
    assert R.body_fingerprint(body | {"cache_salt": "constant"}, "r1", "quote-01") != same
    assert R.body_fingerprint(body | {"priority": 0}, "r1", "quote-01") != same
    assert R.body_fingerprint(body | {"repetition_penalty": 1.3}, "r1", "quote-01") != same
    assert R.body_fingerprint({k: v for k, v in body.items() if k != "priority"}, "r1", "quote-01") != same


def test_busy_arm_gets_nothing(synthetic: tuple[Path, Path, list[Materialized]], tmp_path: Path) -> None:
    """S101 FIX 7 repro: a busy arm used to receive every task (0 /metrics reads); now nothing is sent."""
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    fake.busy_reads = 10**6
    assert R.main(args(bench, tasks, tmp_path / "out", "--idle-wait", "0"), **fake.io()) == 1
    _, rows, summary = outputs(tmp_path / "out")
    assert fake.bodies == [] and rows == [] and summary["stopped_busy"] is True and summary["complete"] is False
    assert fake.metric_reads >= 1


def test_runner_waits_for_an_idle_arm(synthetic: tuple[Path, Path, list[Materialized]], tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    fake.busy_reads = 3
    assert R.main(args(bench, tasks, tmp_path / "out", "--idle-poll", "0", "--idle-wait", "10"), **fake.io()) == 0
    _, rows, summary = outputs(tmp_path / "out")
    assert len(fake.bodies) == 3 and summary["complete"] is True
    assert all(row["others_after"] == 0 for row in rows)


def test_concurrent_request_marks_the_row_and_stops(synthetic: tuple[Path, Path, list[Materialized]],
                                                    tmp_path: Path) -> None:
    """S101 FIX 7 repro: another request arriving during task 2 used to go unnoticed."""
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    fake.busy_after_post = 2
    assert R.main(args(bench, tasks, tmp_path / "out", "--idle-wait", "0"), **fake.io()) == 1
    _, rows, summary = outputs(tmp_path / "out")
    assert [(row["outcome"], row["reason"], row["others_after"]) for row in rows] == [
        ("pass", "verbatim", 0), ("error", "concurrent", 1)]
    assert len(fake.bodies) == 2 and summary["stopped_busy"] is True
    control = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "C"), **control.io()) == 0
    assert K.main([str(tmp_path / "C"), str(tmp_path / "out")]) == 2


def test_metrics_without_load_gauges_fails_loud(synthetic: tuple[Path, Path, list[Materialized]],
                                                tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    fake = FakeServer(items, lambda item: oracle_reply(item.key))
    fake.gauges = False
    assert R.main(args(bench, tasks, tmp_path / "out"), **fake.io()) == 2
    assert fake.bodies == []


def test_load_on_sums_labelled_gauges() -> None:
    text = ("# HELP vllm:num_requests_running x\n# TYPE vllm:num_requests_running gauge\n"
            'vllm:num_requests_running{engine="0",model_name="m"} 1.0\n'
            'vllm:num_requests_running{engine="1",model_name="m"} 2.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="m"} 3.0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",reason="x"} 9.0\n')
    assert R.load_on(lambda url, timeout: text, "http://arm", 1.0) == 6
