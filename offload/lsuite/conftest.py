"""Shared fixtures: the real pinned corpus (Controller / sparkhead only), a tiny synthetic corpus and suite,
an oracle replier and a fake server for the runner (no network)."""
from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import TypedDict
import urllib.error

import pytest

from offload.lsuite import corpus as corpus_module
from offload.lsuite.corpus import Corpus, sha256_file, sha256_text
from offload.lsuite.run_suite import SERVED, Get, Metrics, Post
from offload.lsuite.scoring import CitationKey
from offload.lsuite.tasks import SUITE, FigureKey, Materialized, NeedleKey, QuoteKey, RefusalKey, Suite

BENCH_DATA = Path(os.environ.get("LSUITE_BENCH_DATA", "/opt/dsv41/data/bench-data"))
OPINION_LICENCE = ("public domain (court opinion; government edicts doctrine); slip-opinion PDF as published by the "
                   "court, mirrored by CourtListener (Free Law Project)")
FILING_LICENCE = ("public court record (PACER via RECAP / Free Law Project). Judicial orders are public domain; party "
                  "filings are public records but may carry authors' copyright - used only for calibration statistics, "
                  "never redistributed")


@pytest.fixture(scope="session")
def real_corpus() -> Corpus:
    """The pinned production corpus. Skips (visibly) on a host without bench-data; never on Controller."""
    if not (BENCH_DATA / "opinions_cl.jsonl").is_file():
        pytest.skip(f"bench-data not on this host ({BENCH_DATA}); set LSUITE_BENCH_DATA")
    return Corpus.load(BENCH_DATA)


def record(doc_id: str, collection: str, body: str, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "doc_id": doc_id, "collection": collection, "group": "sc_family", "title": f"Title {doc_id}",
        "court": "Court of Appeals of South Carolina", "court_id": "scctapp", "jurisdiction": "SC",
        "date": "2024-01-02",
        "license": OPINION_LICENCE if collection == "opinions_cl" else FILING_LICENCE,
        "chars": len(body), "sha256": sha256_text(body), "text": body}
    base.update(extra)
    return base


def write_corpus(root: Path, opinions: list[dict[str, object]],
                 filings: list[dict[str, object]]) -> dict[str, str]:
    """Write a synthetic corpus into `root` and return its file pins."""
    root.mkdir(parents=True, exist_ok=True)
    pins: dict[str, str] = {}
    for name, rows in (("opinions_cl.jsonl", opinions), ("filings_recap.jsonl", filings)):
        path = root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pins[name] = sha256_file(path)
    return pins


def oracle_reply(key: object) -> str:
    """A reply that should pass the task: built from the answer key (test oracle; never model output)."""
    if isinstance(key, QuoteKey):
        return key.sentence
    if isinstance(key, FigureKey):
        return {"money": f"${key.value}", "date": key.value, "percent": f"{key.value}%"}[key.kind]
    if isinstance(key, CitationKey):
        court = f"{key.court} " if key.court else ""
        return "Case v. Case, " + ", ".join(" ".join(r) for r in key.reporters) + f" ({court}{key.year})"
    if isinstance(key, NeedleKey):
        return key.docket
    assert isinstance(key, RefusalKey)
    return "NOT IN RECORD"


# tools/data_boundary.py (DOC2, ADR-0013/0014) refuses JSON whose text-like keys carry non-empty strings.
BOUNDARY_TEXT_KEYS = frozenset({"text", "content", "completion", "response", "output_text", "generated", "generation",
                                "prompt", "messages", "reasoning_content", "answer"})
SECRET_MARKS = (b"hf_", b"ghp_", b"gho_", b"sk-ant-", b"AKIA", b"PRIVATE KEY-----")


# ---------------------------------------------------------------- synthetic suite

OPINION = ("The opening sentence is ordinary text. The family court awarded the wife alimony after a careful review "
           "of the record. The ending sentence closes the opinion.")
FILING = "Plaintiff seeks damages. The invoice for the repair work totaled $4,250 according to the vendor. End."
SENT = "The family court awarded the wife alimony after a careful review of the record."


def spec(body: str, piece: str, inline: bool) -> dict[str, object]:
    start = body.index(piece)
    out: dict[str, object] = {"start": start, "end": start + len(piece), "sha256": sha256_text(piece)}
    if inline:
        out["value"] = piece
    return out


def tasks_doc(op_sha: str, filing_sha: str) -> dict[str, object]:
    return {
        "suite": SUITE, "corpus": {}, "families": {"quote": 1, "figure": 1, "citation": 0, "needle": 0, "refusal": 1},
        "instruction": "RULE", "request": {"temperature": 0, "max_tokens": {
            "quote": 9, "figure": 8, "citation": 7, "needle": 6, "refusal": 5}},
        "tasks": [
            {"id": "quote-01", "family": "quote", "doc_id": "op-1", "doc_sha256": op_sha, "licence": "public-domain",
             "question": 'Quote the sentence with "{phrase}".',
             "slots": {"phrase": spec(OPINION, "awarded the wife alimony after", True)},
             "key": {"sentence": spec(OPINION, SENT, True)}},
            {"id": "figure-01", "family": "figure", "doc_id": "recap-1", "doc_sha256": filing_sha,
             "licence": "pinned-only", "question": 'Amount in the sentence with "{phrase}"?',
             "slots": {"phrase": spec(FILING, "for the repair work totaled", False)},
             "key": {"kind": "money", "figure": spec(FILING, "$4,250", False)}},
            {"id": "refusal-01", "family": "refusal", "doc_id": "op-1", "doc_sha256": op_sha,
             "licence": "public-domain", "question": "Who was the guardian?",
             "key": {"kind": "name", "absent": ["guardian"]}},
        ]}


# ---------------------------------------------------------------- fake server for the runner

MODELS: dict[str, object] = {"object": "list", "data": [
    {"id": SERVED, "root": "/srv/dsv41/legal-v2a", "max_model_len": 327680}]}


class MainIO(TypedDict):
    """The transport keyword arguments of run_suite.main, all served by one FakeServer."""

    post: Post
    get: Get
    metrics: Metrics


class FakeServer:
    """Answers each request from a reply function keyed by the request's user message."""

    def __init__(self, items: list[Materialized], reply: Callable[[Materialized], str]) -> None:
        self.by_message = {item.message: item for item in items}
        self.reply = reply
        self.bodies: list[dict[str, object]] = []
        self.fail: set[str] = set()
        self.reasoning: str | None = None
        self.busy_reads = 0                     # the first N /metrics reads report one other running request
        self.busy_after_post: int | None = None  # from this many sent requests on, another request stays running
        self.gauges = True                      # False: /metrics lacks the load gauges
        self.metric_reads = 0

    def io(self) -> MainIO:
        return {"post": self.post, "get": self.get, "metrics": self.metrics}

    def metrics(self, url: str, timeout: float) -> str:
        assert url.endswith("/metrics")
        self.metric_reads += 1
        late = self.busy_after_post is not None and len(self.bodies) >= self.busy_after_post
        busy = 1 if self.metric_reads <= self.busy_reads or late else 0
        if not self.gauges:
            return "# HELP vllm:num_preemptions_total preemptions\nvllm:num_preemptions_total 0.0\n"
        return ("# HELP vllm:num_requests_running running\n"
                f'vllm:num_requests_running{{model_name="m",engine="0"}} {busy}.0\n'
                f'vllm:num_requests_waiting{{model_name="m",engine="0"}} 0.0\n')

    def get(self, url: str, timeout: float) -> dict[str, object]:
        assert url.endswith("/v1/models")
        return MODELS

    def post(self, url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
        assert url.endswith("/v1/chat/completions")
        self.bodies.append(body)
        messages = body["messages"]
        assert isinstance(messages, list)
        item = self.by_message[messages[0]["content"]]
        if item.task_id in self.fail:
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)  # type: ignore[arg-type]
        return {"choices": [{"message": {"role": "assistant", "content": self.reply(item),
                                         "reasoning_content": self.reasoning},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(item.message) // 4, "completion_tokens": 7}}


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[Materialized]]:
    pins = write_corpus(tmp_path / "bench", [record("op-1", "opinions_cl", OPINION)],
                        [record("recap-1", "filings_recap", FILING)])
    monkeypatch.setattr(corpus_module, "CORPUS_FILES", pins)
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks_doc(sha256_text(OPINION), sha256_text(FILING))), encoding="utf-8")
    items = Suite.load(tasks_path).materialize(Corpus.load(tmp_path / "bench"))
    return tmp_path / "bench", tasks_path, items


def run_args(bench: Path, tasks: Path, out: Path, *extra: str) -> list[str]:
    """Runner argv for an --execute run of the synthetic suite against a fake URL."""
    return ["--bench-data", str(bench), "--tasks-file", str(tasks), "--arm", "X", "--url", "http://fake:8001",
            "--out", str(out), "--execute", *extra]
