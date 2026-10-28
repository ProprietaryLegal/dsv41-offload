"""Materialization refuses every drift between the task file and the corpus (synthetic corpus, hand-built tasks)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from offload.lsuite.conftest import FILING, OPINION, SENT, record, tasks_doc, write_corpus
from offload.lsuite.corpus import Corpus, sha256_text
from offload.lsuite.tasks import (
    SPECIAL_OPEN, FigureKey, QuoteKey, RefusalKey, Suite, TaskError, insert_needle, sanitize)

@pytest.fixture
def setup(tmp_path: Path) -> tuple[Corpus, dict[str, object], Path]:
    pins = write_corpus(tmp_path / "bench", [record("op-1", "opinions_cl", OPINION)],
                        [record("recap-1", "filings_recap", FILING)])
    corpus = Corpus.load(tmp_path / "bench", pins)
    doc = tasks_doc(sha256_text(OPINION), sha256_text(FILING))
    return corpus, doc, tmp_path / "tasks.json"


def load(path: Path, doc: dict[str, object]) -> Suite:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return Suite.load(path)


def test_happy_path(setup: tuple[Corpus, dict[str, object], Path]) -> None:
    corpus, doc, path = setup
    items = load(path, doc).materialize(corpus)
    assert [m.task_id for m in items] == ["quote-01", "figure-01", "refusal-01"]
    assert items[0].key == QuoteKey(SENT)
    assert 'Quote the sentence with "awarded the wife alimony after".' in items[0].message
    assert items[1].key == FigureKey("money", "4250")
    assert '"for the repair work totaled"' in items[1].message
    assert items[2].key == RefusalKey("name")
    assert [m.max_tokens for m in items] == [9, 8, 5]
    assert items[0].message.startswith("RULE\n\nBEGIN MATERIALS\nDocument: Title op-1\n")


def mutate(doc: dict[str, object], index: int, path: list[str], value: object) -> dict[str, object]:
    out = copy.deepcopy(doc)
    tasks = out["tasks"]
    assert isinstance(tasks, list)
    node = tasks[index]
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    return out


@pytest.mark.parametrize(("index", "path", "value", "message"), [
    (0, ["key", "sentence", "sha256"], "0" * 64, "span sha256 mismatch"),
    (0, ["key", "sentence", "value"], "Edited.", "inline value differs"),
    (0, ["key", "sentence", "end"], 10_000, "outside"),
    (0, ["doc_sha256"], "f" * 64, "not the pinned"),
    (0, ["licence"], "pinned-only", "licence"),
    (0, ["question"], "No slot here.", "template slots"),
    (1, ["slots", "phrase", "value"], "for the repair work totaled", "pinned-only source"),
    (1, ["key", "kind"], "weight", "figure kind"),
    (2, ["key", "absent"], ["alimony"], "occurs in"),
    (2, ["family"], "essay", "task count"),
])
def test_drift_is_refused(setup: tuple[Corpus, dict[str, object], Path], index: int, path: list[str],
                          value: object, message: str) -> None:
    corpus, doc, tasks_path = setup
    with pytest.raises(TaskError, match=message):
        load(tasks_path, mutate(doc, index, path, value)).materialize(corpus)


def test_header_counts_and_ids_are_checked(setup: tuple[Corpus, dict[str, object], Path]) -> None:
    _, doc, path = setup
    wrong = copy.deepcopy(doc)
    wrong["families"] = {"quote": 2, "figure": 1, "citation": 0, "needle": 0, "refusal": 1}
    with pytest.raises(TaskError, match="task count"):
        load(path, wrong)
    dup = mutate(doc, 1, ["id"], "quote-01")
    with pytest.raises(TaskError, match="duplicate"):
        load(path, dup)
    other = copy.deepcopy(doc)
    other["suite"] = "lsuite-v0"
    with pytest.raises(TaskError, match="suite"):
        load(path, other)


def test_sanitize_and_needle_insertion() -> None:
    assert SPECIAL_OPEN not in sanitize(f"a {SPECIAL_OPEN}x")
    assert insert_needle("one\n\ntwo", 5, "NOTE") == "one\n\nNOTE\n\ntwo"
