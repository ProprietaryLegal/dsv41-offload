"""Corpus pins, record verification and licence classes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.lsuite.conftest import record, write_corpus
from offload.lsuite.corpus import CORPUS_FILES, Corpus, CorpusError, classify_licence, sha256_file


def test_synthetic_corpus_loads_with_licence_classes(tmp_path: Path) -> None:
    pins = write_corpus(tmp_path, [record("op-1", "opinions_cl", "Opinion body.")],
                        [record("recap-1", "filings_recap", "Filing body.")])
    corpus = Corpus.load(tmp_path, pins)
    assert [doc.doc_id for doc in corpus.docs] == ["op-1", "recap-1"]
    assert corpus.opinions()[0].redistributable
    assert not corpus.filings()[0].redistributable
    assert corpus.get("op-1", corpus.opinions()[0].sha256).body == "Opinion body."


def test_file_pin_mismatch_refuses(tmp_path: Path) -> None:
    pins = write_corpus(tmp_path, [record("op-1", "opinions_cl", "Opinion body.")], [])
    pins["opinions_cl.jsonl"] = "0" * 64
    with pytest.raises(CorpusError, match="not the pinned corpus"):
        Corpus.load(tmp_path, pins)


def test_missing_file_refuses(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="missing"):
        Corpus.load(tmp_path)


def test_record_text_must_match_its_sha256(tmp_path: Path) -> None:
    bad = record("op-1", "opinions_cl", "Opinion body.")
    bad["text"] = "Opinion body, edited."
    bad["chars"] = len("Opinion body, edited.")
    pins = write_corpus(tmp_path, [bad], [])
    with pytest.raises(CorpusError, match="does not hash"):
        Corpus.load(tmp_path, pins)


def test_record_chars_must_match(tmp_path: Path) -> None:
    bad = record("op-1", "opinions_cl", "Opinion body.", chars=3)
    pins = write_corpus(tmp_path, [bad], [])
    with pytest.raises(CorpusError, match="chars"):
        Corpus.load(tmp_path, pins)


def test_duplicate_doc_id_refuses(tmp_path: Path) -> None:
    pins = write_corpus(tmp_path, [record("op-1", "opinions_cl", "A."), record("op-1", "opinions_cl", "B.")], [])
    with pytest.raises(CorpusError, match="duplicate"):
        Corpus.load(tmp_path, pins)


def test_get_checks_doc_pin(tmp_path: Path) -> None:
    pins = write_corpus(tmp_path, [record("op-1", "opinions_cl", "A.")], [])
    corpus = Corpus.load(tmp_path, pins)
    with pytest.raises(CorpusError, match="not the pinned"):
        corpus.get("op-1", "f" * 64)
    with pytest.raises(CorpusError, match="not in the corpus"):
        corpus.get("op-2", "f" * 64)


def test_unknown_licence_fails_loud() -> None:
    with pytest.raises(CorpusError, match="unreviewed licence"):
        classify_licence("opinions_cl", "CC-BY 4.0")
    with pytest.raises(CorpusError, match="unreviewed licence"):
        classify_licence("filings_recap", "public domain (court opinion; government edicts doctrine)")
    with pytest.raises(CorpusError, match="unreviewed licence"):
        classify_licence("other", "public domain (court opinion; government edicts doctrine)")


def test_non_object_record_refuses(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "opinions_cl.jsonl").write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="not a JSON object"):
        Corpus.load(tmp_path, {"opinions_cl.jsonl": sha256_file(tmp_path / "opinions_cl.jsonl")})


def test_real_corpus_pins_and_licences(real_corpus: Corpus) -> None:
    assert set(CORPUS_FILES) == {"opinions_cl.jsonl", "filings_recap.jsonl"}
    assert len(real_corpus.opinions()) == 351
    assert len(real_corpus.filings()) == 61
    assert all(doc.redistributable for doc in real_corpus.opinions())
    assert not any(doc.redistributable for doc in real_corpus.filings())
