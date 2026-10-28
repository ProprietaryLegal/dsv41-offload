"""Builder rules on synthetic text, and the committed task file against the real pinned corpus."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from offload.lsuite import build as B
from offload.lsuite.conftest import BOUNDARY_TEXT_KEYS, SECRET_MARKS, oracle_reply, record, write_corpus
from offload.lsuite.corpus import Corpus, Doc
from offload.lsuite.scoring import CitationKey, normalize
from offload.lsuite.tasks import (
    TASKS_FILE, FigureKey, Materialized, NeedleKey, QuoteKey, RefusalKey, Suite, render_doc)


def doc(body: str, doc_id: str = "op-x", jurisdiction: str = "SC") -> Doc:
    return Doc(doc_id, "opinions_cl", "sc_family", "T", "Court", "scctapp", jurisdiction, "2024-01-02",
               "public-domain", "0" * 64, body)


# ---------------------------------------------------------------- sentence and phrase rules (synthetic)


PARA = ("The first sentence of this paragraph ends cleanly here. The family court considered the testimony of each "
        "witness and found the mother more credible on every disputed point. Then the court ruled on fees.")


def test_sentences_take_clean_prose() -> None:
    found = [normalize(PARA[s.start:s.end]) for s in B.sentences(doc(PARA), 60, 300)]
    assert found == [("The family court considered the testimony of each witness and found the mother more credible "
                      "on every disputed point.")]


@pytest.mark.parametrize("middle", [
    "The family court considered the testi-\nmony of each witness and found the mother more credible on every point.",
    "The family court considered the testimony of each witness3 and found the mother more credible on every point.",
    "The family court considered the testimony of Mr. Smith and found the mother more credible on every point.",
    'The family court considered the "testimony" of each witness and found the mother more credible on every point.',
    "The family court considered the testimony of each witness and\n\nfound the mother more credible on every point.",
    "The family court considered the testimony of each witness and found the mother more\n12\ncredible on points.",
    "FACTS\nThe family court considered the testimony of each witness and found the mother more credible on all.",
])
def test_sentences_reject_layout_and_abbreviations(middle: str) -> None:
    body = f"The first sentence of this paragraph ends cleanly here.\n{middle} Then the court ruled on fees."
    assert [normalize(body[s.start:s.end]) for s in B.sentences(doc(body), 60, 300)] == []


def test_phrase_is_unique_literal_and_avoids_the_figure() -> None:
    body = ("Opening words of the opinion appear here. The court awarded the wife $4,000 in fees after a long "
            "hearing on the merits of the case. Closing words follow the award.")
    target = doc(body)
    sentence = next(iter(B.sentences(target, 60, 300)))
    fig = (body.index("$4,000"), body.index("$4,000") + len("$4,000"))
    found = B.phrase(target, sentence, normalize(render_doc(target)), avoid=fig)
    assert found is not None
    words = body[found[0]:found[1]]
    assert len(words.split()) == B.PHRASE_WORDS and "$" not in words and not re.search(r"\d", words)
    assert normalize(render_doc(target)).count(words) == 1


def test_phrase_none_when_every_window_repeats() -> None:
    sentence_text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
    body = f"Lead words are here. {sentence_text} Lead words are here. {sentence_text} Done now."
    target = doc(body)
    sentences = list(B.sentences(target, 40, 300))
    assert sentences and all(B.phrase(target, s, normalize(render_doc(target))) is None for s in sentences)


# ---------------------------------------------------------------- citation rules (synthetic)


def test_citations_strip_signals_and_require_unique_cite() -> None:
    body = ("We review de novo. See Jenkins v. Jenkins, 345 S.C. 88, 95, 545 S.E.2d 531, 535 (Ct. App. 2001). "
            "Accord Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174 (1997). Later, in Belton v. Belton, 325 S.C. 999, "
            "481 S.E.2d 999 (1997), the court said more. The ruling in S.C. Dep't of Soc. Servs. v. Hyatt, 277 S.C. "
            "152, 283 S.E.2d 445 (1981) is old.")
    found = {name: key for _, (name, key, _) in B.citations_in(doc(body))}
    assert set(found) == {"Jenkins v. Jenkins"}          # Belton has two cites; "Soc. Servs. v. Hyatt" is partial
    assert found["Jenkins v. Jenkins"] == {"reporters": [["345", "S.C.", "88"], ["545", "S.E.2d", "531"]],
                                           "year": "2001", "court": "Ct. App.", "sc_style": True}
    start, end = next(cite for _, (_, _, cite) in B.citations_in(doc(body)))
    assert body[start:end].startswith("Jenkins v. Jenkins, 345 S.C. 88")


def test_general_citation_court_parenthetical() -> None:
    body = "We agree. Custer v. Pan Am. Life Ins. Co., 12 F.3d 410, 416 (4th Cir. 1993). The end."
    [(_, (name, key, _))] = B.citations_in(doc(body, jurisdiction="US-federal"))
    assert name == "Custer v. Pan Am. Life Ins. Co."
    assert key == {"reporters": [["12", "F.3d", "410"]], "year": "1993", "court": "4th Cir.", "sc_style": False}


# ---------------------------------------------------------------- the committed file (real corpus)


@pytest.fixture(scope="module")
def suite() -> Suite:
    return Suite.load()


@pytest.fixture(scope="module")
def materialized(suite: Suite, real_corpus: Corpus) -> list[Materialized]:
    return suite.materialize(real_corpus)


@pytest.fixture(scope="module")
def real_name_screen() -> B.NameScreen:
    """The production client-name screen: folder names read into memory only (its repr never shows them)."""
    if not (B.CLIENT_FILES.is_dir() and B.MATTER_EXCLUSIONS.is_file()):
        pytest.skip("client-name screen sources are not on this host (Controller only)")
    return B.load_name_screen()


def test_committed_task_file_is_a_fresh_build(real_corpus: Corpus, real_name_screen: B.NameScreen) -> None:
    assert B.dumps(B.build(real_corpus, real_name_screen)) == TASKS_FILE.read_bytes()


# ---------------------------------------------------------------- client-name screen (synthetic needles only)


@pytest.mark.parametrize(("entry", "pairs"), [
    ("Doe Jane", {("doe", "jane")}),
    ("Doe, Jane", {("doe", "jane")}),
    ("Doe Jane (DSS)", {("doe", "jane")}),
    ("Doe Jane Jr", {("doe", "jane")}),
    ("Doe-Smith Jane", {("doe-smith", "jane")}),
    ("Doe Jane Quinn", {("doe", "jane"), ("jane", "quinn"), ("doe", "quinn")}),
    ("Doe Jane - Roe Rick", {("doe", "jane"), ("roe", "rick")}),
    ("Doe Jane & Roe Rick", {("doe", "jane"), ("roe", "rick")}),
    ("Doe Jane Q", {("doe", "jane")}),
    ("Doe Jane|Doe", {("doe", "jane")}),
    ("!Closed", set()),
    ("Doe", set()),
])
def test_name_pairs(entry: str, pairs: set[tuple[str, str]]) -> None:
    assert B.name_pairs(entry) == pairs


@pytest.mark.parametrize(("text", "hit"), [
    ("Counsel: Jane Doe, of Doe Law LLC, for Appellant.", True),
    ("JANE DOE, Respondent.", True),
    ("Doe, Jane, Appellant.", True),
    ("Doe Jane appeared.", True),
    ("Jane Q. Doe testified.", True),
    ("Jane\nDoe testified.", True),
    ("John Doe testified.", False),
    ("Jane Roe testified.", False),
    ("Janet Doe testified.", False),
    ("Doe testified; Jane did not.", False),
])
def test_name_screen_matches_full_name_in_either_order(text: str, hit: bool) -> None:
    assert B.NameScreen(frozenset({("doe", "jane")})).matches(text) is hit


def test_name_screen_never_shows_its_needles() -> None:
    screen = B.NameScreen(frozenset({("zyxwvut", "qponml")}))
    assert "zyxwvut" not in repr(screen) and "qponml" not in str(screen)


def test_load_name_screen_reads_open_closed_and_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "client folders"
    (root / "!Closed" / "Roe Rick").mkdir(parents=True)
    (root / "Doe Jane").mkdir()
    (root / "fee schedule.txt").write_text("not a client folder", encoding="utf-8")
    exclusions = tmp_path / "exclusions.txt"
    exclusions.write_text("# Comment Words Here\nPoe Edgar|Poe\n", encoding="utf-8")
    screen = B.load_name_screen(root, exclusions)
    assert screen.pairs == {("doe", "jane"), ("roe", "rick"), ("poe", "edgar")}
    with pytest.raises(B.BuildError, match="is missing"):
        B.load_name_screen(root, tmp_path / "absent.txt")
    with pytest.raises(B.BuildError, match="not a directory"):
        B.load_name_screen(tmp_path / "absent", exclusions)
    (root / "!Closed" / "Roe Rick").rmdir()
    (root / "!Closed").rmdir()
    with pytest.raises(B.BuildError, match="not a directory"):
        B.load_name_screen(root, exclusions)


def test_build_cli_refuses_without_screen_sources(tmp_path: Path) -> None:
    assert B.main(["--bench-data", str(tmp_path), "--client-files", str(tmp_path / "none"),
                   "--matter-exclusions", str(tmp_path / "none.txt")]) == 2


def test_build_moves_a_task_off_a_screened_document(real_corpus: Corpus, suite: Suite) -> None:
    """A synthetic needle that occurs only in quote-01's document forces the build onto another document, and no
    selected input (single-document task or haystack) matches the screen."""
    quote01 = next(t for t in suite.tasks if t["id"] == "quote-01")
    target = real_corpus.get(str(quote01["doc_id"]), str(quote01["doc_sha256"]))
    others = [B._word_grams(render_doc(doc)) for doc in real_corpus.docs if doc.doc_id != target.doc_id]
    needle = next(pair for pair in sorted(B._word_grams(render_doc(target)))
                  if pair[0].isalpha() and pair[1].isalpha() and len(pair[0]) > 3 and len(pair[1]) > 3
                  and not any(pair in grams or pair[::-1] in grams for grams in others))
    screen = B.NameScreen(frozenset({needle}))
    rebuilt = B.build(real_corpus, screen)
    tasks = rebuilt["tasks"]
    assert isinstance(tasks, list)
    used = {task["doc_id"] for task in tasks if "doc_id" in task}
    hay = {pair[0] for pair in tasks[-1]["haystack"]["docs"]}
    assert target.doc_id not in used | hay
    assert next(t for t in tasks if t["id"] == "quote-01")["doc_id"] != target.doc_id
    assert not any(screen.matches(render_doc(doc)) for doc in real_corpus.docs if doc.doc_id in used | hay)


def test_family_counts_match_the_spec(suite: Suite) -> None:
    counts = {family: sum(1 for t in suite.tasks if t["family"] == family)
              for family in ("quote", "figure", "citation", "needle", "refusal")}
    assert counts == {"quote": 12, "figure": 12, "citation": 12, "needle": 6, "refusal": 4}
    assert sum(1 for t in suite.tasks if t.get("licence") == "pinned-only") == 3


def test_request_pins_temperature_seed_thinking_and_penalties(suite: Suite) -> None:
    """S101 FIX 5: an omitted penalty takes the served model directory's generation_config default."""
    assert suite.request == {"temperature": 0, "top_p": 1, "seed": 0, "repetition_penalty": 1.0,
                             "presence_penalty": 0.0, "frequency_penalty": 0.0,
                             "chat_template_kwargs": {"thinking": False}}


def test_task_file_respects_the_data_boundary() -> None:
    data = TASKS_FILE.read_bytes()
    assert len(data) < 5 * 1024 * 1024
    assert not any(mark in data for mark in SECRET_MARKS)
    stack: list[object] = [json.loads(data)]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            assert not BOUNDARY_TEXT_KEYS.intersection(k.lower() for k in value), sorted(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def test_pinned_only_tasks_store_no_source_text(suite: Suite, real_corpus: Corpus) -> None:
    for task in suite.tasks:
        if task.get("licence") != "pinned-only":
            continue
        body = normalize(real_corpus.get(str(task["doc_id"]), str(task["doc_sha256"])).body)
        strings: list[str] = []
        stack: list[object] = [task]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                assert "value" not in value and "canonical" not in value, task["id"]
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                strings.append(value)
        for value in strings:
            words = normalize(value).split()
            for i in range(max(0, len(words) - 3)):
                assert " ".join(words[i:i + 4]) not in body, (task["id"], "source words stored")


def test_single_document_tasks_use_distinct_documents(suite: Suite) -> None:
    ids = [t["doc_id"] for t in suite.tasks if t["family"] != "needle"]
    assert len(ids) == len(set(ids)) == 40


def test_every_task_materializes_and_keys_score_themselves(materialized: list[Materialized], suite: Suite) -> None:
    assert len(materialized) == 46
    for item in materialized:
        assert item.message.startswith(suite.instruction)
        key = item.key
        right = oracle_reply(key)
        if isinstance(key, QuoteKey):
            wrong = key.sentence[:-1] + "!"
        elif isinstance(key, FigureKey):
            wrong = {"money": "$1", "date": "1900-01-01", "percent": "0.5%"}[key.kind]
        elif isinstance(key, CitationKey):
            wrong = right.replace(key.reporters[0][2], key.reporters[0][2] + "1", 1)
        elif isinstance(key, NeedleKey):
            wrong = key.docket[:-1] + ("0" if key.docket[-1] != "0" else "1")
        else:
            assert isinstance(key, RefusalKey)
            wrong = {"money": "$500", "date": "May 1, 2001", "percent": "50%", "name": "Jane Doe"}[key.kind]
        assert item.score(right).passed, (item.task_id, item.score(right))
        assert not item.score(wrong).passed, item.task_id
        assert not item.score("NOT IN RECORD").passed or isinstance(key, RefusalKey), item.task_id


def test_quote_and_figure_targets_are_unambiguous(materialized: list[Materialized], suite: Suite) -> None:
    for item, task in zip(materialized, suite.tasks, strict=True):
        materials = normalize(item.message.split("BEGIN MATERIALS\n", 1)[1].rsplit("\nEND MATERIALS", 1)[0])
        phrase = re.search(r'phrase "([^"]+)"', item.message.rsplit("Question: ", 1)[1])
        if isinstance(item.key, QuoteKey):
            assert phrase is not None and materials.count(item.key.sentence) == 1
            assert materials.count(phrase.group(1)) == 1 and phrase.group(1) in item.key.sentence
        elif isinstance(item.key, FigureKey):
            assert phrase is not None and materials.count(phrase.group(1)) == 1
            assert B.figure_values(item.key.kind, materials).count(item.key.value) == 1, task["id"]


def test_needles_plant_one_synthetic_docket_at_registered_depths(materialized: list[Materialized]) -> None:
    needles = [m for m in materialized if isinstance(m.key, NeedleKey)]
    assert len(needles) == 6
    dockets = set()
    for item, depth in zip(needles, B.NEEDLE_DEPTHS, strict=True):
        assert isinstance(item.key, NeedleKey)
        dockets.add(item.key.docket)
        assert item.message.count(B.NEEDLE_PHRASE) == 2         # the planted note and the question
        assert item.message.count(item.key.docket) == 1
        position = item.message.index(item.key.docket) / len(item.message)
        assert abs(position - depth) < 0.01
        assert B.NEEDLE_CHARS - 2_000 < len(item.message) < B.NEEDLE_CHARS + 2_000
    assert len(dockets) == 6


def test_refusal_documents_lack_the_asked_fact(suite: Suite, real_corpus: Corpus) -> None:
    for task in suite.tasks:
        if task["family"] != "refusal":
            continue
        lowered = normalize(render_doc(real_corpus.get(str(task["doc_id"]), str(task["doc_sha256"])))).lower()
        key = task["key"]
        assert isinstance(key, dict)
        for pattern in key["absent"]:
            assert not re.search(pattern, lowered), (task["id"], pattern)


def test_builder_fails_loud_on_a_thin_corpus(tmp_path: Path) -> None:
    pins = write_corpus(tmp_path, [record("op-1", "opinions_cl", "Too short.")], [])
    with pytest.raises(B.BuildError):
        B.build(Corpus.load(tmp_path, pins), B.NameScreen(frozenset({("doe", "jane")})))


def test_s101_quote08_case_name_in_italics(materialized: list[Materialized]) -> None:
    """S101 FIX 2: quote-08 quoted verbatim with the case name in markdown italics is still verbatim."""
    item = next(m for m in materialized if m.task_id == "quote-08")
    assert isinstance(item.key, QuoteKey) and "Burnette" in item.key.sentence
    assert item.score(item.key.sentence.replace("Burnette", "*Burnette*")).passed
    assert not item.score(item.key.sentence.replace("Burnette", "*Burnett*")).passed


def test_s101_citation_courts_on_real_keys(materialized: list[Materialized]) -> None:
    """S101 FIX 3: citation-11 is a Tennessee Supreme Court case; citation-01 is a South Carolina Supreme Court case."""
    def cite(task: str, court: str) -> tuple[Materialized, str]:
        item = next(m for m in materialized if m.task_id == task)
        assert isinstance(item.key, CitationKey)
        cites = ", ".join(" ".join(r) for r in item.key.reporters)
        return item, f"Name v. Name, {cites} ({court}{item.key.year})"

    for task, court, passed in (("citation-11", "Tenn. ", True), ("citation-11", "Tenn. Ct. App. ", False),
                                ("citation-01", "", True), ("citation-01", "S.C. ", True),
                                ("citation-01", "Ct. App. ", False), ("citation-01", "4th Cir. ", False),
                                ("citation-01", "N.C. Ct. ", False), ("citation-03", "S.C. Ct. App. ", True),
                                ("citation-03", "", False), ("citation-09", "D.S.C. ", False)):
        item, reply = cite(task, court)
        assert item.score(reply).passed is passed, (task, court)
