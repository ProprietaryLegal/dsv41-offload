#!/usr/bin/env python3
"""Offline, deterministic builder of the L-SUITE task file from the pinned public corpus. CPU only, no network.

  python3 -m offload.lsuite.build --bench-data /opt/dsv41/data/bench-data            # check (default)
  python3 -m offload.lsuite.build --bench-data /opt/dsv41/data/bench-data --write    # (re)write tasks.v1.json

Selection is a pure function of the corpus and the constants below: documents are ranked by sha256(seed | family |
doc_id), each single-document task takes a distinct document, and within a document the candidate nearest a target
depth wins. `--check` rebuilds in memory and refuses unless the committed file is byte-identical.

Client-name screen (coordinator decision on S101 O2): no suite input may contain a client's full name, not even a
public opinion's counsel block. The build reads the client folder names (open and !Closed) and the matter-cache
exclusion list, and refuses every document whose rendered text holds a client's surname and first name in either
order (optionally with a middle initial, or as "Surname, First"). The names are needles held in memory only: never
written, printed or hashed, and no list of screened documents is stored. The screen is required for --write and
--check alike, so both run only where those sources exist (Controller); a missing source fails loud.

Families (F-METHOD section 3.5):
  quote     12  quote verbatim the sentence that contains a registered 5-word phrase (exact-substring scoring)
  figure    12  the dollar amount / date / percentage in the sentence with a registered phrase; the value occurs once in
                the document (normalized exact match); 3 come from pinned-only RECAP filings (offsets + sha256 only)
  citation  12  the reporter citation of a case the opinion cites; South Carolina style for SC sources (regex scoring)
  refusal    4  a plausible question the document cannot answer (must reply NOT IN RECORD, invent nothing)
  needle     6  ~150K-token haystack of public opinions with one synthetic docket number planted at 6 depths
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import re
import sys
from typing import TypeVar

from offload.lsuite.corpus import CORPUS_FILES, Corpus, Doc, sha256_text
from offload.lsuite.scoring import (
    MONEY_RE, PERCENT_RE, FigureKind, canonical_decimal, canonical_figure, find_dates, normalize)
from offload.lsuite.tasks import DOC_SEPARATOR, SPECIAL_OPEN, SUITE, TASKS_FILE, render_doc

T = TypeVar("T")

SEED = SUITE
INSTRUCTION = ("You answer questions about public court documents for a law office. Use only the materials between "
               "BEGIN MATERIALS and END MATERIALS. If the materials do not state the answer, reply exactly: "
               "NOT IN RECORD")
MAX_TOKENS = {"quote": 256, "figure": 48, "citation": 160, "refusal": 64, "needle": 48}
# Penalties are pinned: a request that omits them takes the model directory's generation_config.json defaults
# (vLLM get_diff_sampling_param), and they apply before the greedy argmax, so T = 0 does not neutralize them (S101).
REQUEST: dict[str, object] = {"temperature": 0, "top_p": 1, "seed": 0, "repetition_penalty": 1.0,
                              "presence_penalty": 0.0, "frequency_penalty": 0.0,
                              "chat_template_kwargs": {"thinking": False}, "max_tokens": MAX_TOKENS}
DOC_CHARS = (3_000, 60_000)            # single-document tasks: about 1K-17K prompt tokens
NEEDLE_CHARS = 525_000                 # 150K tokens at the 3.5 chars/token planning ratio (AUD1); usage gives the truth
NEEDLE_FLOOR_RATIO = 2.5               # refuse a haystack that could exceed 92 % of the context at 2.5 chars/token
MAX_CONTEXT = 327_680
NEEDLE_DEPTHS = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)
NEEDLE_COUNTIES = ("47", "40", "23", "10", "32", "02")
PHRASE_WORDS = 5
CLIENT_FILES = Path("/opt/dsv41/shared/client folders")            # open matters; closed ones are in its !Closed
MATTER_EXCLUSIONS = Path("/opt/dsv41/data/private/matter-exclusions.txt")
CLIENT_SCREEN_NOTE = ("every document whose rendered text holds a client's full name (client folders open and !Closed "
                      "folder names plus the matter-cache exclusion list; surname and first name in either order) is "
                      "refused at build; names are never stored")

QUOTE_Q = ('Quote, word for word, the complete sentence of the document that contains the phrase "{phrase}". '
           "Reply with that sentence only.")
FIGURE_Q = ('Find the sentence of the document that contains the phrase "{phrase}". What {desc} does that sentence '
            "state? Reply with the {desc} only, as written in the document.")
FIGURE_DESC: dict[FigureKind, str] = {"money": "dollar amount", "date": "date", "percent": "percentage"}
CITE_SC_Q = ('The document cites "{name}". Give the full citation for that case as the document supplies it, in South '
             "Carolina citation style: the case name; the South Carolina Reports volume and first page; the South "
             'Eastern Reporter volume and first page; then, in parentheses, "Ct. App." if the Court of Appeals decided '
             "the case, and the year. Reply with the citation only.")
CITE_Q = ('The document cites "{name}". Give the full reporter citation for that case exactly as the document '
          "supplies it: the case name, volume, reporter, first page, and the court and year in parentheses. Reply "
          "with the citation only.")
NEEDLE_PHRASE = "sealed companion matter"
NEEDLE_SENTENCE = "Clerk's note: the " + NEEDLE_PHRASE + " was assigned docket number {docket}."
NEEDLE_Q = "What docket number was assigned to the " + NEEDLE_PHRASE + "? Reply with the docket number only."


@dataclass(frozen=True)
class Probe:
    kind: str
    question: str
    absent: tuple[str, ...]           # regexes that must not match the lower-cased, normalized document


PROBES = (
    Probe("money", "What dollar amount of attorney's fees did the court award?",
          (r"\$", r"dollar", r"attorneys?'?s?'? fee", r"counsel fee")),
    Probe("date", "On what date did the parties marry?",
          (r"marri", r"marry", r"wedding", r"spouse", r"husband", r"\bwife\b")),
    Probe("name", "What is the name of the guardian ad litem appointed for the child?",
          (r"guardian", r"ad litem", r"\bgal\b")),
    Probe("percent", "What percentage of the marital estate did the family court award to the wife?",
          (r"%", r"percent", r"per cent", r"marital", r"\bwife\b")),
)
PROBE_GROUPS = ("sc_family", "sc_civil", "other_family", "federal_civil")
QUOTE_GROUPS = (("sc_family", 5), ("sc_civil", 2), ("other_family", 3), ("federal_civil", 2))
FIGURE_PLAN: tuple[tuple[FigureKind, str, int], ...] = (
    ("percent", "opinions_cl", 3), ("money", "opinions_cl", 4), ("money", "filings_recap", 1),
    ("date", "opinions_cl", 2), ("date", "filings_recap", 2))
CITATION_PLAN = (("scctapp", 5), ("sc", 3), ("ca4", 2), ("tennctapp", 1), ("ncctapp|nc|scotus", 1))

ABBREVIATIONS = frozenset({
    "etc", "seq", "cert", "art", "sec", "para", "pp", "rev", "supp", "vol", "ann", "cl", "ch", "ed", "no", "nos", "viz",
    "al", "app", "jr", "sr", "inc", "corp", "co", "ltd", "bros", "assn", "dept", "govt", "stat", "reg", "ex", "cir",
})
MONTH_WORDS = frozenset({"january", "february", "march", "april", "may", "june", "july", "august", "september",
                         "october", "november", "december"})
_END = re.compile(r"(?<=[a-z])\.(?=\s+[A-Z])")
FIGURE_SOURCE_RE: dict[FigureKind, re.Pattern[str]] = {
    "money": re.compile(r"\$\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?(?!\d|,\d|\.\d)"
                        r"(?!\s*(?:million|billion|thousand))"),
    "date": re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|"
                       r"December)\s+\d{1,2},\s+\d{4}\b"),
    "percent": re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d+)?(?:\s?%|\s+percent\b)"),
}
_W = r"[A-Z][A-Za-z'&-]*\.?"
_NAME = rf"(?P<name>{_W}(?:\s+{_W}){{0,4}}\s+v\.\s+{_W}(?:\s+{_W}){{0,4}})"
_PIN = r"(?:,\s+\d{1,4}(?:-\d{1,4})?)?"
SC_CITE = re.compile(
    _NAME + rf",\s+(?P<v1>\d{{1,3}})\s+S\.C\.\s+(?P<p1>\d{{1,4}}){_PIN},\s+(?P<v2>\d{{1,3}})\s+(?P<r2>S\.E\.(?:2d)?)\s+"
    rf"(?P<p2>\d{{1,4}}){_PIN}\s+\((?P<ct>Ct\.\s+App\.\s+)?(?P<yr>\d{{4}})\)")
GEN_CITE = re.compile(
    _NAME + r",\s+(?P<v1>\d{1,4})\s+(?P<rep>U\.S\.|F\.\s?(?:2d|3d|4th)|F\.\s?Supp\.\s?(?:2d|3d)?|S\.W\.(?:2d|3d)|"
    r"N\.C\.(?:\s?App\.)?|S\.E\.(?:2d)?)\s+(?P<p1>\d{1,5})" + _PIN + r"\s+\((?P<ct>[0-9A-Za-z.\s]*?)\s*(?P<yr>\d{4})\)")
SIGNALS = frozenset({"See", "Cf.", "Accord", "In", "Also", "But", "Compare", "And", "Citing", "Quoting", "The", "Under",
                     "Contra", "Moreover", "Thus", "Further", "However", "Similarly", "Following", "Applying"})
_NAME_OK = re.compile(r"^[A-Z][A-Za-z.'&-]+(?: [A-Za-z.'&-]+)* v\. [A-Z][A-Za-z.'&-]*(?: [A-Za-z.'&-]+)*$")
_BEFORE_NAME = re.compile(r"(?:^|[;:(.)]|\b(?:[Ss]ee(?: also)?|[Cc]f\.|accord|[Ii]n|citing|quoting|e\.g\.,)|"
                          r"\b[a-z]{3,}[.,])\s*$")


class BuildError(Exception):
    """The corpus cannot fill a family under the registered rules (fail loud; never a short suite)."""


# ---------------------------------------------------------------- client-name screen

_TAG_WORDS = frozenset({"jr", "sr", "ii", "iii", "iv", "dss", "gal", "mtr", "v", "vs", "and", "the", "of", "llc",
                        "inc", "closed", "file", "files", "old", "new", "et", "al"})


def name_pairs(entry: str) -> set[tuple[str, str]]:
    """Lower-case (word, word) needles from one client folder name or exclusion entry.

    Segments split on " - ", "&", "+", "and" and "|"; parentheticals, digits, tags (Jr., DSS, ...) and one-letter
    initials drop out. Every adjacent pair of remaining words is a needle, plus the first and last word of a longer
    segment, so "Surname First", "Surname, First" and "Surname First Middle" all yield the surname/first-name pair.
    Over-screening only removes candidate documents; under-screening would leak a client name.
    """
    cleaned = re.sub(r"\([^)]*\)", " ", entry).lstrip("!_ ")
    pairs: set[tuple[str, str]] = set()
    for segment in re.split(r"\s+-\s+|&|\+|\band\b|\|", cleaned):
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", segment.replace(",", " "))]
        words = [w for w in words if len(w) >= 2 and w not in _TAG_WORDS]
        pairs.update(pairwise(words))
        if len(words) > 2:
            pairs.add((words[0], words[-1]))
    return pairs


def _word_grams(text: str) -> set[tuple[str, str]]:
    """Adjacent word pairs of a text, plus pairs around a one-letter middle initial ("Jane Q. Doe" -> (jane, doe))."""
    words = re.findall(r"[a-z0-9][a-z0-9'\-]*", normalize(text).lower())
    grams = set(pairwise(words))
    grams.update((words[i], words[i + 2]) for i in range(len(words) - 2) if len(words[i + 1]) == 1)
    return grams


@dataclass(frozen=True)
class NameScreen:
    """Client-name needles, held in memory only. repr never shows them, so no traceback or test report can."""

    pairs: frozenset[tuple[str, str]] = field(repr=False)

    def __repr__(self) -> str:
        return f"NameScreen(<{len(self.pairs)} needle pairs, not shown>)"

    def matches(self, text: str) -> bool:
        grams = _word_grams(text)
        return any((a, b) in grams or (b, a) in grams for a, b in self.pairs)


def load_name_screen(client_files: Path = CLIENT_FILES, exclusions: Path = MATTER_EXCLUSIONS) -> NameScreen:
    """Needles from the open and closed client folder names and the exclusion list; a missing source fails loud."""
    entries: list[str] = []
    for root in (client_files, client_files / "!Closed"):
        if not root.is_dir():
            raise BuildError(f"client-name screen: {root} is not a directory (the screen never runs without it)")
        entries.extend(child.name for child in root.iterdir() if child.is_dir())
    if not exclusions.is_file():
        raise BuildError(f"client-name screen: {exclusions} is missing (the screen never runs without it)")
    entries.extend(line for line in exclusions.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#"))
    pairs = frozenset(pair for entry in entries for pair in name_pairs(entry))
    if not pairs:
        raise BuildError("client-name screen: the sources yielded no names")
    return NameScreen(pairs)


# ---------------------------------------------------------------- ranking and spans


def rank(docs: list[Doc], family: str) -> list[Doc]:
    return sorted(docs, key=lambda doc: hashlib.sha256(f"{SEED}|{family}|{doc.doc_id}".encode()).hexdigest())


def span(doc: Doc, start: int, end: int) -> dict[str, object]:
    raw = doc.body[start:end]
    out: dict[str, object] = {"start": start, "end": end, "sha256": sha256_text(raw)}
    if doc.redistributable:
        out["value"] = normalize(raw)
    return out


def sized(doc: Doc) -> bool:
    return DOC_CHARS[0] <= len(doc.body) <= DOC_CHARS[1]


# ---------------------------------------------------------------- sentences and phrases


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int          # exclusive; the span ends with the sentence's final period


def _clean_end(body: str, index: int) -> bool:
    word = re.search(r"([A-Za-z']+)$", body[max(0, index - 40):index])
    if word is None:
        return False
    token = word.group(1)
    return token.isalpha() and token.islower() and len(token) >= 3 and token not in ABBREVIATIONS


def _layout_clean(body: str, start: int, raw: str) -> bool:
    """No hyphenation, page break, heading line or page-number line inside the span."""
    if "-\n" in raw or "\n\n" in raw or "\f" in raw or "\u00ad" in raw:
        return False
    lines = raw.split("\n")
    starts_line = start == 0 or body[start - 1] == "\n"
    for index, line in enumerate(lines):
        full = (index > 0 or starts_line) and index < len(lines) - 1
        if full and len(line.strip()) < 40:
            return False
        if line.strip() and line.strip().upper() == line.strip() and re.search(r"[A-Z]{3}", line):
            return False
    return True


def sentences(doc: Doc, low: int, high: int) -> Iterator[Sentence]:
    body = doc.body
    ends = [match.start() for match in _END.finditer(body) if _clean_end(body, match.start())]
    for left, right in pairwise(ends):
        gap = re.match(r"\s+", body[left + 1:])
        if gap is None:
            continue
        start, end = left + 1 + gap.end(), right + 1
        raw = body[start:end]
        if re.search(r"[.?!](?=\s)", raw[:-1]):
            continue
        if re.search(r"[A-Za-z][0-9]|[A-Za-z][.,;:)\]][0-9]", raw):
            continue
        if re.search('["\u201c\u201d\\[\\]\u00a7]', raw) or not _layout_clean(body, start, raw):
            continue
        text = normalize(raw)
        if low <= len(text) <= high and text[0].isupper():
            yield Sentence(start, end)


def phrase(doc: Doc, sentence: Sentence, haystack_text: str, avoid: tuple[int, int] | None = None
           ) -> tuple[int, int] | None:
    """A 5-word window inside the sentence (never its first or last word) that occurs once in the document."""
    tokens = [(m.start() + sentence.start, m.end() + sentence.start)
              for m in re.finditer(r"\S+", doc.body[sentence.start:sentence.end])]
    count = len(tokens)
    windows = sorted(range(1, count - PHRASE_WORDS), key=lambda i: (abs(2 * i + PHRASE_WORDS - count), i))
    for index in windows:
        start, end = tokens[index][0], tokens[index + PHRASE_WORDS - 1][1]
        if avoid is not None and not (end <= avoid[0] or start >= avoid[1]):
            continue
        words = normalize(doc.body[start:end])
        if re.search(r"[\d$%]", words) or MONTH_WORDS.intersection(re.findall(r"[a-z]+", words.lower())):
            continue
        if not (words[0].isalpha() and words[-1].isalpha()):
            continue
        if words != " ".join(doc.body[start:end].split()):
            continue          # typography changed by normalization (curly quote, dash): the phrase must be literal
        if haystack_text.count(words) == 1:
            return start, end
    return None


def nearest(items: list[tuple[int, T]], length: int, depth: float) -> T:
    """The candidate whose offset is nearest `depth` of the document (ties: the earlier one)."""
    return min(items, key=lambda item: (abs(item[0] / length - depth), item[0]))[1]


# ---------------------------------------------------------------- families


@dataclass
class Builder:
    corpus: Corpus
    used: set[str]                      # documents already holding a single-document task
    names: set[str]                     # case names already asked for (one task per case across the suite)
    blocked: frozenset[str]             # documents refused by the client-name screen (in memory only)
    haystack: set[str] = field(default_factory=set)   # opinions rendered into the needle haystack

    def take(self, docs: list[Doc], family: str, need: int,
             pick: Callable[[Doc, int], dict[str, object] | None]) -> list[dict[str, object]]:
        chosen: list[dict[str, object]] = []
        for doc in rank(docs, family):
            if len(chosen) == need:
                break
            if doc.doc_id in self.used or doc.doc_id in self.blocked:
                continue
            task = pick(doc, len(chosen))
            if task is not None:
                chosen.append(task)
                self.used.add(doc.doc_id)
        if len(chosen) < need:
            raise BuildError(f"{family}: only {len(chosen)} of {need} tasks under the registered rules")
        return chosen

    @staticmethod
    def base(doc: Doc) -> dict[str, object]:
        return {"doc_id": doc.doc_id, "doc_sha256": doc.sha256, "licence": doc.licence, "court_id": doc.court_id,
                "group": doc.group}

    def quotes(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        total = sum(n for _, n in QUOTE_GROUPS)

        def pick(doc: Doc, index: int) -> dict[str, object] | None:
            depth = (len(out) + index + 0.5) / total
            materials = normalize(render_doc(doc))
            items: list[tuple[int, tuple[Sentence, tuple[int, int]]]] = []
            for sentence in sentences(doc, 100, 260):
                if materials.count(normalize(doc.body[sentence.start:sentence.end])) != 1:
                    continue
                words = phrase(doc, sentence, materials)
                if words is not None:
                    items.append((sentence.start, (sentence, words)))
            if not items:
                return None
            sentence, words = nearest(items, len(doc.body), depth)
            return self.base(doc) | {"question": QUOTE_Q, "slots": {"phrase": span(doc, *words)},
                                     "key": {"sentence": span(doc, sentence.start, sentence.end)}}

        for group, need in QUOTE_GROUPS:
            docs = [d for d in self.corpus.opinions() if d.group == group and sized(d)]
            out.extend(self.take(docs, f"quote-{group}", need, pick))
        return out

    def figures(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        total = sum(n for _, _, n in FIGURE_PLAN)
        for kind, collection, need in FIGURE_PLAN:
            docs = [d for d in self.corpus.docs if d.collection == collection and sized(d)]

            def pick(doc: Doc, index: int, kind: FigureKind = kind) -> dict[str, object] | None:
                depth = (len(out) + index + 0.5) / total
                materials = normalize(render_doc(doc))
                in_doc = figure_values(kind, materials)
                items: list[tuple[int, tuple[tuple[int, int], tuple[int, int], str]]] = []
                for sentence in sentences(doc, 80, 320):
                    raw = doc.body[sentence.start:sentence.end]
                    found = list(FIGURE_SOURCE_RE[kind].finditer(raw))
                    if len(found) != 1 or len(figure_values(kind, normalize(raw))) != 1:
                        continue
                    fig = (sentence.start + found[0].start(), sentence.start + found[0].end())
                    value = canonical_figure(kind, doc.body[fig[0]:fig[1]])
                    if in_doc.count(value) != 1:
                        continue
                    words = phrase(doc, sentence, materials, avoid=fig)
                    if words is not None:
                        items.append((sentence.start, (fig, words, value)))
                if not items:
                    return None
                fig, words, value = nearest(items, len(doc.body), depth)
                key: dict[str, object] = {"kind": kind, "figure": span(doc, *fig)}
                if doc.redistributable:
                    key["canonical"] = value
                return self.base(doc) | {"question": FIGURE_Q.replace("{desc}", FIGURE_DESC[kind]),
                                         "slots": {"phrase": span(doc, *words)}, "key": key}

            out.extend(self.take(docs, f"figure-{kind}-{collection}", need, pick))
        return out

    def citations(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        total = sum(n for _, n in CITATION_PLAN)
        for courts, need in CITATION_PLAN:
            wanted = set(courts.split("|"))
            docs = [d for d in self.corpus.opinions() if d.court_id in wanted and sized(d)]

            def pick(doc: Doc, index: int) -> dict[str, object] | None:
                depth = (len(out) + index + 0.5) / total
                items = [item for item in citations_in(doc) if item[1][0] not in self.names]
                if not items:
                    return None
                name, key, cite = nearest(items, len(doc.body), depth)
                self.names.add(name)
                question = CITE_SC_Q if key["sc_style"] else CITE_Q
                return self.base(doc) | {"question": question.replace("{name}", name),
                                         "key": key | {"citation": span(doc, *cite)}}

            out.extend(self.take(docs, f"citation-{courts}", need, pick))
        return out

    def refusals(self) -> list[dict[str, object]]:
        """One probe each; documents are tried group by group (family groups first, so the question is plausible)."""
        out: list[dict[str, object]] = []
        for probe in PROBES:
            ranked = [d for group in PROBE_GROUPS
                      for d in rank([d for d in self.corpus.opinions() if d.group == group and sized(d)],
                                    f"refusal-{probe.kind}")]
            for doc in ranked:
                lowered = normalize(render_doc(doc)).lower()
                if doc.doc_id in self.used or doc.doc_id in self.blocked or any(
                        re.search(pattern, lowered) for pattern in probe.absent):
                    continue
                out.append(self.base(doc) | {"question": probe.question,
                                             "key": {"kind": probe.kind, "absent": list(probe.absent)}})
                self.used.add(doc.doc_id)
                break
            else:
                raise BuildError(f"refusal {probe.kind}: no document satisfies the absence rule")
        return out

    def needles(self) -> list[dict[str, object]]:
        docs = [doc for doc in rank(self.corpus.opinions(), "haystack") if doc.doc_id not in self.blocked]
        chosen: list[Doc] = []
        size = 0
        for doc in docs:
            chosen.append(doc)
            size += len(render_doc(doc)) + len(DOC_SEPARATOR)
            if size >= NEEDLE_CHARS + 10_000:
                break
        self.haystack.update(doc.doc_id for doc in chosen)
        joined = DOC_SEPARATOR.join(render_doc(doc) for doc in chosen)
        chars = joined.rfind("\n\n", 0, NEEDLE_CHARS)
        hay = joined[:chars]
        if chars / NEEDLE_FLOOR_RATIO > 0.92 * MAX_CONTEXT:
            raise BuildError("haystack could exceed the context at 2.5 chars/token")
        corpus_text = "\n".join(doc.body for doc in self.corpus.docs)
        if NEEDLE_PHRASE in normalize(corpus_text).lower():
            raise BuildError(f"needle phrase {NEEDLE_PHRASE!r} occurs in the corpus")
        recipe = {"docs": [[doc.doc_id, doc.sha256] for doc in chosen], "chars": chars, "sha256": sha256_text(hay)}
        out: list[dict[str, object]] = []
        for index, depth in enumerate(NEEDLE_DEPTHS):
            at = hay.find("\n\n", int(depth * chars)) + 2
            if not 2 <= at < chars - 2:
                raise BuildError(f"needle depth {depth}: no paragraph break after {int(depth * chars)}")
            number = int(hashlib.sha256(f"{SEED}|needle|{index}".encode()).hexdigest()[:8], 16) % 9000 + 1000
            docket = f"2031-DR-{NEEDLE_COUNTIES[index]}-{number:04d}"
            if docket in corpus_text:
                raise BuildError(f"synthetic docket {docket} occurs in the corpus")
            out.append({"depth": depth, "haystack": recipe, "insert_at": at,
                        "needle": NEEDLE_SENTENCE.format(docket=docket), "question": NEEDLE_Q,
                        "key": {"docket": docket}})
        if len({task["insert_at"] for task in out}) != len(out):
            raise BuildError("two needle depths share one paragraph break")
        return out


def figure_values(kind: FigureKind, normalized: str) -> list[str]:
    """Canonical values of every figure of `kind` in a normalized text, in order (the scorer's own parsers)."""
    if kind == "date":
        return [value for _, value in find_dates(normalized)]
    pattern = MONEY_RE if kind == "money" else PERCENT_RE
    return [canonical_decimal(m["int"], m["frac"]) for m in pattern.finditer(normalized)]


def case_name(match: re.Match[str]) -> str | None:
    """The cited case name with leading signal words removed, or None when it is not a clean 'X v. Y' name."""
    tokens = normalize(match["name"]).split()
    while tokens and tokens[0] in SIGNALS:
        tokens = tokens[1:]
    name = " ".join(tokens)
    if not _NAME_OK.match(name) or len(tokens[0].rstrip(".")) < 3 or len(tokens) > 9:
        return None
    return name


def citations_in(doc: Doc) -> list[tuple[int, tuple[str, dict[str, object], tuple[int, int]]]]:
    """Full citations whose case name maps to exactly one citation in the document, as (offset, (name, key, span)).

    Every occurrence counts toward uniqueness; only an occurrence whose name starts cleanly (after a signal word,
    punctuation or a sentence end, never mid-name such as "Soc. Servs." inside a longer agency name) is offered.
    """
    sc_source = doc.jurisdiction == "SC"
    pattern = SC_CITE if sc_source else GEN_CITE
    by_name: dict[str, set[str]] = {}
    first: dict[str, tuple[int, dict[str, object], tuple[int, int]]] = {}
    for match in pattern.finditer(doc.body):
        name = case_name(match)
        if name is None:
            continue
        if sc_source:
            reporters = [[match["v1"], "S.C.", match["p1"]], [match["v2"], match["r2"], match["p2"]]]
            court = "Ct. App." if match["ct"] else ""
        else:
            reporters = [[match["v1"], normalize(match["rep"]), match["p1"]]]
            court = normalize(match["ct"])
        key: dict[str, object] = {"reporters": reporters, "year": match["yr"], "court": court, "sc_style": sc_source}
        by_name.setdefault(name, set()).add(json.dumps(key, sort_keys=True))
        if not _BEFORE_NAME.search(doc.body[max(0, match.start() - 40):match.start()]):
            continue
        signals = len(normalize(match["name"]).split()) - len(name.split())
        name_start = [m.start() for m in re.finditer(r"\S+", match["name"])][signals] + match.start("name")
        first.setdefault(name, (match.start(), key, (name_start, match.end())))
    out: list[tuple[int, tuple[str, dict[str, object], tuple[int, int]]]] = []
    for name, keys in by_name.items():
        if len(keys) != 1 or name not in first:
            continue
        if any(other != name and name in other for other in by_name):
            continue
        start, key, cite = first[name]
        out.append((start, (name, key, cite)))
    return sorted(out, key=lambda item: item[0])


# ---------------------------------------------------------------- assembly


def build(corpus: Corpus, screen: NameScreen) -> dict[str, object]:
    blocked = frozenset(doc.doc_id for doc in corpus.docs if screen.matches(render_doc(doc)))
    builder = Builder(corpus, set(), set(), blocked)
    figures = builder.figures()            # percent first: the scarcest kind claims its documents first
    citations = builder.citations()
    refusals = builder.refusals()
    quotes = builder.quotes()
    needles = builder.needles()
    families = [("quote", quotes), ("figure", figures), ("citation", citations), ("refusal", refusals),
                ("needle", needles)]
    tasks: list[dict[str, object]] = []
    for family, items in families:
        for number, item in enumerate(items, start=1):
            tasks.append({"id": f"{family}-{number:02d}", "family": family} | item)
    for doc in corpus.opinions() + corpus.filings():
        if doc.doc_id in builder.used and SPECIAL_OPEN in render_doc(doc):
            raise BuildError(f"{doc.doc_id} carries a literal special-token opener (sanitizing would move spans)")
    inputs = builder.used | builder.haystack
    if any(screen.matches(render_doc(doc)) for doc in corpus.docs if doc.doc_id in inputs):
        raise BuildError("a selected document matches the client-name screen")   # never names it
    return {
        "suite": SUITE, "spec": "F-METHOD.plan.md section 3.5; brief documents/plans/lanes/L-SUITE.md",
        "corpus": dict(CORPUS_FILES),
        "needle_tokens_nominal": 150_000, "chars_per_token_planning": 3.5,
        "families": {family: len(items) for family, items in families},
        "client_name_screen": CLIENT_SCREEN_NOTE,
        "instruction": INSTRUCTION, "request": REQUEST, "tasks": tasks,
    }


def dumps(suite: dict[str, object]) -> bytes:
    return (json.dumps(suite, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the L-SUITE task file (offline).")
    parser.add_argument("--bench-data", type=Path, required=True, help="directory holding the pinned corpus")
    parser.add_argument("--out", type=Path, default=TASKS_FILE)
    parser.add_argument("--write", action="store_true", help="write the file instead of checking it")
    parser.add_argument("--client-files", type=Path, default=CLIENT_FILES,
                        help="client-folder root holding open matters and !Closed (names read for the screen only)")
    parser.add_argument("--matter-exclusions", type=Path, default=MATTER_EXCLUSIONS)
    args = parser.parse_args(argv)
    try:
        screen = load_name_screen(args.client_files, args.matter_exclusions)
        data = dumps(build(Corpus.load(args.bench_data), screen))
    except BuildError as error:
        print(f"BUILD REFUSED: {error}", file=sys.stderr)
        return 2
    if args.write:
        args.out.write_bytes(data)
        print(f"wrote {args.out} ({len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()})")
        return 0
    current = args.out.read_bytes() if args.out.is_file() else b""
    if current != data:
        print(f"CHECK FAIL: {args.out} differs from a fresh build ({len(current):,} vs {len(data):,} bytes)")
        return 1
    print(f"CHECK PASS: {args.out} is byte-identical to a fresh build (sha256 {hashlib.sha256(data).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
