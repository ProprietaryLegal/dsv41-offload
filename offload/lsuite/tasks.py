"""Task file schema, prompt rendering and materialization (task + pinned corpus -> request body + answer key).

The task file (`tasks.v1.json`) is the pre-registered suite. It names every source span by document id, document
sha256 and character offsets into the document body, plus the sha256 of the span. Spans from public-domain opinions
also carry their normalized `value` inline so a reviewer can read the key; spans from pinned-only RECAP filings carry
no text at all and are rebuilt here from the corpus at run time. Every span is re-verified on every materialization,
so a corpus or task-file drift fails before any request is sent.

The file deliberately avoids the data-boundary text keys (`text`, `prompt`, `answer`, `content`, ...): it holds no
generated text and no prompts, only questions written for this suite, public-domain spans and synthetic needles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal, cast

from offload.lsuite.corpus import Corpus, CorpusError, Doc, sha256_text
from offload.lsuite.scoring import (
    CitationKey, FigureKind, ProbeKind, Score, canonical_figure, normalize, score_citation, score_figure, score_needle,
    score_quote, score_refusal)

SUITE = "lsuite-v1"
TASKS_FILE = Path(__file__).resolve().parent / "tasks.v1.json"
FAMILIES = ("quote", "figure", "citation", "needle", "refusal")
Family = Literal["quote", "figure", "citation", "needle", "refusal"]
SPECIAL_OPEN = "<\uff5c"          # DeepSeek special-token opener; user text must not carry it literally (RUNBOOK)
DOC_SEPARATOR = "\n\n----------\n\n"


class TaskError(Exception):
    """The task file is malformed or no longer matches the pinned corpus."""


# ---------------------------------------------------------------- JSON helpers (strict, typed)


def as_obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TaskError(f"{where}: expected an object")
    return cast(dict[str, object], value)


def as_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise TaskError(f"{where}: expected a string")
    return value


def as_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TaskError(f"{where}: expected an integer")
    return value


def as_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise TaskError(f"{where}: expected a list")
    return cast(list[object], value)


# ---------------------------------------------------------------- rendering


def render_doc(doc: Doc) -> str:
    """One document as the model sees it: a three-line header, a blank line, the body as published."""
    return f"Document: {doc.title}\nCourt: {doc.court}\nDate: {doc.date}\n\n{doc.body}"


def sanitize(materials: str) -> str:
    return materials.replace(SPECIAL_OPEN, "<\u200b\uff5c")


def user_message(instruction: str, materials: str, question: str) -> str:
    return f"{instruction}\n\nBEGIN MATERIALS\n{sanitize(materials)}\nEND MATERIALS\n\nQuestion: {question}"


def haystack(corpus: Corpus, doc_ids: list[tuple[str, str]], chars: int) -> str:
    """Rendered opinions joined by the separator, cut at the last paragraph break at or before `chars`."""
    joined = DOC_SEPARATOR.join(render_doc(corpus.get(doc_id, digest)) for doc_id, digest in doc_ids)
    if len(joined) < chars:
        raise TaskError(f"haystack docs give {len(joined)} chars < {chars}")
    return joined[:chars]


def insert_needle(hay: str, at: int, needle: str) -> str:
    return f"{hay[:at]}{needle}\n\n{hay[at:]}"


# ---------------------------------------------------------------- spans


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    sha256: str
    value: str | None          # normalized span text; only for public-domain sources

    @classmethod
    def parse(cls, raw: object, where: str) -> Span:
        obj = as_obj(raw, where)
        value = obj.get("value")
        return cls(as_int(obj.get("start"), f"{where}.start"), as_int(obj.get("end"), f"{where}.end"),
                   as_str(obj.get("sha256"), f"{where}.sha256"),
                   None if value is None else as_str(value, f"{where}.value"))

    def resolve(self, doc: Doc, where: str) -> str:
        """The raw span from the corpus, verified against its pin (and its inline value when present)."""
        if not 0 <= self.start < self.end <= len(doc.body):
            raise TaskError(f"{where}: offsets {self.start}:{self.end} outside {doc.doc_id}")
        raw = doc.body[self.start:self.end]
        if sha256_text(raw) != self.sha256:
            raise TaskError(f"{where}: span sha256 mismatch in {doc.doc_id}")
        if self.value is not None and normalize(raw) != self.value:
            raise TaskError(f"{where}: inline value differs from the corpus span")
        if self.value is not None and not doc.redistributable:
            raise TaskError(f"{where}: inline value from a pinned-only source ({doc.doc_id})")
        return raw


# ---------------------------------------------------------------- materialized tasks


@dataclass(frozen=True)
class QuoteKey:
    sentence: str


@dataclass(frozen=True)
class FigureKey:
    kind: FigureKind
    value: str


@dataclass(frozen=True)
class NeedleKey:
    docket: str


@dataclass(frozen=True)
class RefusalKey:
    kind: ProbeKind


Key = QuoteKey | FigureKey | CitationKey | NeedleKey | RefusalKey


@dataclass(frozen=True)
class Materialized:
    task_id: str
    family: Family
    message: str
    max_tokens: int
    key: Key

    def score(self, reply: str) -> Score:
        key = self.key
        if isinstance(key, QuoteKey):
            return score_quote(reply, key.sentence)
        if isinstance(key, FigureKey):
            return score_figure(reply, key.kind, key.value)
        if isinstance(key, CitationKey):
            return score_citation(reply, key)
        if isinstance(key, NeedleKey):
            return score_needle(reply, key.docket)
        return score_refusal(reply, key.kind)


@dataclass(frozen=True)
class Suite:
    path: Path
    sha256: str
    suite: str
    corpus: dict[str, str]
    instruction: str
    request: dict[str, object]
    max_tokens: dict[str, int]
    tasks: list[dict[str, object]]

    @classmethod
    def load(cls, path: Path = TASKS_FILE) -> Suite:
        data = path.read_bytes()
        top = as_obj(json.loads(data), "task file")
        if top.get("suite") != SUITE:
            raise TaskError(f"task file suite {top.get('suite')!r} is not {SUITE}")
        corpus = {name: as_str(digest, f"corpus.{name}")
                  for name, digest in as_obj(top.get("corpus"), "corpus").items()}
        request = as_obj(top.get("request"), "request")
        max_tokens = {fam: as_int(value, f"max_tokens.{fam}")
                      for fam, value in as_obj(request.get("max_tokens"), "request.max_tokens").items()}
        if set(max_tokens) != set(FAMILIES):
            raise TaskError(f"request.max_tokens must name exactly {FAMILIES}")
        tasks = [as_obj(task, f"tasks[{index}]") for index, task in enumerate(as_list(top.get("tasks"), "tasks"))]
        ids = [as_str(task.get("id"), "task id") for task in tasks]
        if len(set(ids)) != len(ids):
            raise TaskError("duplicate task ids")
        counts = as_obj(top.get("families"), "families")
        for family in FAMILIES:
            if sum(1 for task in tasks if task.get("family") == family) != counts.get(family):
                raise TaskError(f"family {family}: task count differs from the header")
        return cls(path, sha256_bytes(data), SUITE, corpus, as_str(top.get("instruction"), "instruction"),
                   {k: v for k, v in request.items() if k != "max_tokens"}, max_tokens, tasks)

    def materialize(self, corpus: Corpus) -> list[Materialized]:
        return [materialize(self, task, corpus) for task in self.tasks]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _doc(task: dict[str, object], corpus: Corpus, where: str) -> Doc:
    doc = corpus.get(as_str(task.get("doc_id"), f"{where}.doc_id"),
                     as_str(task.get("doc_sha256"), f"{where}.doc_sha256"))
    if task.get("licence") != doc.licence:
        raise TaskError(f"{where}: licence {task.get('licence')!r} is not the corpus class {doc.licence!r}")
    return doc


def _fill(template: str, slots: dict[str, str], where: str) -> str:
    names = set(re.findall(r"\{(\w+)\}", template))
    if names != set(slots):
        raise TaskError(f"{where}: template slots {sorted(names)} != {sorted(slots)}")
    return template.format(**slots)


def _reporter(item: object, where: str) -> tuple[str, str, str]:
    triple = as_list(item, where)
    if len(triple) != 3:
        raise TaskError(f"{where}: a reporter cite is [volume, reporter, first page]")
    return as_str(triple[0], where), as_str(triple[1], where), as_str(triple[2], where)


def materialize(suite: Suite, task: dict[str, object], corpus: Corpus) -> Materialized:
    task_id = as_str(task.get("id"), "task id")
    where = f"task {task_id}"
    family = as_str(task.get("family"), f"{where}.family")
    if family not in FAMILIES:
        raise TaskError(f"{where}: unknown family {family!r}")
    fam = cast(Family, family)
    key_obj = as_obj(task.get("key"), f"{where}.key")
    try:
        if fam == "needle":
            return _needle(suite, task, key_obj, corpus, task_id, where)
        doc = _doc(task, corpus, where)
        slots = {name: normalize(Span.parse(spec, f"{where}.slots.{name}").resolve(doc, f"{where}.slots.{name}"))
                 for name, spec in as_obj(task.get("slots", {}), f"{where}.slots").items()}
        question = _fill(as_str(task.get("question"), f"{where}.question"), slots, where)
        key: Key
        if fam == "quote":
            sentence = Span.parse(key_obj.get("sentence"), f"{where}.key.sentence").resolve(
                doc, f"{where}.key.sentence")
            key = QuoteKey(normalize(sentence))
        elif fam == "figure":
            kind = as_str(key_obj.get("kind"), f"{where}.key.kind")
            if kind not in ("money", "date", "percent"):
                raise TaskError(f"{where}: figure kind {kind!r}")
            raw = Span.parse(key_obj.get("figure"), f"{where}.key.figure").resolve(doc, f"{where}.key.figure")
            value = canonical_figure(cast(FigureKind, kind), raw)
            stated = key_obj.get("canonical")
            if stated is not None and stated != value:
                raise TaskError(f"{where}: canonical {stated!r} != {value!r}")
            key = FigureKey(cast(FigureKind, kind), value)
        elif fam == "citation":
            Span.parse(key_obj.get("citation"), f"{where}.key.citation").resolve(doc, f"{where}.key.citation")
            reporters = tuple(_reporter(item, f"{where}.key.reporters") for item in
                              as_list(key_obj.get("reporters"), f"{where}.key.reporters"))
            if not reporters:
                raise TaskError(f"{where}: no reporters")
            key = CitationKey(reporters, as_str(key_obj.get("year"), where), as_str(key_obj.get("court"), where),
                              key_obj.get("sc_style") is True)
        else:
            kind = as_str(key_obj.get("kind"), f"{where}.key.kind")
            if kind not in ("money", "date", "percent", "name"):
                raise TaskError(f"{where}: probe kind {kind!r}")
            lowered = normalize(render_doc(doc)).lower()
            for pattern in as_list(key_obj.get("absent"), f"{where}.key.absent"):
                if re.search(as_str(pattern, where), lowered):
                    raise TaskError(f"{where}: probe term {pattern!r} occurs in {doc.doc_id}")
            key = RefusalKey(cast(ProbeKind, kind))
        message = user_message(suite.instruction, render_doc(doc), question)
    except CorpusError as error:
        raise TaskError(f"{where}: {error}") from error
    return Materialized(task_id, fam, message, suite.max_tokens[fam], key)


def _needle(suite: Suite, task: dict[str, object], key_obj: dict[str, object], corpus: Corpus, task_id: str,
            where: str) -> Materialized:
    recipe = as_obj(task.get("haystack"), f"{where}.haystack")
    doc_ids = [(as_str(pair[0], where), as_str(pair[1], where))
               for pair in (as_list(item, where) for item in as_list(recipe.get("docs"), f"{where}.haystack.docs"))]
    hay = haystack(corpus, doc_ids, as_int(recipe.get("chars"), f"{where}.haystack.chars"))
    if sha256_text(hay) != recipe.get("sha256"):
        raise TaskError(f"{where}: haystack sha256 mismatch")
    at = as_int(task.get("insert_at"), f"{where}.insert_at")
    if not 0 < at < len(hay) or hay[at - 2:at] != "\n\n":
        raise TaskError(f"{where}: needle offset {at} is not just after a paragraph break")
    needle = as_str(task.get("needle"), f"{where}.needle")
    docket = as_str(key_obj.get("docket"), f"{where}.key.docket")
    if docket not in needle or docket in hay:
        raise TaskError(f"{where}: docket must be in the needle and absent from the haystack")
    message = user_message(suite.instruction, insert_needle(hay, at, needle), as_str(task.get("question"), where))
    return Materialized(task_id, "needle", message, suite.max_tokens["needle"], NeedleKey(docket))
