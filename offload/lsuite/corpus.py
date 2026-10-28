"""Pinned loader for the public bench corpus (`bench-data`: CourtListener opinions and RECAP filings).

Every file is checked against its pinned sha256 and every record against its own `sha256` (of the UTF-8 text) and
`chars`, so a task built today rebuilds byte-identical spans on any host that holds the same corpus. The corpus lives
outside git: Controller `/opt/dsv41/data/bench-data`, sparkhead `~/dsv41-deploy/data` (same pins as
`data/SHA256SUMS`).

Licence classes decide what may be stored in the repository:
  public-domain   court opinions (government edicts doctrine): source spans may be fixture data in the task file
  pinned-only     RECAP party filings ("never redistributed"): the task file holds sha256 pins and character offsets
                  only; every span is rebuilt from the corpus at run time
An unknown licence string fails loud, so a corpus change forces a fresh licence review.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

CORPUS_FILES: dict[str, str] = {
    "opinions_cl.jsonl": "80461203f88a6b837eed8fdf9b7e77a40660acb36dfa9887e00a0ae37483b6a1",
    "filings_recap.jsonl": "b1f2744634cddb0716074c451df5f46e2da3d851324ab794f0caff05ab7202a9",
}
OPINION_LICENCE_PREFIX = "public domain (court opinion; government edicts doctrine)"
FILING_LICENCE_MARK = "never redistributed"

Licence = Literal["public-domain", "pinned-only"]


class CorpusError(Exception):
    """The corpus is missing, altered or carries a licence this suite has not reviewed."""


@dataclass(frozen=True)
class Doc:
    doc_id: str
    collection: str
    group: str
    title: str
    court: str
    court_id: str
    jurisdiction: str
    date: str
    licence: Licence
    sha256: str
    body: str

    @property
    def redistributable(self) -> bool:
        return self.licence == "public-domain"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_licence(collection: str, licence: str) -> Licence:
    if collection == "opinions_cl" and licence.startswith(OPINION_LICENCE_PREFIX):
        return "public-domain"
    if collection == "filings_recap" and FILING_LICENCE_MARK in licence:
        return "pinned-only"
    raise CorpusError(f"unreviewed licence for collection {collection!r}: {licence[:80]!r}")


def _field(record: dict[str, object], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise CorpusError(f"{where}: field {key!r} is not a string")
    return value


def _parse(line: str, where: str) -> Doc:
    loaded: object = json.loads(line)
    if not isinstance(loaded, dict):
        raise CorpusError(f"{where}: record is not a JSON object")
    record: dict[str, object] = loaded
    body = _field(record, "text", where)
    digest = _field(record, "sha256", where)
    if sha256_text(body) != digest:
        raise CorpusError(f"{where}: text does not hash to its recorded sha256 {digest[:12]}")
    chars = record.get("chars")
    if chars != len(body):
        raise CorpusError(f"{where}: chars {chars!r} != text length {len(body)}")
    collection = _field(record, "collection", where)
    return Doc(
        doc_id=_field(record, "doc_id", where), collection=collection, group=_field(record, "group", where),
        title=_field(record, "title", where), court=_field(record, "court", where),
        court_id=_field(record, "court_id", where), jurisdiction=_field(record, "jurisdiction", where),
        date=_field(record, "date", where), licence=classify_licence(collection, _field(record, "license", where)),
        sha256=digest, body=body)


class Corpus:
    """The verified corpus, keyed by doc_id, in file order (opinions first, then filings)."""

    def __init__(self, root: Path, docs: list[Doc]) -> None:
        self.root = root
        self.docs = docs
        self._by_id: dict[str, Doc] = {}
        for doc in docs:
            if doc.doc_id in self._by_id:
                raise CorpusError(f"duplicate doc_id {doc.doc_id}")
            self._by_id[doc.doc_id] = doc

    @classmethod
    def load(cls, root: Path, pins: dict[str, str] | None = None) -> Corpus:
        """Verify and load `root`; `pins` (file name -> sha256) defaults to the production corpus pins."""
        docs: list[Doc] = []
        chosen = CORPUS_FILES if pins is None else pins
        for name, pinned in chosen.items():
            path = root / name
            if not path.is_file():
                raise CorpusError(f"{path}: missing (bench-data must hold {', '.join(chosen)})")
            got = sha256_file(path)
            if got != pinned:
                raise CorpusError(f"{path}: sha256 {got[:12]} is not the pinned corpus {pinned[:12]}")
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if line.strip():
                        docs.append(_parse(line, f"{name}:{number}"))
        return cls(root, docs)

    def get(self, doc_id: str, sha256: str) -> Doc:
        doc = self._by_id.get(doc_id)
        if doc is None:
            raise CorpusError(f"doc {doc_id} is not in the corpus")
        if doc.sha256 != sha256:
            raise CorpusError(f"doc {doc_id}: sha256 {doc.sha256[:12]} is not the pinned {sha256[:12]}")
        return doc

    def opinions(self) -> list[Doc]:
        return [doc for doc in self.docs if doc.collection == "opinions_cl"]

    def filings(self) -> list[Doc]:
        return [doc for doc in self.docs if doc.collection == "filings_recap"]
