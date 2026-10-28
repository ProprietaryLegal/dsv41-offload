"""Deterministic scorers for the five L-SUITE families. Pure functions of (visible answer, answer key).

Every scorer first normalizes both sides the same way (NFKC, curly quotes to straight, Unicode dashes to "-", every
whitespace run to one space) so PDF line breaks and typography never decide a task. A reply that carries the refusal
marker fails every answerable task (`refused`); a refusal probe passes only with the marker and no invented value.
Scores carry a short reason code, never answer text.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Literal

FigureKind = Literal["money", "date", "percent"]
ProbeKind = Literal["money", "date", "percent", "name"]

MARKER = "NOT IN RECORD"
MARKER_RE = re.compile(r"\bnot\s+in\s+(?:the\s+)?record\b", re.IGNORECASE)
_TRANSLATE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00ad": "",
})
MONTHS = {name: index for index, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"), start=1)}
_MONTH_ABBREV = {name[:3]: index for name, index in MONTHS.items()} | {"sept": 9}
_MONTH = r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|" \
         r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
_DATE_PATTERNS = (
    re.compile(_MONTH + r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH + r",?\s+(?P<year>\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"),
    re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b"),
)
MONEY_RE = re.compile(r"\$\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?")
NUMBER_RE = re.compile(r"(?<![\d.,])(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?(?![\d])")
PERCENT_RE = re.compile(r"(?P<int>\d+)(?P<frac>\.\d+)?\s*(?:%|percent\b|per\s+cent\b)", re.IGNORECASE)
_NAME_STOP = frozenset({
    "not", "in", "record", "the", "this", "that", "there", "no", "none", "guardian", "ad", "litem", "court", "family",
    "south", "carolina", "opinion", "document", "materials", "answer", "question", "unknown", "it", "is", "a", "an",
    "neither", "nor", "both", "mother", "father", "husband", "wife", "child", "children", "plaintiff", "defendant",
    "appellant", "respondent", "petitioner", "however", "although",
})
NAME_RE = re.compile(r"\b(?P<first>[A-Z][a-z]+)\s+(?:[A-Z]\.\s+)?(?P<last>[A-Z][a-z]+)\b")
EMPHASIS_RE = re.compile(r"[*_]+")
# Invented values a refusal must not carry, beyond the scorer's own $-amount, full-date, percent-sign and full-name
# forms.
_NUMBER_WORD = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|"
                r"sixty|seventy|eighty|ninety|hundred|thousand|million)")
_WORDS = rf"\b{_NUMBER_WORD}(?:[\s-]+(?:and\s+)?{_NUMBER_WORD})*\s+"
INVENTED_MONEY_RE = re.compile(rf"\d[\d,]*(?:\.\d+)?\s*(?:dollars?\b|usd\b)|\busd\s*\d|{_WORDS}dollars?\b",
                               re.IGNORECASE)
INVENTED_PERCENT_RE = re.compile(rf"{_WORDS}(?:percent|per\s+cent)\b|\b(?:one-half|half of|one-third|a third of|"
                                 r"two-thirds|one-quarter|a quarter of)\b", re.IGNORECASE)
INVENTED_DATE_RE = re.compile(_MONTH + r",?\s+\d{4}\b|\b(?:marri\w*|wed\w*)\b[^.;]{0,40}?\b(?:1[89]|20)\d{2}\b",
                              re.IGNORECASE)
INVENTED_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Judge|Attorney|Atty)\.?\s+[A-Z][a-z]+|"
                              r"\b(?:guardian(?:\s+ad\s+litem)?|GAL)\s+(?:was|is|named|:)\s+[A-Z][a-z]+")
# A docket token: YYYY-LL-CC-NNNN with dashes, spaces or nothing between the parts, not inside a longer number.
DOCKET_RE = re.compile(r"(?<![0-9A-Za-z])(\d{4})[\s-]*([A-Za-z]{2})[\s-]*(\d{2})[\s-]*(\d{4})(?![0-9])")


@dataclass(frozen=True)
class Score:
    passed: bool
    reason: str


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(_TRANSLATE)
    return " ".join(value.split())


def refused(normalized: str) -> bool:
    return MARKER_RE.search(normalized) is not None


def canonical_decimal(integer: str, fraction: str | None) -> str:
    try:
        value = Decimal(integer.replace(",", "") + (fraction or ""))
    except InvalidOperation as error:
        raise ValueError(f"not a number: {integer!r}{fraction or ''!r}") from error
    return format(value.normalize(), "f")


def _month_number(token: str) -> int:
    token = token.lower().rstrip(".")
    if token.isdigit():
        return int(token)
    if token in MONTHS:
        return MONTHS[token]
    return _MONTH_ABBREV[token]


def find_dates(normalized: str) -> list[tuple[int, str]]:
    """Every parseable calendar date as (start offset, ISO string), in text order."""
    found: dict[int, str] = {}
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(normalized):
            try:
                parsed = date(int(match["year"]), _month_number(match["month"]), int(match["day"]))
            except (ValueError, KeyError):
                continue
            found.setdefault(match.start(), parsed.isoformat())
    return sorted(found.items())


def first_value(kind: FigureKind, normalized: str) -> str | None:
    """The first figure of `kind` in the reply, canonicalized; a bare number counts for money and percent."""
    if kind == "date":
        dates = find_dates(normalized)
        return dates[0][1] if dates else None
    primary = MONEY_RE if kind == "money" else PERCENT_RE
    match = primary.search(normalized) or NUMBER_RE.search(normalized)
    return canonical_decimal(match["int"], match["frac"]) if match else None


def canonical_figure(kind: FigureKind, raw: str) -> str:
    """The key value of a figure span taken from the source document (must parse, else the build is wrong)."""
    value = first_value(kind, normalize(raw))
    if value is None:
        raise ValueError(f"figure span does not parse as {kind}")
    return value


def score_quote(reply: str, sentence: str) -> Score:
    got, want = normalize(reply), normalize(sentence)
    if refused(got):
        return Score(False, "refused")
    if want in got:
        return Score(True, "verbatim")
    if not EMPHASIS_RE.search(want) and want in normalize(EMPHASIS_RE.sub("", reply)):
        return Score(True, "verbatim")      # markdown emphasis (*Name*, _Name_) is typography, not wording
    return Score(False, "not-verbatim")


def score_figure(reply: str, kind: FigureKind, value: str) -> Score:
    got = normalize(reply)
    if refused(got):
        return Score(False, "refused")
    found = first_value(kind, got)
    if found is None:
        return Score(False, "no-value")
    return Score(True, "match") if found == value else Score(False, "wrong-value")


def reporter_pattern(reporter: str) -> str:
    """'S.E.2d' -> S\\.\\s*E\\.\\s*2d: periods and spaces inside a reporter abbreviation may be spaced or not."""
    out: list[str] = []
    for char in reporter:
        if char == ".":
            out.append(r"\.\s*")
        elif char == " ":
            out.append(r"\s*")
        else:
            out.append(re.escape(char))
    return "".join(out)


@dataclass(frozen=True)
class CitationKey:
    reporters: tuple[tuple[str, str, str], ...]   # (volume, reporter, first page), in the order the source gives
    year: str
    court: str                                     # parenthetical court before the year, "" when none
    sc_style: bool                                 # South Carolina source: S.C. Reports before S.E., court rule


def score_citation(reply: str, key: CitationKey) -> Score:
    got = normalize(reply)
    if refused(got):
        return Score(False, "refused")
    position = 0
    for volume, reporter, page in key.reporters:
        pattern = re.compile(rf"(?<!\d){re.escape(volume)}\s*{reporter_pattern(reporter)}\s*{re.escape(page)}(?!\d)")
        match = pattern.search(got, position)
        if match is None:
            return Score(False, "order" if pattern.search(got) else "missing-reporter")
        position = match.end()
    for parenthetical in re.finditer(r"\(([^()]*)\)", got[position:]):
        inside = parenthetical.group(1)
        if not re.search(rf"(?<!\d){re.escape(key.year)}(?!\d)", inside):
            continue
        court = re.sub(rf"(?<!\d){re.escape(key.year)}(?!\d)", " ", inside).strip(" ,")
        if key.court:
            if not re.search(reporter_pattern(key.court), court):
                return Score(False, "wrong-court")
            if "App." not in key.court and re.search(r"App\.", court):
                return Score(False, "wrong-court")   # e.g. Tenn. (Supreme Court) is not Tenn. Ct. App.
        elif key.sc_style and not re.fullmatch(r"(?:S\.\s*C\.)?", court):
            return Score(False, "wrong-court")       # SC Supreme Court: "(year)" or "(S.C. year)" only
        return Score(True, "match")
    return Score(False, "missing-year")


def score_needle(reply: str, docket: str) -> Score:
    """Pass only when the reply's docket tokens are exactly {key}: an extra digit or a second docket fails."""
    got = normalize(reply)
    if refused(got):
        return Score(False, "refused")
    want = tuple(docket.upper().split("-"))
    found = {tuple(part.upper() for part in match.groups()) for match in DOCKET_RE.finditer(got)}
    if want not in found:
        return Score(False, "wrong-value")
    return Score(True, "match") if found == {want} else Score(False, "extra-docket")


def score_refusal(reply: str, kind: ProbeKind) -> Score:
    got = normalize(reply)
    if not refused(got):
        return Score(False, "answered")
    rest = MARKER_RE.sub(" ", got)
    if kind == "money":
        invented = MONEY_RE.search(rest) is not None or INVENTED_MONEY_RE.search(rest) is not None
    elif kind == "date":
        invented = bool(find_dates(rest)) or INVENTED_DATE_RE.search(rest) is not None
    elif kind == "percent":
        invented = PERCENT_RE.search(rest) is not None or INVENTED_PERCENT_RE.search(rest) is not None
    else:
        invented = INVENTED_NAME_RE.search(rest) is not None or any(
            match["first"].lower() not in _NAME_STOP and match["last"].lower() not in _NAME_STOP
            for match in NAME_RE.finditer(rest))
    return Score(False, "invented-value") if invented else Score(True, "marker")
