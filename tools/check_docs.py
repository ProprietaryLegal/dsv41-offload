#!/usr/bin/env python3
"""CPU-only checks of local Markdown links, brief coverage and Window-4 claims.

No network, git mutation, host probes or third-party dependencies. Fenced/inline
code and comments are excluded. Inline, image and defined reference links are
checked, including local heading fragments. Absolute paths and URI links are
outside the relative-link check. HTML href/src attributes are checked as well.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import html
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Link:
    target: str
    line: int


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


def visible_markdown(source: str) -> str:
    """Mask code/comments while preserving offsets for diagnostic line numbers."""
    result: list[str] = []
    fence = ""
    for line in source.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        masked = bool(fence) or bool(match) or line.startswith("    ")
        if match:
            marker = match[1]
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = ""
        result.append(re.sub(r"[^\n]", " ", line) if masked else line)
    text = "".join(result)
    text = re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S)
    return re.sub(
        r"(`+)(?!`)(.+?)(?<!`)\1(?!`)",
        lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S,
    )


def destination(text: str, start: int) -> tuple[str, int]:
    """Read an angle destination or balanced bare destination, before a title."""
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor < len(text) and text[cursor] == "<":
        end = text.find(">", cursor + 1)
        if end != -1:
            return text[cursor + 1:end], end + 1
    begin = cursor
    depth = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace() and depth == 0:
            break
        cursor += 1
    return text[begin:cursor], cursor


def reference_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def links(source: str) -> list[Link]:
    text = visible_markdown(source)
    refs: dict[str, str] = {}
    output: list[Link] = []
    for match in re.finditer(r"^ {0,3}\[([^]\n]+)\]:[ \t]*(.*)$", text, re.M):
        ref_target, _ = destination(match[2], 0)
        refs.setdefault(reference_key(match[1]), ref_target)
        output.append(Link(ref_target, text.count("\n", 0, match.start()) + 1))
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] != "[":
            cursor += 1
            continue
        begin = cursor
        depth = 1
        cursor += 1
        label_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "\\":
                cursor += 2
                continue
            depth += (text[cursor] == "[") - (text[cursor] == "]")
            cursor += 1
        if depth:
            # An unmatched outer bracket must not hide a later valid link.
            cursor = begin + 1
            continue
        label = text[label_start:cursor - 1]
        target: str | None = None
        if cursor < len(text) and text[cursor] == "(":
            target, end = destination(text, cursor + 1)
            # A remaining optional quoted title is allowed before the close.
            tail = re.match(r"\s*(?:\"[^\"]*\"|'[^']*'|\([^)]*\))?\s*\)", text[end:])
            if tail is None:
                target = None
            else:
                cursor = end + tail.end()
        elif cursor < len(text) and text[cursor] == "[":
            end = text.find("]", cursor + 1)
            if end != -1:
                key = text[cursor + 1:end] or label
                target = refs.get(reference_key(key))
                cursor = end + 1
        elif cursor >= len(text) or text[cursor] != ":":
            target = refs.get(reference_key(label))
        if target is not None:
            output.append(Link(target, text.count("\n", 0, begin) + 1))
    for match in re.finditer(r"\b(?:href|src)\s*=\s*([\"'])(.*?)\1", text, re.I):
        output.append(Link(match[2], text.count("\n", 0, match.start()) + 1))
    return output


def local_target(path: Path, target: str) -> tuple[Path, str] | None:
    target = html.unescape(re.sub(r"\\([\W_])", r"\1", target))
    if target.startswith(("/", "\\")) or re.match(r"^[A-Za-z][\w+.-]*:", target):
        return None
    parts = urlsplit(target)
    return (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve(), unquote(parts.fragment)


def heading_ids(source: str) -> set[str]:
    # Keep inline code TEXT when building GitHub-style heading ids.
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    text = source
    ids: set[str] = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", source))
    counts: dict[str, int] = {}
    fence = ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
            continue
        if fence:
            continue
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$", line)
        title = match[1] if match else ""
        if not title and i + 1 < len(lines) and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[i + 1]):
            title = line.strip()
        if not title:
            continue
        title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", title)
        title = re.sub(r"<[^>]*>", "", title)
        slug = re.sub(r"[^\w\- ]", "", html.unescape(title).lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        candidate = slug + (f"-{count}" if count else "")
        while candidate in ids:
            count += 1
            candidate = f"{slug}-{count}"
        counts[slug] = count + 1
        ids.add(candidate)
    return ids


def check_links(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    anchors: dict[Path, set[str]] = {}
    for path in paths:
        for link in links(path.read_text()):
            resolved = local_target(path, link.target)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.exists():
                findings.append(Finding(path, link.line, f"missing relative target: {link.target}"))
            elif fragment and target.is_file() and target.suffix.lower() == ".md":
                if target not in anchors:
                    anchors[target] = heading_ids(target.read_text())
                if fragment not in anchors[target]:
                    findings.append(Finding(path, link.line, f"missing heading fragment: {link.target}"))
    return findings


def check_briefs(root: Path) -> list[Finding]:
    index = root / "documents/README.md"
    if not index.is_file():
        return [Finding(index, 1, "missing lane index")]
    indexed = {resolved[0] for link in links(index.read_text()) if (resolved := local_target(index, link.target))}
    return [Finding(index, 1, f"lane brief missing from index: {path.relative_to(root)}")
            for path in sorted((root / "documents/plans/lanes").glob("*.md")) if path.resolve() not in indexed]


def percent(value: str) -> Decimal:
    return Decimal(value.strip().removesuffix("%").strip().replace("−", "-"))


def window4_cells(comparison: str) -> dict[str, set[Decimal]]:
    window = comparison.split("## Window 4 (", 1)[1].split("## Window 5 (", 1)[0]
    headings = {"one-card": "### W4-sync:", "tp2-async": "### W4-E1p:", "tp2-sync": "### W4-E0:"}
    output: dict[str, set[Decimal]] = {}
    for arm, heading in headings.items():
        section = window.split(heading, 1)[1].split("\n### ", 1)[0]
        cells = [line.strip().strip("|").split("|")[3] for line in section.splitlines()
                 if line.startswith("| ") and len(line.split("|")) > 5 and "%" in line]
        output[arm] = {percent(cell) for cell in cells}
        if len(cells) != 4:
            raise ValueError(f"COMPARISON Window 4 {arm}: expected four percentage rows, found {len(cells)}")
    return output


def claim_arm(prefix: str) -> str | None:
    labels = list(re.finditer(
        r"one.card(?:\s+(?:async|process))?|tp2\s+(?:\+\s*)?(?:async|sync)|w4-(?:p1|e1p|e0)", prefix, re.I))
    if labels:
        label = labels[-1][0].lower()
        if "one" in label or "p1" in label:
            return "one-card"
        return "tp2-sync" if "sync" in label and "async" not in label or "e0" in label else "tp2-async"
    if "tp2" in prefix.lower() and re.search(r"async|process proxy", prefix, re.I):
        return "tp2-async"
    return None


def improvement(value: str) -> Decimal:
    """Compare improvement magnitudes regardless of signed-change prose style."""
    return -abs(percent(value))


def check_window4(root: Path, readmes: list[Path]) -> list[Finding]:
    comparison = root / "offload/results/COMPARISON.md"
    try:
        canonical = window4_cells(comparison.read_text())
    except (OSError, IndexError, ValueError) as error:
        return [Finding(comparison, 1, f"cannot read canonical Window-4 tables: {error}")]
    findings: list[Finding] = []
    ranges = re.compile(r"([−-]?\d+(?:\.\d+)?)\s*%?\s*(?:to|[–…]|\.\.)\s*([−-]?\d+(?:\.\d+)?)\s*%")
    all_values = set().union(*canonical.values())
    for path in readmes:
        text = visible_markdown(path.read_text())
        for block in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.S):
            paragraph = block[0]
            if not re.search(r"window[ -]?4|w4-(?:p1|e1p|e0)", paragraph, re.I):
                continue
            claimed_ranges: list[tuple[int, int]] = []
            for match in ranges.finditer(paragraph):
                arm = claim_arm(paragraph[:match.start()])
                values = {improvement(match[1]), improvement(match[2])}
                expected = ({min(canonical[arm]), max(canonical[arm])} if arm else None)
                valid = values == expected if expected else any(values == {min(v), max(v)} for v in canonical.values())
                if not valid:
                    line = text.count("\n", 0, block.start() + match.start()) + 1
                    findings.append(Finding(path, line, f"Window-4 range {match[0]} disagrees with COMPARISON ({arm or 'unlabelled arm'})"))
                claimed_ranges.append(match.span())
            for match in re.finditer(r"(?<![\w.])([+−-]?\d+(?:\.\d+)?)\s*%", paragraph):
                if any(start <= match.start() < end for start, end in claimed_ranges):
                    continue
                arm = claim_arm(paragraph[:match.start()])
                if improvement(match[1]) not in (canonical[arm] if arm else all_values):
                    line = text.count("\n", 0, block.start() + match.start()) + 1
                    findings.append(Finding(path, line, f"Window-4 percentage {match[0]} disagrees with COMPARISON ({arm or 'unlabelled arm'})"))
    return findings


def check(root: Path) -> tuple[list[Finding], int]:
    paths = sorted(p for p in root.rglob("*.md") if not any(part in {".git", ".venv", "__pycache__"} for part in p.relative_to(root).parts))
    return check_links(paths) + check_briefs(root) + check_window4(root, [p for p in paths if p.name == "README.md"]), len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    findings, count = check(root)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.message}")
    print(f"{'FAIL' if findings else 'PASS'}: {count} Markdown files; relative targets/fragments, lane index, Window-4 README claims; {len(findings)} issue(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
