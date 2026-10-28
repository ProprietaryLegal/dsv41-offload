"""S23 probes of D2-FIX1 item 4 (reserved TEST): independent overlap screen and exclusion from generation/selection.

The shingle screen here is written independently of prompts.check_near_duplicates: lower-cased word 12-grams over
all message text, compared with every existing prompt (all splits) and the decoded acap capture prompts. Prints
fractions only, never text.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from d2 import generate as G

FROZEN = Path(os.environ.get('D2_RESERVED_TEST_ROOT',
                            '/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-FIX1-reserved-test'))
PRIV = Path(os.environ.get('D2_PRIVATE_TEST_ROOT', '/opt/dsv41/data/d2'))
real = pytest.mark.skipif(not (FROZEN / "prompts.jsonl").exists() or not (PRIV / "prompts/prompts.jsonl").exists(),
                          reason="private D2 inputs absent")
K = 12


def shingles(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + K]) for i in range(len(w) - K + 1)}


def text(p: dict) -> str:
    return "\n".join(m["content"] for m in p["messages"])


@real
def test_no_test_prompt_overlaps_existing_or_acap() -> None:
    from tokenizers import Tokenizer
    test = [json.loads(x) for x in open(FROZEN / "prompts.jsonl")]
    old = [json.loads(x) for x in open(PRIV / "prompts/prompts.jsonl")]
    tok = Tokenizer.from_file(str(PRIV / "tokenizer/tokenizer.json"))
    acap = []
    for f in sorted((PRIV / "acap").glob("*.json")):
        a = json.loads(f.read_text())
        acap.append(tok.decode(a["tokens"][:a["gen_start"]], skip_special_tokens=True))
    old_sh = [shingles(text(p)) for p in old] + [shingles(t) for t in acap]
    test_sh = [shingles(text(p)) for p in test]
    # instruction/frame boilerplate: shingles shared by >= 8 existing prompts (the pipeline's own rule) are ignored
    from collections import Counter
    freq = Counter(s for sh in old_sh for s in sh)
    boiler = {s for s, n in freq.items() if n >= 8}
    worst = []
    for sh in test_sh:
        content = sh - boiler
        worst.append(max(len(content & o) / max(1, len(content)) for o in old_sh))
    raw = [max(len(sh & o) / max(1, len(sh)) for o in old_sh) for sh in test_sh]
    print(f"S23 TEST overlap vs 538 existing + 4 acap: max content-shingle fraction {max(worst):.4f}, "
          f"max raw fraction {max(raw):.4f}")
    # MEASURED 2026-10-23: max 0.0749 (quoted statutory best-interest factors / standard-of-review text, not a
    # copy of the same document); the pipeline rule is 0.20 (S14 F8 found 0.32-0.987 for real near-copies)
    assert len(test) == 12 and max(worst) < 0.20
    assert not {d for p in test for d in p["doc_ids"]} & {d for p in old for d in p["doc_ids"]}
    assert not {d for p in test for d in p["source_docs"]} & {d for p in old for d in p["source_docs"]}
    assert not {p["prompt_sha256"] for p in test} & {p["prompt_sha256"] for p in old}
    excluded = set(json.loads((PRIV / "prompts/manifest.json").read_text())["excluded_acap_docs"])
    assert not {d for p in test for d in p["doc_ids"]} & excluded


@real
def test_test_split_is_never_loaded_by_default_generation() -> None:
    assert "test" not in G.ORDER
    assert G.load_prompts(str(FROZEN / "prompts.jsonl"), G.ORDER, 20261023) == []


def test_pilot_and_training_split_registries_have_no_test() -> None:
    root = Path(__file__).resolve().parents[1]
    hits = []
    for f in list((root / "train").glob("*.py")) + [root / "g1.py", root / "shards.py", root / "dataset.py"]:
        src = f.read_text()
        if re.search(r"""['"]test['"]""", src) or "reserved_test" in src or "D2-FIX1-reserved-test" in src:
            hits.append(f.name)
    assert hits == [], hits
