"""Fail before a coordinator pair run if its staged source/receipt is stale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...identity import compute
from ....spark.tools.identity import identities


def verify(receipt: Path, expected_path: Path) -> None:
    d, m, files = identities(json.loads(receipt.read_text()))
    expected = json.loads(expected_path.read_text())
    for arm, want in expected.items():
        got = compute(arm, dict(drafter_identity=d, markov_identity=m, files=files))
        if any(got[k] != want[k] for k in ("drafter_identity", "markov_identity")):
            raise RuntimeError(f"{arm}: staged arithmetic/source identity differs; regenerate configs and pair expectations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    a = ap.parse_args()
    verify(a.receipt, a.expected)
    print("all four native identities match frozen N4 pair expectations")


if __name__ == "__main__":
    main()
