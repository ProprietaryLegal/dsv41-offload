"""Aggregate unchanged V1 pair verdicts; private tensors never enter this output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

ARMS = ("native-hybrid-kx2", "native-packed")
KEYS = [f"r{r:06d}_b0" for r in range(8, 20)]


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object mapping")
    return cast(dict[str, object], value)


def record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], value)


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric aggregate, got {value!r}")
    return float(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--verify-root", type=Path, required=True)
    ap.add_argument("--pair", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n4 = Path(__file__).parents[1] / "n4/evidence/v1-summary.json"
    pins = record(n4)["source_sha256"]
    if not isinstance(pins, dict):
        raise ValueError("missing N4 source pins")
    sources = {name: a.verify_root / "verify" / name for name in pins if name != "N3-wrapper"}
    sources["N3-wrapper"] = Path(__file__).parents[1] / "n3/classify.py"
    hashes = {name: sha(path) for name, path in sources.items()}
    if hashes != pins:
        raise ValueError(f"V1/N3 source drift against N4: {hashes}")
    runs: list[dict[str, object]] = []
    for arm in ARMS:
        pair = record(a.campaign / f"N5-{arm}-gates.json")
        solo = record(a.campaign / f"N4-v2-{arm}-gates.json")
        raw = record(a.campaign / f"N5-{arm}-gates.classification.json")
        receipt = record(a.pair / f"{arm}-capture/capture-receipt.json")
        if pair["requested_keys"] != KEYS or solo["requested_keys"] != KEYS or receipt["keys"] != KEYS:
            raise ValueError(f"{arm}: requested key mismatch")
        if raw["reference"] != "mirror" or raw["band"] != "mirror64" or raw["representation"] != "native":
            raise ValueError(f"{arm}: classifier settings changed")
        fields = ("pass", "gates", "samples", "state_defects", "unexplained", "s0_attn_excess",
                  "kl_median", "summation_band_kl_median")
        if pair["samples"] != 12 or raw["samples"] != 12:
            raise ValueError(f"{arm}: incomplete scope")
        runs.append({"arm": arm, "identity": receipt["identity"],
                     "pair": {k: pair[k] for k in fields}, "solo": {k: solo[k] for k in fields},
                     "pair_equals_solo_aggregates": all(pair[k] == solo[k] for k in fields),
                     "classes": raw["classes"], "routing_set_flips": raw["routing_set_flips"],
                     "kl_over_band": number(pair["kl_median"]) / number(pair["summation_band_kl_median"]),
                     "wrapper_exit": int((a.campaign / f"N5-{arm}-classify.exit").read_text()),
                     "classification_sha256": sha(a.campaign / f"N5-{arm}-gates.classification.json")})
    out = {"scope": "12 early B1 keys from one request; not full or held-out qualification",
           "keys": KEYS, "reference": "mirror", "band": "mirror64", "representation": "native",
           "unchanged_n4_source_sha256": hashes, "strict_v1_gates_unchanged": True, "runs": runs}
    a.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
