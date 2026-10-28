"""S23 probes of D2-FIX1 item 3: a served-data build needs complete identity receipts and refuses guesses."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from d2 import dataset as D
from d2.tests.test_dataset import _fake_replay, _gate, _recs
from d2.tests.test_s16_fixes import _ns
from d2.tests.test_sequence_audit import context_fixture


def served(tmp_path: Path):
    replay = _fake_replay(tmp_path, _recs())
    pm_path = replay / "prompt-manifest.json"
    pm = json.loads(pm_path.read_text())
    ctx, inputs = context_fixture(tmp_path)
    pm.update(schema="d2.prompts.v1", **inputs)
    for p in pm["prompts"]:
        p["source"] = "public_fixture"
    pm_path.write_text(json.dumps(pm))
    return replay, ctx


def build(tmp_path: Path, replay: Path, ctx: dict, name: str) -> Path:
    cp = tmp_path / f"{name}.context.json"
    cp.write_text(json.dumps(ctx))
    out = tmp_path / f"shards-{name}"
    D.cmd_build(_ns(replay, out, sequence_context=str(cp), equiv_gate=_gate(tmp_path, True, replay)))
    return out


def test_served_build_records_audit_and_refuses_each_missing_receipt(tmp_path: Path) -> None:
    replay, ctx = served(tmp_path)
    out = build(tmp_path, replay, ctx, "ok")
    side = json.loads(next((out / "train").glob("*.json")).read_text())
    a = side["sequences"][0]["sequence_audit"]
    assert a["schema"] == "d2.sequence-audit.v1" and a["label_policy"]["prompt_labels"] is False
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["generation"]["synthetic"] is False and man["checks"]["split_audit"]["exact_ids_checked"]
    mutations = {
        "no seed key": lambda c: c["sampling"].pop("seed"),
        "T != 1": lambda c: c["sampling"].update(temperature=0.6),
        "replay target differs": lambda c: c["replay_target"].update(quantization="other"),
        "tools unrecorded": lambda c: c["formatting"].pop("tools"),
        "tokenizer file swapped": lambda c: c.update(tokenizer_sha256="0" * 64),
        "marker guessed": lambda c: c["markers"].update(dsml=99),
        "ctk differs": lambda c: c["formatting"].update(chat_template_kwargs={"thinking": False}),
    }
    for name, mutate in mutations.items():
        bad = deepcopy(ctx)
        mutate(bad)
        with pytest.raises(SystemExit, match="provenance refused"):
            build(tmp_path, replay, bad, name.replace(" ", "-"))
        assert not (tmp_path / f"shards-{name.replace(' ', '-')}").exists(), name


def test_segment_sampling_contradicting_receipt_is_not_published(tmp_path: Path) -> None:
    replay, ctx = served(tmp_path)
    plan = [json.loads(x) for x in open(replay / "plan.jsonl")]
    src = Path(plan[0]["src"])
    st = json.loads(src.read_text())
    st["segments"] = [{"seg": 0, "start": 0, "n": len(st["gen_ids"]), "priority": 20,
                       "generator_revision": "d2-priority20-v1", "sampling": {"temperature": 1.0, "top_p": 0.95}}]
    src.write_text(json.dumps(st))
    with pytest.raises(SystemExit):
        build(tmp_path, replay, ctx, "seg")
    assert not (tmp_path / "shards-seg").exists()
