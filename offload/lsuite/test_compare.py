"""The pre-registered pass rule (lsuite-rule-v1) on hand-made rows, plus one end-to-end runner pair."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.lsuite import compare as K
from offload.lsuite import run_suite as R
from offload.lsuite.conftest import FakeServer, oracle_reply, run_args as args
from offload.lsuite.tasks import Materialized, QuoteKey, Suite

FAMILY_OF = {"quote-01": "quote", "quote-02": "quote", "needle-01": "needle", "figure-01": "figure",
             "figure-02": "figure", "citation-01": "citation", "refusal-01": "refusal"}


def write_run(root: Path, name: str, failing: frozenset[str] | set[str] = frozenset(), *,
              errors: frozenset[str] | set[str] = frozenset(), drop: frozenset[str] | set[str] = frozenset(),
              header: dict[str, object] | None = None, prompt_of: dict[str, str] | None = None,
              body_of: dict[str, str] | None = None, reasoning: frozenset[str] | set[str] = frozenset()) -> Path:
    out = root / name
    out.mkdir()
    base: dict[str, object] = {"suite": "lsuite-v1", "tasks_sha256": "a" * 64, "request": {"temperature": 0},
                               "max_tokens": {"quote": 256}, "priority": 10, "corpus": {"f": "b" * 64},
                               "code_sha256": "c" * 64, "planned": len(FAMILY_OF), "arm": name, "run_id": name}
    (out / "run.json").write_text(json.dumps(base | (header or {})), encoding="utf-8")
    rows = []
    for task, family in FAMILY_OF.items():
        if task in drop:
            continue
        outcome = "error" if task in errors else "fail" if task in failing else "pass"
        rows.append({"task": task, "family": family, "outcome": outcome, "reason": "r",
                     "request_sha256": (prompt_of or {}).get(task, task),
                     "body_sha256": (body_of or {}).get(task, "shape"),
                     "reasoning_chars": 400 if task in reasoning else 0})
    (out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    complete = not drop and not errors
    (out / "summary.json").write_text(json.dumps({"complete": complete}), encoding="utf-8")
    return out


def verdict(control: Path, candidate: Path) -> dict[str, object]:
    return K.compare(K.load_run(control), K.load_run(candidate))


@pytest.mark.parametrize(("control_fails", "candidate_fails", "expected"), [
    (set(), set(), "PASS"),
    (set(), {"quote-01"}, "FAIL"),
    (set(), {"needle-01"}, "FAIL"),
    (set(), {"figure-01"}, "PASS"),
    (set(), {"refusal-01"}, "PASS"),
    (set(), {"figure-01", "citation-01"}, "FAIL"),
    (set(), {"figure-01", "figure-02"}, "FAIL"),
    ({"quote-01"}, {"quote-01"}, "PASS"),                     # control failed it too: not a loss
    ({"quote-01", "figure-01"}, {"figure-02"}, "PASS"),       # gains never offset, but one other loss is allowed
    ({"quote-01", "figure-01"}, {"figure-02", "citation-01"}, "FAIL"),
])
def test_rule(tmp_path: Path, control_fails: set[str], candidate_fails: set[str], expected: str) -> None:
    result = verdict(write_run(tmp_path, "C", control_fails), write_run(tmp_path, "X", candidate_fails))
    assert result["verdict"] == expected
    assert result["rule"] == "lsuite-rule-v1"


def test_lost_and_gained_lists(tmp_path: Path) -> None:
    result = verdict(write_run(tmp_path, "C", {"figure-01"}), write_run(tmp_path, "X", {"quote-02", "citation-01"}))
    assert result["lost_strict"] == [("quote-02", "r")]
    assert result["lost_other"] == [("citation-01", "r")]
    assert result["gained"] == ["figure-01"]


@pytest.mark.parametrize(("kwargs", "reason"), [
    ({"drop": {"figure-02"}}, "partial"),
    ({"errors": {"figure-02"}}, "request errors"),
    ({"header": {"tasks_sha256": "c" * 64}}, "tasks_sha256"),
    ({"header": {"priority": 0}}, "priority"),
    ({"header": {"request": {"temperature": 1}}}, "request"),
    ({"prompt_of": {"quote-01": "other"}}, "sent prompts differ"),
    # S101 FIX 6 repros: what was actually sent, whether thinking stayed off, and which scorer ran
    ({"body_of": {"figure-02": "other-shape"}}, "sent request settings differ"),
    ({"reasoning": {"citation-01"}}, "not thinking-off"),
    ({"header": {"code_sha256": "d" * 64}}, "code_sha256"),
])
def test_invalid(tmp_path: Path, kwargs: dict[str, object], reason: str) -> None:
    result = verdict(write_run(tmp_path, "C"), write_run(tmp_path, "X", **kwargs))  # type: ignore[arg-type]
    assert result["verdict"] == "INVALID"
    invalid = result["invalid"]
    assert isinstance(invalid, list) and any(reason in item for item in invalid)


def test_same_run_twice_is_invalid(tmp_path: Path) -> None:
    """S101 FIX 6: passing one run directory as both arms used to PASS."""
    control = write_run(tmp_path, "C")
    result = verdict(control, control)
    invalid = result["invalid"]
    assert result["verdict"] == "INVALID" and isinstance(invalid, list)
    assert "control and candidate are the same run" in invalid
    assert K.main([str(control), str(control)]) == 2


def test_exit_codes(tmp_path: Path) -> None:
    control = write_run(tmp_path, "C")
    assert K.main([str(control), str(write_run(tmp_path, "X"))]) == 0
    assert K.main([str(control), str(write_run(tmp_path, "X2", {"quote-01"})), "--json",
                   str(tmp_path / "v.json")]) == 1
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["verdict"] == "FAIL"
    assert K.main([str(control), str(write_run(tmp_path, "X3", drop={"quote-01"}))]) == 2
    assert K.main([str(control), str(tmp_path / "missing")]) == 2


def test_end_to_end_with_the_runner(synthetic: tuple[Path, Path, list[Materialized]],
                                    tmp_path: Path) -> None:
    bench, tasks, items = synthetic
    control = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "C"), **control.io()) == 0

    def drops_a_word(item: Materialized) -> str:
        return item.key.sentence.replace("careful ", "") if isinstance(item.key, QuoteKey) else oracle_reply(item.key)

    candidate = FakeServer(items, drops_a_word)
    assert R.main(args(bench, tasks, tmp_path / "X"), **candidate.io()) == 0
    assert K.main([str(tmp_path / "C"), str(tmp_path / "X")]) == 1
    same = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "X2"), **same.io()) == 0
    assert K.main([str(tmp_path / "C"), str(tmp_path / "X2")]) == 0


def test_s101_patched_sender_with_reasoning_is_invalid(synthetic: tuple[Path, Path, list[Materialized]],
                                                       tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S101 FIX 6 repro: an arm that sent other settings (4x max_tokens, thinking on, one constant salt, a repetition
    penalty) to a server that returned reasoning text used to compare as PASS against a clean control."""
    bench, tasks, items = synthetic
    control = FakeServer(items, lambda item: oracle_reply(item.key))
    assert R.main(args(bench, tasks, tmp_path / "C"), **control.io()) == 0
    original = R.request_body

    def patched(suite: Suite, item: Materialized, model: str, salt: str, priority: int) -> dict[str, object]:
        body = original(suite, item, model, "constant-salt", priority)
        body.update(max_tokens=item.max_tokens * 4, chat_template_kwargs={"thinking": True}, repetition_penalty=1.3)
        return body

    monkeypatch.setattr(R, "request_body", patched)
    candidate = FakeServer(items, lambda item: oracle_reply(item.key))
    candidate.reasoning = "r" * 400
    assert R.main(args(bench, tasks, tmp_path / "X"), **candidate.io()) == 0
    result = verdict(tmp_path / "C", tmp_path / "X")
    invalid = result["invalid"]
    assert result["verdict"] == "INVALID" and isinstance(invalid, list)
    assert any("not thinking-off" in reason for reason in invalid)
    assert any("sent request settings differ" in reason for reason in invalid)
