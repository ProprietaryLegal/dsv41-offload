#!/usr/bin/env python3
"""Apply the pre-registered L-SUITE pass rule to a control run and a candidate run (rows only; no text).

  python3 -m offload.lsuite.compare <control run dir> <candidate run dir> [--json verdict.json]

Rule lsuite-rule-v1 (F-METHOD plan section 3.5, registered before any run):
  A task is LOST when the control passes it and the candidate fails it.
  FAIL      if the candidate loses any exact-quotation or needle task the control passes,
            or loses more than one task in the other families (figure, citation, refusal) combined.
  PASS      otherwise. Gains (candidate passes, control fails) are reported and never offset a loss.
  INVALID   if either run is partial or has a request error, or any row carries reasoning text (thinking was not
            off), or both directories are the same run, or the two runs differ in suite, task file, request settings,
            priority, corpus pins, scorer code (code_sha256), task set, or any task's sent prompt (request_sha256) or
            sent settings (body_sha256). An INVALID comparison decides nothing; rerun.
Exit 0 = PASS, 1 = FAIL, 2 = INVALID (or unreadable input).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from offload.lsuite.tasks import TaskError, as_obj

RULE = "lsuite-rule-v1"
RULE_TEXT = ("FAIL if the candidate loses any quote or needle task the control passes, or more than one task in "
             "figure/citation/refusal combined; otherwise PASS. Partial, errored, thinking-on, self-compared or "
             "mismatched runs are INVALID.")
STRICT = frozenset({"quote", "needle"})
SAME_SETTINGS = ("suite", "tasks_sha256", "request", "max_tokens", "priority", "corpus", "code_sha256")


@dataclass(frozen=True)
class Run:
    path: Path
    header: dict[str, object]
    summary: dict[str, object]
    rows: dict[str, dict[str, object]]


def load_run(path: Path) -> Run:
    try:
        header = as_obj(json.loads((path / "run.json").read_text(encoding="utf-8")), f"{path}/run.json")
        summary = as_obj(json.loads((path / "summary.json").read_text(encoding="utf-8")), f"{path}/summary.json")
        rows: dict[str, dict[str, object]] = {}
        for number, line in enumerate((path / "rows.jsonl").read_text(encoding="utf-8").splitlines(), start=1):
            row = as_obj(json.loads(line), f"{path}/rows.jsonl:{number}")
            task = row.get("task")
            if not isinstance(task, str) or task in rows:
                raise TaskError(f"{path}/rows.jsonl:{number}: missing or duplicate task id")
            rows[task] = row
    except (OSError, ValueError) as error:
        raise TaskError(f"{path}: unreadable run ({error})") from error
    return Run(path, header, summary, rows)


def invalid_reasons(control: Run, candidate: Run) -> list[str]:
    reasons: list[str] = []
    for run, label in ((control, "control"), (candidate, "candidate")):
        planned = run.header.get("planned")
        errors = [task for task, row in run.rows.items() if row.get("outcome") not in ("pass", "fail")]
        if run.summary.get("complete") is not True or len(run.rows) != planned:
            reasons.append(f"{label} run is partial ({len(run.rows)} of {planned} tasks)")
        if errors:
            reasons.append(f"{label} run has request errors: {sorted(errors)}")
        thinking = sorted(task for task, row in run.rows.items() if row.get("reasoning_chars", 0) != 0)
        if thinking:
            reasons.append(f"{label} run was not thinking-off (reasoning text) for {thinking}")
    if control.header.get("run_id") == candidate.header.get("run_id"):
        reasons.append("control and candidate are the same run")
    for key in SAME_SETTINGS:
        if control.header.get(key) != candidate.header.get(key):
            reasons.append(f"runs differ in {key}")
    if set(control.rows) != set(candidate.rows):
        reasons.append("runs cover different task sets")
    mismatched = sorted(task for task in set(control.rows) & set(candidate.rows)
                        if control.rows[task].get("request_sha256") != candidate.rows[task].get("request_sha256"))
    if mismatched:
        reasons.append(f"sent prompts differ for {mismatched}")
    shapes = sorted(task for task in set(control.rows) & set(candidate.rows)
                    if control.rows[task].get("body_sha256") != candidate.rows[task].get("body_sha256"))
    if shapes:
        reasons.append(f"sent request settings differ for {shapes}")
    return reasons


def compare(control: Run, candidate: Run) -> dict[str, object]:
    def passed(run: Run, task: str) -> bool:
        return run.rows[task].get("outcome") == "pass"

    def family(task: str) -> str:
        value = control.rows[task].get("family")
        return value if isinstance(value, str) else "?"

    tasks = sorted(set(control.rows) & set(candidate.rows))
    lost = [task for task in tasks if passed(control, task) and not passed(candidate, task)]
    gained = [task for task in tasks if passed(candidate, task) and not passed(control, task)]
    lost_strict = [task for task in lost if family(task) in STRICT]
    lost_other = [task for task in lost if family(task) not in STRICT]
    reasons = invalid_reasons(control, candidate)
    if reasons:
        verdict = "INVALID"
    elif lost_strict or len(lost_other) > 1:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    per_family: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts = per_family.setdefault(family(task), {"tasks": 0, "control_pass": 0, "candidate_pass": 0})
        counts["tasks"] += 1
        counts["control_pass"] += passed(control, task)
        counts["candidate_pass"] += passed(candidate, task)
    return {
        "verdict": verdict, "rule": RULE, "rule_text": RULE_TEXT, "invalid": reasons,
        "control": {"dir": str(control.path), "arm": control.header.get("arm"), "run_id": control.header.get("run_id"),
                    "pass": sum(passed(control, task) for task in tasks)},
        "candidate": {"dir": str(candidate.path), "arm": candidate.header.get("arm"),
                      "run_id": candidate.header.get("run_id"), "pass": sum(passed(candidate, task) for task in tasks)},
        "lost_strict": [(task, candidate.rows[task].get("reason")) for task in lost_strict],
        "lost_other": [(task, candidate.rows[task].get("reason")) for task in lost_other],
        "gained": gained, "families": per_family,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Apply {RULE} to two L-SUITE runs.")
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, help="also write the verdict as JSON")
    args = parser.parse_args(argv)
    try:
        result = compare(load_run(args.control), load_run(args.candidate))
    except TaskError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2
    print(f"{result['verdict']} ({RULE}): control {result['control']} candidate {result['candidate']}")
    for key in ("invalid", "lost_strict", "lost_other", "gained", "families"):
        print(f"  {key}: {result[key]}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return {"PASS": 0, "FAIL": 1}.get(str(result["verdict"]), 2)


if __name__ == "__main__":
    sys.exit(main())
