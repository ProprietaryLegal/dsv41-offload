from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

from offload.dspark.d2.train import pilot, pilot_eval, safety


def test_documented_shell_commands_have_valid_bash_syntax() -> None:
    root = Path(__file__).parents[1]
    overview = (root/"PILOT.md").read_text()
    # Executable procedures now live in these two authoritative runbooks.
    assert "[PILOT-STAGE.md](PILOT-STAGE.md)" in overview
    assert "[FULL-RUN.md](FULL-RUN.md)" in overview
    for name, expected in (("PILOT-STAGE.md", 2), ("FULL-RUN.md", 4)):
        blocks = re.findall(r"```bash\n(.*?)```", (root/name).read_text(), re.S)
        assert len(blocks) == expected
        for block in blocks:
            subprocess.run(["bash", "-n"], input=block, text=True, check=True)


def test_sample_cli_dispatches_without_a_gpu_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path, dict[str, Path]]] = []
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: pytest.fail("CPU sample attempted GPU lease"))
    monkeypatch.setattr(pilot, "sample_plan", lambda plan, manifest, k2: calls.append((plan, manifest, k2)))
    monkeypatch.setattr(sys, "argv", ["pilot", "sample", "--plan", "plan.json", "--manifest", "MANIFEST.json",
                                    "--k2-dev", "dev.jsonl", "--k2-legal", "legal.jsonl"])
    pilot.main()
    assert calls == [(Path("plan.json"), Path("MANIFEST.json"), {"dev": Path("dev.jsonl"), "heldout_legal": Path("legal.jsonl")})]


@pytest.mark.parametrize("split", ["dev", "heldout_legal"])
def test_reference_split_job_cli_routes_to_the_requested_split(split: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pilot_eval, "evaluate", lambda args: calls.append(args.split))
    monkeypatch.setattr(sys, "argv", ["pilot_eval", "--plan", "plan.json", "--manifest", "MANIFEST.json",
                                    "--root", "arms", "--ckpt", "native", "--k2-dev", "dev.jsonl",
                                    "--k2-legal", "legal.jsonl", "--split", split, "--out", split+".part.json"])
    pilot_eval.main()
    assert calls == [split]


def test_merge_cli_requires_no_model_paths_or_gpu_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, list[Path], Path]] = []
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: pytest.fail("CPU merge attempted GPU lease"))
    monkeypatch.setattr(pilot_eval, "evaluate", lambda *_: pytest.fail("merge reached model evaluation"))
    monkeypatch.setattr(pilot_eval, "merge_parts", lambda plan, parts, out: calls.append((plan, parts, out)))
    monkeypatch.setattr(sys, "argv", ["pilot_eval", "--plan", "plan.json", "--merge-parts", "dev.part.json", "legal.part.json",
                                    "--out", "evaluation.json"])
    pilot_eval.main()
    assert calls == [(Path("plan.json"), [Path("dev.part.json"), Path("legal.part.json")], Path("evaluation.json"))]


def test_incomplete_reference_job_arguments_refuse_before_model_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot_eval, "evaluate", lambda *_: pytest.fail("incomplete job reached model dispatch"))
    monkeypatch.setattr(sys, "argv", ["pilot_eval", "--plan", "plan.json", "--split", "dev", "--out", "dev.part.json"])
    with pytest.raises(SystemExit) as exc:
        pilot_eval.main()
    assert exc.value.code == 2
