"""CPU-only checks of shell argument preservation and profiling lifecycle."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

import pytest

from offload.gb10_budget.capture import run
from offload.gb10_budget.evidence import Object
from offload.gb10_budget.prepare_profile import render


def test_config_preserves_base_and_json_without_shell_execution(tmp_path: Path) -> None:
    base = tmp_path / "base $(touch SHOULD_NOT_EXIST).env"
    base.write_text("EXP_EXTRA_ARGS='--original yes'\nEXTRA_DOCKER_ARGS='-v original:/original:ro'\n")
    target = tmp_path / "profile.env"
    target.write_text(render(str(base), 16))
    result = subprocess.run(["bash", "-c", 'source ~/.config/ndm/paths.sh\nset -eu\n. "$1"\nprintf "%s\\n%s\\n" "$EXP_EXTRA_ARGS" "$EXTRA_DOCKER_ARGS"',
                             "bash", str(target)], cwd=tmp_path, capture_output=True, text=True, check=True)
    extra, mounts = result.stdout.splitlines()
    tokens = shlex.split(extra)
    assert tokens[:3] == ["--original", "yes", "--profiler-config"]
    config = json.loads(tokens[3])
    assert config["max_iterations"] == 16
    assert config["ignore_frontend"] is True
    assert config["torch_profiler_with_stack"] is False
    assert mounts == "-v original:/original:ro -v $HOME/dsv41-offload-run/sp1-profile:/opt/dsv41-sp1-profile"
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


class FakeBench:
    URL = ""
    MODEL = ""
    DATA_DIR = Path(".")

    def __init__(self, batch: int, fail: bool = False) -> None:
        self.batch = batch
        self.fail = fail
        self.emitted = 0
        self.lock = threading.Lock()

    def models(self) -> Object:
        return {"data": [{"id": "fixture", "max_model_len": 327680}]}

    def scrape(self, url: str | None = None) -> str:
        return "idle"

    def parse_metrics(self, text: str) -> dict[str, float]:
        return {"idle": 1.0}

    def require_idle(self, values: dict[str, float]) -> None:
        assert values["idle"] == 1

    def make_prompt(self, ctx: int, workload: str, filler: str, seed: int,
                    template: Object) -> tuple[str, int]:
        return "PRIVATE_PROMPT_SENTINEL", ctx

    def request_body(self, prompt: str, args: argparse.Namespace, salt: str) -> Object:
        return {"private": prompt}

    def stream_request(self, body: Object, started: threading.Event | None = None) -> Object:
        assert started is not None
        with self.lock:
            self.emitted += 1
        started.set()
        time.sleep(0.08)
        if self.fail:
            raise RuntimeError("stream fixture failure")
        return {"completion_tokens": 10, "answer_text": "PRIVATE_OUTPUT_SENTINEL",
                "reasoning_text": "PRIVATE_OUTPUT_SENTINEL"}


@pytest.mark.parametrize("phase,batch", [("none", 1), ("prefill", 1), ("decode", 4)])
def test_capture_phase_and_aggregate_privacy(phase: str, batch: int) -> None:
    bench = FakeBench(batch)
    calls: list[str] = []

    def command(url: str, verb: str) -> None:
        if verb == "start_profile" and phase == "decode":
            assert bench.emitted == batch
        calls.append(verb)

    args = argparse.Namespace(url="http://127.0.0.1:18001", data_dir=Path("."), ctx=32000,
                              seed=42, max_tokens=1500, batch=batch, phase=phase, seconds=0.005,
                              ttft_timeout=1)
    result = run(bench, args, command)
    assert calls == ([] if phase == "none" else ["start_profile", "stop_profile"])
    assert "PRIVATE_" not in json.dumps(result)


def test_stream_failure_still_stops_profile() -> None:
    args = argparse.Namespace(url="http://127.0.0.1:18001", data_dir=Path("."), ctx=32000,
                              seed=42, max_tokens=1500, batch=1, phase="prefill", seconds=0.005,
                              ttft_timeout=1)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="stream fixture failure"):
        run(FakeBench(1, fail=True), args, lambda url, verb: calls.append(verb))
    assert calls == ["start_profile", "stop_profile"]


def test_delayed_prefill_cannot_silently_capture_decode() -> None:
    args = argparse.Namespace(url="http://127.0.0.1:18001", data_dir=Path("."), ctx=150000,
                              seed=42, max_tokens=1500, batch=1, phase="prefill", seconds=0.005,
                              prefill_delay=0.02, ttft_timeout=1)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="Decode already started"):
        run(FakeBench(1), args, lambda url, verb: calls.append(verb))
    assert calls == []
