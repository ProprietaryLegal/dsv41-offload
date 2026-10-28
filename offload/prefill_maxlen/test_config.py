from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from offload.prefill_maxlen.build_overlay import build

REPO = Path(__file__).resolve().parents[2]
CONFIG = "offload/launch/configs/PF1-prefill.env"


def source_config(overlay: Path, **settings: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PF1_", "K1_"))}
    env.update(settings)
    env["PF1_SOURCE_OVERLAY"] = str(overlay)
    # Sourcing a config does not invoke any launcher, SSH, docker or HTTP.
    return subprocess.run(
        ["bash", "-euc", f'source ~/.config/ndm/paths.sh\n. {CONFIG}\n'
         'printf "%s\\n" "$MAX_LEN_OVERRIDE" "$LABEL" "$EXP_DOCKER_ENV" "$OVERLAYS"'],
        cwd=REPO, env=env, capture_output=True, text=True,
    )


def test_valid_config_retains_base_mounts(tmp_path: Path) -> None:
    overlay = build(Path("/opt/dsv41/data/src-image"), tmp_path / "overlay")
    for cap in ("327680", "393216", "524288"):
        for enabled in ("0", "1"):
            for factor in ("1", "4"):
                result = source_config(overlay, PF1_CAP=cap, PF1_WORKSPACE=enabled,
                                       K1_INDEXER_FACTOR=factor)
                assert result.returncode == 0, result.stderr
                lines = result.stdout.splitlines()
                assert lines[0] == cap
                assert lines[1] == f"PF1-{cap}-views{enabled}-factor{factor}"
                assert f"VLLM_DSV41_INDEXER_REQUEST_WORKSPACE={enabled}" in lines[2]
                assert f"VLLM_INDEXER_PREFILL_BUFFER_FACTOR={factor}" in lines[2]
                assert "VLLM_DSV41_KV_PAIR=3" in lines[2]
                assert len(lines[3].split()) == 16  # 2 DSpark, 13 K1, 1 PF1
    default = source_config(overlay)
    assert default.returncode == 0
    assert default.stdout.splitlines()[1] == "PF1-524288-views0-factor4"


@pytest.mark.parametrize("settings", [
    {"PF1_CAP": "1048576"}, {"PF1_CAP": "bad"}, {"PF1_WORKSPACE": "2"},
    {"K1_INDEXER_FACTOR": "0"}, {"K1_INDEXER_FACTOR": "2.5"},
])
def test_invalid_config_refuses(tmp_path: Path, settings: dict[str, str]) -> None:
    overlay = build(Path("/opt/dsv41/data/src-image"), tmp_path / "overlay")
    result = source_config(overlay, **settings)
    assert result.returncode != 0
    assert "PF1" in result.stderr


def test_foreign_overlay_refuses(tmp_path: Path) -> None:
    overlay = tmp_path / "foreign.py"
    overlay.write_text("pass\n")
    result = source_config(overlay)
    assert result.returncode != 0
    assert "PF1 overlay SHA mismatch" in result.stderr
