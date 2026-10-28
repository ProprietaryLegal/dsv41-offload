"""CPU-only profile drift, stale status and read-only command regressions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("live_profile", HERE / "live_profile.py")
assert SPEC and SPEC.loader
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def registry() -> dict[str, object]:
    return live.obj(json.loads((HERE / "profiles.v1.json").read_text()))


def observation() -> dict[str, object]:
    reg = registry()
    shared = live.obj(reg["shared"])
    profile = next(p for p in live.objects(reg["profiles"]) if p["name"] == "K1-327680")
    match, source = live.obj(profile["match"]), live.obj(profile["source_control"])
    return {"schema": "dsv41-live-observation-v1", "observed_at": "2026-10-23T22:39:10+00:00",
            "container_state": "running", "sparkhead_head": live.strings(source["allowed_heads"])[0],
            "dirty": source["dirty"], "applied_diff_sha256": source["diff_sha256"],
            "image": shared["image_id"], "env": deepcopy(match["env"]), "launch": deepcopy(match["launch"]),
            "kv_tokens": 2502600, "worker_current": "fb68c52",
            "tp2_unit": {"active": "active", "substate": "running", "pid": 123, "enabled": "enabled"},
            "one_unit": {"active": "inactive", "substate": "dead", "pid": None, "enabled": "disabled"},
            "worker": {"tp": 2, "pid": 123, "time": "2026-10-23T22:39:08Z", "tp_comm_errors_in_a_row": 0},
            "mounts": deepcopy(match["mounts"])}


class ProfilesTest(unittest.TestCase):
    def test_observed_profile(self) -> None:
        result = live.identify(observation(), registry())
        self.assertEqual(result["profile"], "K1-327680")
        self.assertEqual(result["result"], "CONFIGURATION_MATCH")
        self.assertIn("NOT a runtime-byte", result["qualification"])

    def test_identity_image_head_diff_and_dirty_drift(self) -> None:
        for key, value in (("image", "sha256:wrong"), ("worker_current", "unknown"),
                           ("sparkhead_head", "a" * 40), ("dirty", [" M dsv41.env"]),
                           ("applied_diff_sha256", "b" * 64), ("container_state", "exited")):
            with self.subTest(key=key):
                data = observation(); data[key] = value
                self.assertEqual(live.identify(data, registry())["result"], "DRIFT_OR_UNKNOWN")
        for key in ("VLLM_DSPARK_REMOTE_IDENTITY", "VLLM_DSPARK_REMOTE_MARKOV_IDENTITY",
                    "VLLM_DSPARK_REMOTE_ASYNC_PROXY", "VLLM_EXL3_TF_PROMPT_ALIAS"):
            data = observation(); live.obj(data["env"])[key] = "wrong"
            self.assertEqual(live.identify(data, registry())["profile"], "UNKNOWN")

    def test_stale_status_pid_and_conflicting_units(self) -> None:
        for key, value in (("time", "2026-10-23T22:37:08Z"), ("pid", 456),
                           ("tp_comm_errors_in_a_row", 1)):
            data = observation(); live.obj(data["worker"])[key] = value
            self.assertEqual(live.identify(data, registry())["result"], "DRIFT_OR_UNKNOWN")
        data = observation(); live.obj(data["one_unit"])["active"] = "active"
        self.assertEqual(live.identify(data, registry())["result"], "DRIFT_OR_UNKNOWN")

    def test_mount_count_path_version_and_mutability(self) -> None:
        for key, value in (("rw", True), ("source", "/wrong/release"), ("destination", "/wrong/target")):
            data = observation(); live.objects(data["mounts"])[0][key] = value
            self.assertEqual(live.identify(data, registry())["result"], "DRIFT_OR_UNKNOWN")
        data = observation()
        data["mounts"] = live.objects(data["mounts"])[:-1]
        self.assertEqual(live.identify(data, registry())["result"], "DRIFT_OR_UNKNOWN")

    def test_524288_stays_hold(self) -> None:
        data = observation(); launch = live.obj(data["launch"])
        launch["max_model_len"] = 524288; data["kv_tokens"] = 2820756
        result = live.identify(data, registry())
        self.assertEqual(result["profile"], "K1-524288")
        self.assertEqual(result["result"], "DRIFT_OR_UNKNOWN")
        self.assertIn("profile is HOLD, not qualified production", result["issues"])

    def test_native_and_adapted_never_implicitly_promoted(self) -> None:
        reg = registry()
        profiles = live.objects(reg["profiles"])
        self.assertEqual(len({p["name"] for p in profiles}), 8)
        for name in ("native-source-faithful", "adapted"):
            entry = next(p for p in profiles if p["name"] == name)
            self.assertIsNone(entry["match"])
        data = observation()
        live.obj(data["env"])["VLLM_DSPARK_REMOTE_IDENTITY"] = "5c3be5c7ee557b06fbef607030d857a038987b8d9f0d11257e1e376e37c5654e"
        self.assertEqual(live.identify(data, reg)["profile"], "UNKNOWN")

    def test_registered_files_and_git_refs(self) -> None:
        reg = registry()
        for profile in live.objects(reg["profiles"]):
            if profile["config"] is None:
                continue
            config = live.obj(profile["config"])
            self.assertEqual(hashlib.sha256((ROOT / str(config["path"])).read_bytes()).hexdigest(), config["sha256"])
        for manifest in live.obj(live.obj(reg["shared"])["manifests"]).values():
            entry = live.obj(manifest)
            self.assertEqual(hashlib.sha256((ROOT / str(entry["path"])).read_bytes()).hexdigest(), entry["sha256"])
        for name, value in live.obj(reg["rollback_refs"]).items():
            commit = subprocess.check_output(["git", "rev-parse", name + "^{}"], cwd=ROOT, text=True).strip()
            self.assertEqual(commit, live.obj(value)["commit"])

    def test_fixed_remote_read_commands_and_inactive_unit(self) -> None:
        commands = dict(live.READS)
        self.assertEqual({host for host, _ in commands.values()}, {"sparkhead", "x86host"})
        for host, command in commands.values():
            if host == "x86host":
                self.assertTrue(command.startswith(("systemctl --user status ", "readlink ", "cat ")))
            else:
                self.assertTrue(command.startswith(("cd ~/dsv41-deploy && git ", "docker inspect ", "docker logs ")))
            for forbidden in ("docker exec", "curl ", " restart ", "sudo ", "git add", "git commit", "git tag", " > "):
                self.assertNotIn(forbidden, command)
        result = subprocess.CompletedProcess([], 3, stdout="Active: inactive (dead)", stderr="")
        with patch.object(live.subprocess, "run", return_value=result):
            self.assertIn("inactive", live.read("one_unit"))
            with self.assertRaises(ValueError):
                live.read("head")

    def test_snapshot_does_not_contact_hosts(self) -> None:
        with patch.object(live.subprocess, "run", side_effect=AssertionError("no remote access")):
            self.assertEqual(live.identify(observation(), registry())["result"], "CONFIGURATION_MATCH")

    def test_enablement_and_extra_overlay_drift(self) -> None:
        data = observation(); live.obj(data["one_unit"])["enabled"] = "enabled"
        self.assertIn("worker unit enablement differs from production ownership", live.identify(data, registry())["issues"])
        data = observation()
        data["mounts"] = live.objects(data["mounts"]) + [{"source": "/new/overlay", "destination": "/new/target", "rw": False}]
        self.assertIn("Docker mount set differs from registered profile", live.identify(data, registry())["issues"])
        parsed = live.unit("Loaded: loaded (/example.service; enabled; preset: enabled)\nActive: active (running)\nMain PID: 123 (python)")
        self.assertEqual(parsed, {"active": "active", "substate": "running", "pid": 123, "enabled": "enabled"})


if __name__ == "__main__":
    unittest.main()
