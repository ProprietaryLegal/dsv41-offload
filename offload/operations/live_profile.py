"""Read-only production profile observation; no HTTP, CUDA, writes or service actions.

Use --live for fixed allowlisted SSH reads, or --snapshot for offline analysis.
A configuration match is not runtime hash/health qualification on both ranks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import cast

HERE = Path(__file__).resolve().parent
# No caller-supplied host/command. Only the OPS1 brief's read operations.
READS: dict[str, tuple[str, str]] = {
    "head": ("sparkhead", "cd ~/dsv41-deploy && git log -1 --format=%H"),
    "dirty": ("sparkhead", "cd ~/dsv41-deploy && git status --porcelain=v1 --untracked-files=normal"),
    "diff": ("sparkhead", "cd ~/dsv41-deploy && git diff --no-ext-diff --no-textconv -- dsv41.env start.sh"),
    "container": ("sparkhead", "docker inspect --format '{{json .}}' dsv41-serve"),
    "startup": ("sparkhead", "docker logs dsv41-serve 2>&1 | grep -E 'non-default args:|GPU KV cache size:'"),
    "tp2_unit": ("x86host", "systemctl --user status --no-pager --lines=0 dsv41-dspark-worker-tp2-prod.service"),
    "one_unit": ("x86host", "systemctl --user status --no-pager --lines=0 dsv41-dspark-worker.service"),
    "current": ("x86host", "readlink ~/dsv41-offload/prod/dspark-worker/current"),
    "worker": ("x86host", "cat ~/dsv41-offload/prod/run/dspark-worker.status.json"),
}
ENV_KEYS = frozenset({
    "VLLM_DSPARK_REMOTE", "VLLM_DSPARK_REMOTE_IDENTITY", "VLLM_DSPARK_REMOTE_MARKOV_IDENTITY",
    "VLLM_DSPARK_REMOTE_ASYNC", "VLLM_DSPARK_REMOTE_ASYNC_PROXY", "VLLM_DSPARK_REMOTE_POLICY",
    "VLLM_DSPARK_REMOTE_ASYNC_SPIN_US", "VLLM_DSPARK_REMOTE_ASYNC_NAP_US",
    "VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS", "VLLM_DSPARK_REMOTE_ASYNC_RESPAWN_RESET_S",
    "VLLM_DSPARK_REMOTE_TIMEOUT_MS", "VLLM_DSPARK_REMOTE_DRAFT_MS", "VLLM_DSPARK_REMOTE_ROUND_MS",
    "VLLM_ENGRAM_REMOTE", "VLLM_ENGRAM_REMOTE_REQUIRED", "VLLM_DSV41_KV_PAIR",
    "VLLM_DSV41_DROP_MTP_BUFFER", "VLLM_EXL3_TF_PROMPT_TRANSIENT", "VLLM_EXL3_TF_PROMPT_ALIAS",
    "VLLM_INDEXER_PREFILL_BUFFER_FACTOR", "CUDA_MODULE_LOADING", "DSPARK_WORKER_ARM",
})
WORKER_KEYS = frozenset({"worker", "pid", "time", "uptime_s", "steps", "rows", "drafted_requests",
                         "tp", "tp_comm_errors", "tp_comm_errors_in_a_row", "session", "new_sessions"})


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return cast(dict[str, object], value)


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("expected object list")
    return [obj(v) for v in value]


def strings(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("expected string list")
    return cast(list[str], value)


def read(key: str) -> str:
    host, command = READS[key]
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                            capture_output=True, text=True, timeout=25)
    # systemctl status returns 3 for an inactive unit; inspection still succeeds.
    if result.returncode != 0 and not (key.endswith("_unit") and result.returncode == 3):
        raise ValueError(f"read {key} on {host} failed: exit {result.returncode}")
    return result.stdout


def unit(raw: str) -> dict[str, object]:
    state = re.search(r"Active: (\w+) \(([^)]+)\)", raw)
    if not state:
        raise ValueError("systemctl status lacks Active state")
    pid = re.search(r"Main PID: (\d+)", raw)
    enabled = re.search(r"Loaded:.*; (enabled|disabled|static|masked)(?:;|\))", raw)
    if not enabled:
        raise ValueError("systemctl status lacks unit enablement")
    return {"active": state[1], "substate": state[2], "pid": int(pid[1]) if pid else None,
            "enabled": enabled[1]}


def collect() -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = dict(zip(READS, pool.map(read, READS)))
    container = obj(json.loads(values["container"]))
    config = obj(container.get("Config"))
    env: dict[str, str] = {}
    for entry in strings(config.get("Env")):
        key, sep, value = entry.partition("=")
        if key in ENV_KEYS and sep:
            if key in env:
                raise ValueError(f"duplicate Docker env key: {key}")
            env[key] = value
    mounts = [{"source": m.get("Source"), "destination": m.get("Destination"), "rw": m.get("RW")}
              for m in objects(container.get("Mounts"))]
    startup = values["startup"]
    capacities = re.findall(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens", startup)
    if len(capacities) != 1:
        raise ValueError("need exactly one current-container KV/max-length startup pair")
    flags: dict[str, int] = {}
    for name in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "max_model_len"):
        matches = re.findall(r"'" + name + r"': (\d+)", startup)
        if len(matches) != 1:
            raise ValueError(f"startup {name} absent/ambiguous")
        flags[name] = int(matches[0])
    if flags["max_model_len"] != int(capacities[0][1].replace(",", "")):
        raise ValueError("startup context observations disagree")
    worker = obj(json.loads(values["worker"]))
    return {
        "schema": "dsv41-live-observation-v1", "started_at": started,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "sparkhead_head": values["head"].strip(), "dirty": values["dirty"].splitlines(),
        "applied_diff_sha256": hashlib.sha256(values["diff"].encode()).hexdigest(),
        "container_id": container.get("Id"), "image": container.get("Image"),
        "container_state": obj(container.get("State")).get("Status"),
        "container_started_at": obj(container.get("State")).get("StartedAt"),
        "env": env, "mounts": mounts, "launch": flags,
        "kv_tokens": int(capacities[0][0].replace(",", "")),
        "tp2_unit": unit(values["tp2_unit"]), "one_unit": unit(values["one_unit"]),
        "worker_current": values["current"].strip(),
        "worker": {k: v for k, v in worker.items() if k in WORKER_KEYS},
    }


def identify(data: dict[str, object], registry: dict[str, object]) -> dict[str, object]:
    if data.get("schema") != "dsv41-live-observation-v1":
        raise ValueError("unsupported observation schema")
    if registry.get("schema") != "dsv41-production-profiles-v1":
        raise ValueError("unsupported registry schema")
    env = obj(data.get("env"))
    launch = obj(data.get("launch"))
    worker = obj(data.get("worker"))
    shared = obj(registry.get("shared"))
    issues: list[str] = []
    if data.get("container_state") != "running":
        issues.append("target container not running")
    if data.get("image") != shared["image_id"]:
        issues.append("target image ID mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(data.get("sparkhead_head"))):
        issues.append("missing/invalid sparkhead HEAD")
    selected: list[str] = []
    matched: dict[str, object] | None = None
    for profile in objects(registry.get("profiles")):
        match = profile.get("match")
        if match is None:
            continue  # native/adapted have no production-qualified match contract yet
        rule = obj(match)
        expected = obj(rule["env"])
        # Absence is explicit: no silent native/K1 mismatch matched as a baseline.
        if any(env.get(k) != v for k, v in expected.items()):
            continue
        if any(launch.get(k) != v for k, v in obj(rule["launch"]).items()):
            continue
        if data.get("kv_tokens") != rule["kv_tokens"]:
            continue
        if rule["worker_tp"] is not None and worker.get("tp", 1) != rule["worker_tp"]:
            continue
        selected.append(str(profile["name"]))
        matched = profile
    if len(selected) != 1:
        issues.append(f"expected one registry match, found {selected}")
    elif matched is not None:
        if matched["status"] == "HOLD":
            issues.append("profile is HOLD, not qualified production")
        source = obj(matched["source_control"])
        if data.get("sparkhead_head") not in strings(source["allowed_heads"]):
            issues.append("sparkhead HEAD differs from registered profile")
        dirty = strings(data.get("dirty"))
        if dirty != strings(source["dirty"]):
            issues.append("sparkhead dirty state differs from registered profile")
        if data.get("applied_diff_sha256") != source["diff_sha256"]:
            issues.append("sparkhead applied diff differs from registered profile")
        mounts = obj(matched["match"]).get("mounts")
        if mounts is not None:
            def canonical(value: object) -> list[str]:
                return sorted(json.dumps(m, sort_keys=True) for m in objects(value))
            if canonical(data.get("mounts")) != canonical(mounts):
                issues.append("Docker mount set differs from registered profile")
    if env.get("VLLM_DSPARK_REMOTE"):
        kind = "tp2_unit" if worker.get("tp", 1) == 2 else "one_unit"
        service = obj(data.get(kind))
        other = obj(data.get("one_unit" if kind == "tp2_unit" else "tp2_unit"))
        if service.get("active") != "active" or service.get("substate") != "running":
            issues.append("selected worker unit not running")
        if other.get("active") == "active":
            issues.append("conflicting worker units active")
        if service.get("enabled") != "enabled" or other.get("enabled") != "disabled":
            issues.append("worker unit enablement differs from production ownership")
        if service.get("pid") != worker.get("pid"):
            issues.append("worker status pid differs from live unit")
        when = datetime.fromisoformat(str(worker.get("time")).replace("Z", "+00:00"))
        observed = datetime.fromisoformat(str(data["observed_at"]))
        if not -5 <= (observed - when).total_seconds() <= 30:
            issues.append("worker status stale/future (>30s old or >5s ahead)")
        if worker.get("tp_comm_errors_in_a_row", 0) != 0:
            issues.append("worker reports consecutive TP communication errors")
        expected_current = obj(matched["identity"])["worker_release"] if matched else None
        if expected_current is None or str(data.get("worker_current")) != expected_current:
            issues.append("worker current differs from recorded production release")
    if env.get("VLLM_DSV41_KV_PAIR") == "3":
        expected_paths = strings(shared["k1_paths"])
        kv_mounts = [m for m in objects(data.get("mounts")) if "/dsv41-kvpack/" in str(m.get("source"))]
        want = {"/usr/local/lib/python3.12/dist-packages/" + p for p in expected_paths}
        if len(kv_mounts) != len(want) or {str(m.get("destination")) for m in kv_mounts} != want:
            issues.append("K1 overlay mount coverage mismatch")
        if any(m.get("rw") is not False or "/559c5867ba42/" not in str(m.get("source")) for m in kv_mounts):
            issues.append("K1 overlay mount version/read-only mismatch")
    return {"profile": selected[0] if len(selected) == 1 else "UNKNOWN",
            "result": "CONFIGURATION_MATCH" if not issues else "DRIFT_OR_UNKNOWN",
            "issues": issues,
            "qualification": "NOT a runtime-byte/health attestation; rank1 and worker hashes not read",
            "observation": data}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--snapshot", type=Path)
    parser.add_argument("--registry", type=Path, default=HERE / "profiles.v1.json")
    args = parser.parse_args()
    try:
        data = collect() if args.live else obj(json.loads(args.snapshot.read_text()))
        # Accept the aggregate observation itself or this command's full output.
        if "observation" in data:
            data = obj(data["observation"])
        result = identify(data, obj(json.loads(args.registry.read_text())))
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["result"] != "CONFIGURATION_MATCH":
            parser.exit(2)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"profile observation FAILED: {exc}\n")


if __name__ == "__main__":
    main()
