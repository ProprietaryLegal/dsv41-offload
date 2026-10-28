"""CPU-only integration tests; network/service commands are local recording stubs.

Fixtures are synthetic. Their PROMOTE records never leave the temporary test root
and are not real approvals. Archive main dynamically to catch a moving base.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import cast

import pytest

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
SOURCE = Path("/opt/dsv41/data/src-image")
FILES = ("dsv41.env", "start.sh")
SWITCHES = {
    "VLLM_DSV41_KV_PAIR": "3", "VLLM_DSV41_DROP_MTP_BUFFER": "1",
    "VLLM_EXL3_TF_PROMPT_TRANSIENT": "1", "VLLM_EXL3_TF_PROMPT_ALIAS": "1",
    "VLLM_INDEXER_PREFILL_BUFFER_FACTOR": "4",
}


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PACKAGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(PACKAGE))
from profiles import Profile

MANAGE = load("manage")
RUNTIME = load("check_runtime")


def command(args: list[str], root: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, check=False)


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture(params=("524288", "327680"))
def profile(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def selected(root: Path) -> Profile:
    return MANAGE.get_profile((root / ".test-profile").read_text())


def other_profile(profile: str) -> str:
    return "327680" if profile == "524288" else "524288"


@pytest.fixture
def root(tmp_path: Path, profile: str) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in (*FILES, "stop.sh", "serve.sh"):
        (tmp_path / name).write_bytes(subprocess.check_output(["git", "show", f"main:{name}"], cwd=REPO))
        (tmp_path / name).chmod(0o755 if name.endswith(".sh") else 0o644)
    shutil.copytree(PACKAGE, tmp_path / "offload/promote/k1", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "offload/kvpack", tmp_path / "offload/kvpack", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".test-profile").write_text(profile)
    return tmp_path


def approval(root: Path) -> tuple[Path, Path]:
    report = root / "S18.synthetic.md"
    report.write_text("SYNTHETIC CPU FIXTURE ONLY: final review passes both windows\n")
    receipt = root / "S18.synthetic.json"
    receipt.write_text(json.dumps({
        "schema": "dsv41-k1-promotion-v1", "reviewer": "S18", "final": True,
        "window6": "PROMOTE", "window6b": "PROMOTE" if selected(root).requires_r5 else "HOLD",
        "max_model_len": selected(root).max_model_len, "kv_tokens": selected(root).kv_tokens, "manifest_sha256": MANAGE.MANIFEST_SHA,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }))
    return receipt, report


def manage(root: Path, action: str) -> subprocess.CompletedProcess[str]:
    args = ["python3", str(root / "offload/promote/k1/manage.py"), action]
    if action in ("check", "apply"):
        receipt, report = approval(root)
        args += ["--profile", selected(root).name, "--receipt", str(receipt), "--report", str(report)]
    return command(args, root)


def sourced(root: Path) -> dict[str, str]:
    result = command(["bash", "-c", 'set -a; . ./dsv41.env; python3 -c "import os,json; print(json.dumps(dict(os.environ)))"'], root)
    assert result.returncode == 0, result.stderr
    return cast(dict[str, str], json.loads(result.stdout))


def test_patches_apply_current_main_and_preserve_settings(root: Path) -> None:
    before = sourced(root)
    assert manage(root, "check").returncode == 0
    assert manage(root, "apply").returncode == 0
    after = sourced(root)
    assert after["PROD_MAX_LEN"] == selected(root).name
    if selected(root).name == "327680":
        assert after["PROD_MAX_LEN"] == before["PROD_MAX_LEN"]
    assert after["PROD_KV_BYTES"] == before["PROD_KV_BYTES"] == "6442450944"
    assert after["OFFLOAD_DSPARK_ENV"] == before["OFFLOAD_DSPARK_ENV"]
    extra_env = after["OFFLOAD_DOCKER_ENV"].removeprefix(before["OFFLOAD_DOCKER_ENV"])
    assert dict(item.split("=", 1) for item in extra_env.split()) == SWITCHES
    extra_args = after["OFFLOAD_DOCKER_ARGS"].removeprefix(before["OFFLOAD_DOCKER_ARGS"])
    expected = command(["bash", "offload/kvpack/stage_overlay.sh", "args"], root)
    assert extra_args.strip() == expected.stdout.strip()
    for key, value in before.items():
        if key not in {"OFFLOAD_DOCKER_ENV", "OFFLOAD_DOCKER_ARGS", "PROD_MAX_LEN", "_"}:
            assert after[key] == value, key
    for name in (*FILES, "serve.sh", "stop.sh"):
        assert command(["bash", "-n", name], root).returncode == 0
    assert command(["git", "diff", "--check"], root).returncode == 0


@pytest.mark.parametrize(("key", "value"), [
    ("final", False), ("final", 1), ("window6", "PENDING"), ("window6", "FAIL"),
    ("window6b", "PENDING"), ("window6b", "FAIL"), ("reviewer", "other"),
    ("max_model_len", 1048576), ("max_model_len", "524288"), ("kv_tokens", 2502600),
    ("manifest_sha256", "wrong"), ("report_sha256", "wrong"), ("schema", "wrong"),
])
def test_gate_refuses_before_mutation(root: Path, key: str, value: object) -> None:
    receipt, report = approval(root)
    record = json.loads(receipt.read_text())
    if key == "kv_tokens":
        value = MANAGE.get_profile(other_profile(selected(root).name)).kv_tokens
    record[key] = value
    receipt.write_text(json.dumps(record))
    before = {name: (root / name).read_bytes() for name in FILES}
    if key == "window6b" and not selected(root).requires_r5:
        MANAGE.gate(receipt, report, selected(root))  # R5 does not gate the R1 profile
    else:
        with pytest.raises(ValueError, match="S18 final PROMOTE gate"):
            MANAGE.apply(root, receipt, report, selected(root))
    assert all((root / name).read_bytes() == data for name, data in before.items())
    assert not (root / "logs/k1-promotion").exists()


def test_missing_gate_refuses(root: Path) -> None:
    result = command(["python3", "offload/promote/k1/manage.py", "apply", "--profile", "524288"], root)
    assert result.returncode != 0 and "require --receipt and --report" in result.stderr


def test_restore_byte_exact_and_backup_not_overwritten(root: Path) -> None:
    # Preserve unrelated pre-existing edits in the actual production snapshot.
    env_file = root / "dsv41.env"
    env_file.write_text("# coordinator's unrelated local note\n" + env_file.read_text())
    originals = {name: ((root / name).read_bytes(), (root / name).stat().st_mode) for name in FILES}
    assert manage(root, "apply").returncode == 0
    assert manage(root, "verify").returncode == 0
    assert manage(root, "restore").returncode == 0
    assert manage(root, "restore").returncode == 0  # safe repeated recovery
    for name, (data, mode) in originals.items():
        assert (root / name).read_bytes() == data
        assert (root / name).stat().st_mode == mode
    result = manage(root, "apply")
    assert result.returncode != 0
    assert all((root / name).read_bytes() == data for name, (data, _) in originals.items())


def test_patch_context_conflict_is_a_safe_refusal(root: Path) -> None:
    # This patch changes the end of dsv41.env. An appended line changes its
    # EOF context and requires explicit re-review, rather than relaxed matching.
    with (root / "dsv41.env").open("a") as stream:
        stream.write("# changed EOF context\n")
    before = {name: (root / name).read_bytes() for name in FILES}
    result = manage(root, "apply")
    assert result.returncode != 0 and "patch does not apply" in result.stderr
    assert all((root / name).read_bytes() == data for name, data in before.items())
    assert not (root / "logs/k1-promotion").exists()


@pytest.mark.parametrize("defect", ["backup", "later-edit", "saved-review"])
def test_restore_and_start_record_tamper_refusals(root: Path, defect: str) -> None:
    assert manage(root, "apply").returncode == 0
    path = root / {"backup": "logs/k1-promotion/start.sh", "later-edit": "start.sh",
                   "saved-review": "logs/k1-promotion/S18.report.md"}[defect]
    path.write_text(path.read_text() + "\n# later change\n")
    before = {name: (root / name).read_bytes() for name in FILES}
    assert manage(root, "verify").returncode != 0
    if defect != "saved-review":
        assert manage(root, "restore").returncode != 0
        assert all((root / name).read_bytes() == data for name, data in before.items())
    else:
        assert manage(root, "restore").returncode == 0  # rollback never needs new approval


def test_stage_refuses_changed_checkout_before_ssh(root: Path) -> None:
    (root / "offload/kvpack/overlay/vllm/v1/kv_cache_interface.py").write_text("# tampered\n")
    executable(root / "bin/ssh", "#!/bin/sh\necho UNEXPECTED_REMOTE >&2; exit 99\n")
    env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"])
    result = command(["bash", "offload/promote/k1/stage.sh", "one", "two"], root, env)
    assert result.returncode != 0 and "UNEXPECTED_REMOTE" not in result.stderr


def test_stage_refuses_changed_manifest_before_ssh(root: Path) -> None:
    manifest = root / "offload/kvpack/MANIFEST.json"
    manifest.write_text(manifest.read_text() + "\n")
    executable(root / "bin/ssh", "#!/bin/sh\necho UNEXPECTED_REMOTE >&2; exit 99\n")
    env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"])
    result = command(["bash", "offload/promote/k1/stage.sh", "one", "two"], root, env)
    assert result.returncode != 0 and "not the reviewed production manifest" in result.stderr
    assert "UNEXPECTED_REMOTE" not in result.stderr


@pytest.mark.parametrize("defect", ["image", "staged"])
def test_real_staging_refuses_mismatch(root: Path, defect: str) -> None:
    # Exercise the real hash comparison/copy path locally, without ssh/docker.
    image = root / "image"
    shutil.copytree(SOURCE, image, ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, HOME=str(root / "home"), KVPACK_TEST_IMAGE_ROOT=str(image))
    stage = ["bash", "offload/kvpack/stage_overlay.sh", "stage", "local"]
    if defect == "staged":
        assert command(stage, root, env).returncode == 0
        target = next((root / "home/dsv41-kvpack").iterdir()) / "vllm/v1/worker/utils.py"
    else:
        target = image / "vllm/v1/worker/utils.py"
    target.write_text(target.read_text() + "\n# wrong bytes\n")
    result = command(stage, root, env)
    assert result.returncode != 0
    assert "not staging" in result.stderr or "staged copies differ" in result.stderr


GOOD_LOG = "(EngineCore pid=123) GPU KV cache size: 2,820,756 tokens, Maximum concurrency for 524,288 tokens per request: 5.38x\n"


@pytest.mark.parametrize("text", ["", GOOD_LOG.replace("2,820,756", "2,502,600"),
                                  GOOD_LOG.replace("524,288", "327,680"),
                                  GOOD_LOG + GOOD_LOG.replace("524,288", "1,048,576")])
def test_runtime_rejects_missing_wrong_or_conflicting_log(text: str, profile: str) -> None:
    if profile == "327680":
        # Swap good/wrong profile figures without weakening the refusal cases.
        text = text.replace("2,820,756", "KV_GOOD").replace("2,502,600", "2,820,756").replace("KV_GOOD", "2,502,600")
        text = text.replace("524,288", "LEN_GOOD").replace("327,680", "524,288").replace("LEN_GOOD", "327,680")
    with pytest.raises(ValueError):
        RUNTIME.check_log(text, MANAGE.get_profile(profile))


def test_runtime_checks_allocator_and_api(profile: str) -> None:
    contract = MANAGE.get_profile(profile)
    good = f"GPU KV cache size: {contract.kv_tokens:,} tokens, Maximum concurrency for {contract.max_model_len:,} tokens per request: 1x"
    RUNTIME.check_log(good, contract)
    RUNTIME.check_log("\x1b[32m" + good + "\x1b[0m", contract)
    RUNTIME.check_models(json.dumps({"data": [{"id": "alias", "max_model_len": contract.max_model_len}]}), "alias", contract)
    for data in ({}, {"data": []}, {"data": [{"id": "wrong", "max_model_len": contract.max_model_len}]},
                 {"data": [{"id": "alias", "max_model_len": int(other_profile(profile))}]},
                 {"data": [{"id": "alias", "max_model_len": profile}]}):
        with pytest.raises(ValueError):
            RUNTIME.check_models(json.dumps(data), "alias", contract)


# Stubs model remote files by manifest; all subprocesses stay on Controller CPU.
SSH_STUB = '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['STUB_ROOT'])
args=sys.argv[1:]
while args and args[0]=='-o': args=args[2:]
node,cmd=args[0],args[1]
with (root/'events').open('a') as f: f.write(node+' '+cmd+'\\n')
fault=os.environ.get('STUB_FAULT','')
rows=json.loads((root/'offload/kvpack/MANIFEST.json').read_text())['files']
if 'docker ps ' in cmd:
    if fault=='already-running': print('dsv41-serve')
    if fault=='inspect-error': sys.exit(2)
elif 'nvidia-smi --query-compute-apps' in cmd: pass
elif 'docker exec dsv41-serve sha256sum' in cmd:
    for row in rows:
        sha=row['overlay_sha256']
        if fault=='worker-overlay' and node=='sparkworker': sha='0'*64
        print(sha+'  '+row['container_path'])
elif 'docker inspect dsv41-serve --format' in cmd:
    switches='VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4'
    if fault=='worker-switch' and node=='sparkworker': switches+=' VLLM_DSV41_KV_NVFP4=1'
    print(switches.replace(' ','\\n'))
elif 'docker logs dsv41-serve' in cmd:
    if fault not in ('missing-log','fallback-stop','fallback-start'):
        small=os.environ['STUB_PROFILE']=='327680'
        kv='2,502,600' if small != (fault=='kv') else '2,820,756'
        ml='327,680' if small != (fault=='log-len') else '524,288'
        print('GPU KV cache size: '+kv+' tokens, Maximum concurrency for '+ml+' tokens per request: 5.38x')
elif 'docker run ' in cmd and '--entrypoint sha256sum' in cmd:
    for row in rows:
        if row['base_sha256']: print(row['base_sha256']+'  '+row['container_path'])
elif 'docker run ' in cmd and '--entrypoint sh ' in cmd:
    for row in rows:
        if not row['base_sha256']: print('ABSENT '+row['container_path'])
elif cmd.startswith('test -d '): sys.exit(1)
elif 'tar xf - ' in cmd: sys.stdin.buffer.read()
elif 'sha256sum -c --quiet -' in cmd:
    sys.stdin.buffer.read()
    if fault=='stage' and node=='sparkworker': sys.exit(1)
elif any(token in cmd for token in ('test -f ', 'docker image inspect', 'sudo -n nvidia-smi', 'mkdir -p ', 'mv ')): pass
else: print('UNEXPECTED SSH: '+cmd,file=sys.stderr);sys.exit(98)
'''


def stub_environment(root: Path, fault: str = "") -> dict[str, str]:
    executable(root / "bin/ssh", SSH_STUB)
    executable(root / "bin/curl", '''#!/usr/bin/env python3
import json,os
print(json.dumps({'data':[{'id':'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2',
 'max_model_len':(327680 if os.environ['STUB_PROFILE']=='524288' else 524288)
 if os.environ.get('STUB_FAULT')=='api' else int(os.environ['STUB_PROFILE'])}]}))
''')
    stub = '#!/bin/bash\nset -euo pipefail\necho "stub $0" >> "$STUB_ROOT/events"\n'
    for name in ("firewall.sh", "offload/engram/spark/stage_overlay.sh", "offload/engram/spark/prebuild_jit.sh",
                 "offload/dspark/spark/stage_overlay.sh", "offload/launch/warmup_serve.sh"):
        executable(root / name, stub)
    executable(root / "serve.sh", '''#!/bin/bash
set -euo pipefail
if grep -q '^PROD_K1=1$' ./dsv41.env; then
  echo CANDIDATE_START >> "$STUB_ROOT/events"
  [ "${STUB_FAULT:-}" != launch ] || exit 3
else
  echo PREVIOUS_START >> "$STUB_ROOT/events"
  [ "${STUB_FAULT:-}" != fallback-start ] || exit 4
fi
''')
    executable(root / "home/dgx-spark-deepseek-v41/scripts/launch/dsv41_tp2.sh", '''#!/bin/bash
set -euo pipefail
echo STOP_CANDIDATE >> "$STUB_ROOT/events"
[ "${STUB_FAULT:-}" != fallback-stop ]
''')
    env = dict(os.environ, HOME=str(root / "home"), STUB_ROOT=str(root), STUB_FAULT=fault, STUB_PROFILE=selected(root).name,
               PATH=str(root / "bin") + ":" + os.environ["PATH"], CUDA_VISIBLE_DEVICES="")
    # Prevent the parent invocation/environment from masking fixture conditions.
    for key in list(env):
        if key.startswith("KVPACK_TEST_") or key in {"K1_START_INNER", "PROD_K1"}:
            del env[key]
    return env


def test_complete_patched_start_verifies_both_nodes(root: Path) -> None:
    env = stub_environment(root)
    assert manage(root, "apply").returncode == 0
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "K1 START VERIFIED" in result.stdout
    events = (root / "events").read_text()
    for node in ("spark", "sparkworker"):
        assert f"{node} docker exec dsv41-serve sha256sum" in events
        assert f"{node} docker inspect dsv41-serve --format" in events
    assert "CANDIDATE_START" in events and "PREVIOUS_START" not in events
    assert "STOP_CANDIDATE" not in events


@pytest.mark.parametrize("fault", ["stage", "worker-overlay", "worker-switch", "kv", "log-len", "api", "missing-log", "launch"])
def test_start_failure_restores_previous_files_and_restarts(root: Path, fault: str) -> None:
    env = stub_environment(root, fault)
    originals = {name: (root / name).read_bytes() for name in FILES}
    assert manage(root, "apply").returncode == 0
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0
    assert "K1 START VERIFIED" not in result.stdout
    assert "previous configuration restarted" in result.stderr, result.stdout + result.stderr
    assert all((root / name).read_bytes() == data for name, data in originals.items())
    events = (root / "events").read_text()
    assert events.count("STOP_CANDIDATE") == 1
    assert events.count("PREVIOUS_START") == 1


def test_repeated_start_never_stops_running_service(root: Path) -> None:
    env = stub_environment(root, "already-running")
    assert manage(root, "apply").returncode == 0
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0 and "already runs" in result.stderr
    events = (root / "events").read_text()
    assert "STOP_CANDIDATE" not in events and "CANDIDATE_START" not in events


def test_inspection_failure_never_stops_service(root: Path) -> None:
    env = stub_environment(root, "inspect-error")
    assert manage(root, "apply").returncode == 0
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0
    events = (root / "events").read_text()
    assert "STOP_CANDIDATE" not in events and "CANDIDATE_START" not in events


@pytest.mark.parametrize("fault", ["fallback-stop", "fallback-start"])
def test_failed_fallback_is_loud_and_never_success(root: Path, fault: str) -> None:
    env = stub_environment(root, fault)
    assert manage(root, "apply").returncode == 0
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0
    assert "K1 FALLBACK FAILED" in result.stderr
    assert "previous configuration restarted" not in result.stderr
    assert "K1 START VERIFIED" not in result.stdout


def test_start_rechecks_final_review(root: Path) -> None:
    env = stub_environment(root)
    originals = {name: (root / name).read_bytes() for name in FILES}
    assert manage(root, "apply").returncode == 0
    receipt = root / "logs/k1-promotion/S18.receipt.json"
    record = json.loads(receipt.read_text())
    record["window6b" if selected(root).requires_r5 else "window6"] = "PENDING"
    receipt.write_text(json.dumps(record))
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0 and "previous configuration restarted" in result.stderr
    assert all((root / name).read_bytes() == data for name, data in originals.items())
    assert "CANDIDATE_START" not in (root / "events").read_text()


def test_production_stage_rejects_test_hook(root: Path) -> None:
    env = stub_environment(root)
    env["KVPACK_TEST_IMAGE_ROOT"] = str(SOURCE)
    result = command(["bash", "offload/promote/k1/stage.sh", "spark", "sparkworker"], root, env)
    assert result.returncode != 0 and "refuses test hook" in result.stderr
    assert not (root / "events").exists()


@pytest.mark.parametrize("action", ["check", "apply"])
def test_profile_must_be_explicit(root: Path, action: str) -> None:
    receipt, report = approval(root)
    result = command(["python3", "offload/promote/k1/manage.py", action,
                      "--receipt", str(receipt), "--report", str(report)], root)
    assert result.returncode != 0 and "require --profile" in result.stderr
    assert not (root / "logs/k1-promotion").exists()


@pytest.mark.parametrize("name", ["327680", "524288"])
def test_profile_receipt_and_saved_selection(root: Path, name: str) -> None:
    profile = MANAGE.get_profile(name)
    receipt, report = approval(root)
    record = json.loads(receipt.read_text())
    record.update(max_model_len=profile.max_model_len, kv_tokens=profile.kv_tokens, window6b="HOLD")
    receipt.write_text(json.dumps(record))
    if profile.requires_r5:
        with pytest.raises(ValueError, match="window6b"):
            MANAGE.gate(receipt, report, profile)
        record["window6b"] = "PROMOTE"
        receipt.write_text(json.dumps(record))
    MANAGE.apply(root, receipt, report, profile)
    assert sourced(root)["PROD_K1_PROFILE"] == name
    saved = root / "logs/k1-promotion/files.json"
    assert json.loads(saved.read_text())["profile"] == name
    MANAGE.verify(root, name)
    other = "524288" if name == "327680" else "327680"
    with pytest.raises(ValueError, match="differs from configured profile"):
        MANAGE.verify(root, other)
    changed = json.loads(saved.read_text())
    changed["profile"] = other
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="S18 final PROMOTE gate"):
        MANAGE.verify(root)
    MANAGE.restore(root)  # recovery never depends on a valid approval/profile


@pytest.mark.parametrize("name", ["327680", "524288"])
def test_shell_constants_match_runtime_and_allocator(name: str) -> None:
    profile = MANAGE.get_profile(name)
    result = command(["bash", "-c", '. offload/promote/k1/constants.sh; '
                      f'k1_select_profile {name}; echo "$K1_MAX_LEN $K1_KV_TOKENS $K1_KV_BYTES"'], REPO)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [str(profile.max_model_len), str(profile.kv_tokens), "6442450944"]
    spec = importlib.util.spec_from_file_location("kv_arith", REPO / "offload/kvpack/kv_arith.py")
    assert spec is not None and spec.loader is not None
    arithmetic = importlib.util.module_from_spec(spec)
    sys.modules["kv_arith"] = arithmetic
    spec.loader.exec_module(arithmetic)
    assert arithmetic.capacity(6442450944, profile.max_model_len, 3, 14)[0] == profile.kv_tokens


@pytest.mark.parametrize("name", ["", "unknown"])
def test_shell_constants_refuse_missing_or_unknown_profile(name: str) -> None:
    result = command(["bash", "-c", '. offload/promote/k1/constants.sh; k1_select_profile "$1"', "test", name], REPO)
    assert result.returncode != 0 and "explicit profile" in result.stderr


def test_runtime_327680_contract_is_distinct() -> None:
    profile = MANAGE.get_profile("327680")
    r1_log = GOOD_LOG.replace("2,820,756", "2,502,600").replace("524,288", "327,680")
    RUNTIME.check_log(r1_log, profile)
    RUNTIME.check_models(json.dumps({"data": [{"id": "alias", "max_model_len": 327680}]}), "alias", profile)
    with pytest.raises(ValueError, match="log mismatch"):
        RUNTIME.check_log(GOOD_LOG, profile)
    with pytest.raises(ValueError, match="alias/max_model_len"):
        RUNTIME.check_models(json.dumps({"data": [{"id": "alias", "max_model_len": 524288}]}), "alias", profile)


@pytest.mark.parametrize("defect", ["missing", "unknown", "other"])
def test_start_rechecks_saved_profile(root: Path, defect: str) -> None:
    env = stub_environment(root)
    originals = {name: (root / name).read_bytes() for name in FILES}
    assert manage(root, "apply").returncode == 0
    state = root / "logs/k1-promotion/files.json"
    record = json.loads(state.read_text())
    if defect == "missing":
        del record["profile"]
    else:
        record["profile"] = "unknown" if defect == "unknown" else other_profile(selected(root).name)
    state.write_text(json.dumps(record))
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0 and "previous configuration restarted" in result.stderr
    assert "CANDIDATE_START" not in (root / "events").read_text()
    assert all((root / name).read_bytes() == data for name, data in originals.items())


def test_start_refuses_configured_profile_tamper(root: Path) -> None:
    env = stub_environment(root)
    assert manage(root, "apply").returncode == 0
    path = root / "dsv41.env"
    path.write_text(path.read_text().replace(f"PROD_K1_PROFILE={selected(root).name}",
                                           f"PROD_K1_PROFILE={other_profile(selected(root).name)}"))
    before = path.read_bytes()
    result = command(["bash", "start.sh"], root, env)
    assert result.returncode != 0 and "K1 FALLBACK FAILED" in result.stderr
    assert "CANDIDATE_START" not in (root / "events").read_text()
    assert path.read_bytes() == before  # later edits never discarded


def test_runtime_cli_requires_explicit_profile(root: Path) -> None:
    result = command(["python3", "offload/promote/k1/check_runtime.py", "--log", "unused",
                      "--models", "unused", "--served", "alias"], root)
    assert result.returncode != 0 and "--profile" in result.stderr
