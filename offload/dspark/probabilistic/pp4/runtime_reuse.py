"""Scratch-only EXL3 object reuse, certified against complete Ninja dependencies.

Only translation units with byte-identical transitive inputs and identical
compiler arguments may reuse an object. The two known module definitions are
removed from comparison only when neither identifier appears in ANY input.
The numerical-header patch and pybind inputs automatically prevent reuse.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

ROOT = Path("/opt/dsv41/x86host-data/pp4/runtime")
STOCK = ROOT / "exllamav3"
FACC = ROOT / "exllamav3-facc"
MACROS = (b"TORCH_EXTENSION_NAME", b"PYBIND11_STDLIB")


@cache
def evidence(path: str) -> tuple[str, bool]:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), any(macro in data for macro in MACROS)


def compdb(build: Path) -> dict[str, dict[str, Any]]:
    result = subprocess.run(["ninja", "-t", "compdb"], cwd=build, check=True, text=True, capture_output=True)
    entries = cast(list[dict[str, Any]], json.loads(result.stdout))
    return {str(Path(entry["output"]).resolve()): entry for entry in entries}


def normalized(command: list[str], source: Path, build: Path) -> list[str]:
    return [arg.replace(str(build), "<BUILD>").replace(str(source), "<SOURCE>") for arg in command
            if not arg.startswith(("-DTORCH_EXTENSION_NAME=", "-DPYBIND11_STDLIB="))]


def certify(build: Path) -> Path:
    stock_build = STOCK / "build" / build.name
    stocks, faccs = compdb(stock_build), compdb(build)
    result = subprocess.run(["ninja", "-t", "deps"], cwd=stock_build, check=True, text=True, capture_output=True)
    dependencies: dict[str, list[str]] = {}
    for section in result.stdout.split("\n\n"):
        if not section.strip():
            continue
        lines = section.splitlines()
        if ": #deps " not in lines[0]:
            raise RuntimeError("unexpected Ninja dependency record")
        dependencies[lines[0].split(": #deps ")[0]] = [line.strip() for line in lines[1:] if line.strip()]
    approved: dict[str, object] = {}
    rejected: dict[str, str] = {}
    for output, entry in faccs.items():
        destination = Path(output)
        old_output = str(stock_build / destination.relative_to(build))
        previous = stocks.get(old_output)
        if previous is None or old_output not in dependencies:
            rejected[output] = "missing stock compile/dependency record"
            continue
        command = shlex.split(str(entry["command"]))
        old_command = shlex.split(str(previous["command"]))
        if normalized(command, FACC, build) != normalized(old_command, STOCK, stock_build):
            rejected[output] = "compiler argument difference beyond unused module definitions"
            continue
        inputs: dict[str, str] = {}
        reason = ""
        for name in dependencies[old_output]:
            path = Path(name)
            counterpart = FACC / path.relative_to(STOCK) if path.is_relative_to(STOCK) else path
            if not counterpart.is_file() or not path.is_file():
                reason = "missing transitive input"
                break
            old_hash, has_macro = evidence(str(path))
            new_hash, other_macro = evidence(str(counterpart))
            if has_macro or other_macro:
                reason = "module macro appears in transitive input"
                break
            if old_hash != new_hash:
                reason = "transitive input bytes differ"
                break
            inputs[str(counterpart)] = new_hash
        if reason:
            rejected[output] = reason
            continue
        approved[output] = {"object": old_output, "object_sha256": evidence(old_output)[0],
                            "command": command, "inputs": inputs}
    manifest = ROOT / "reuse-manifest.json"
    manifest.write_text(json.dumps({"approved": approved, "rejected": rejected}, indent=2) + "\n")
    print(f"PP4 exact-input reuse: {len(approved)} approved, {len(rejected)} rebuild", flush=True)
    return manifest


class NinjaModule(Protocol):
    _run_ninja_build: Callable[[str, bool, str], None]


def install() -> None:
    """Called only by the instrumented scratch setup.py, after its normal imports."""
    cpp = cast(NinjaModule, importlib.import_module("torch.utils.cpp_extension"))
    original = cpp._run_ninja_build

    def run(build_directory: str, verbose: bool, error_prefix: str) -> None:
        build = Path(build_directory).resolve()
        if not build.is_relative_to(FACC):
            raise RuntimeError("reuse hook is scoped exclusively to the scratch facc build")
        manifest = certify(build)
        ninja = build / "build.ninja"
        contents = ninja.read_text()
        wrapper = shlex.join([sys.executable, str(Path(__file__).resolve()), str(manifest), "--"])
        for compiler in ("$cxx", "$nvcc"):
            contents = contents.replace("command = " + compiler, "command = " + wrapper + " " + compiler)
        ninja.write_text(contents)
        original(build_directory, verbose, error_prefix)

    cpp._run_ninja_build = run


def main() -> None:
    manifest_path, separator, *command = sys.argv[1:]
    if separator != "--" or not command:
        raise RuntimeError("usage: runtime_reuse.py MANIFEST -- COMPILER ARGS")
    manifest = json.loads(Path(manifest_path).read_text())
    output = command[command.index("-o") + 1] if "-o" in command else ""
    approved = manifest["approved"].get(output)
    if approved is None:
        os.execvp(command[0], command)
    if command != approved["command"]:
        raise RuntimeError("reuse compile command changed after certification")
    # Recheck every dependency at use, to avoid a stale certificate.
    for path, expected in approved["inputs"].items():
        if evidence(path)[0] != expected:
            raise RuntimeError(f"reuse input changed after certification: {path}")
    if evidence(approved["object"])[0] != approved["object_sha256"]:
        raise RuntimeError("stock object changed after certification")
    shutil.copy2(approved["object"], output)
    escaped = [path.replace(" ", "\\ ") for path in approved["inputs"]]
    Path(output + ".d").write_text(output + ": " + " \\\n ".join(escaped) + "\n")
    print("PP4_REUSED " + output, flush=True)


if __name__ == "__main__":
    main()
