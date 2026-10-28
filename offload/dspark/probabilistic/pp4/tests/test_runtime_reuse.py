"""Certification refuses changed numerical headers, module macros and flags."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from offload.dspark.probabilistic.pp4 import runtime_reuse as reuse


@pytest.mark.parametrize("difference,approved", [("none", True), ("header", False), ("macro", False), ("flags", False)])
def test_exact_input_certificate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 difference: str, approved: bool) -> None:
    stock, facc = tmp_path / "stock", tmp_path / "facc"
    stock_build, facc_build = stock / "build/temp", facc / "build/temp"
    for path in (stock_build, facc_build):
        path.mkdir(parents=True)
    old_source, new_source = stock / "kernel.cu", facc / "kernel.cu"
    text = "int kernel = 1;\n" + ("// PYBIND11_STDLIB\n" if difference == "macro" else "")
    old_source.write_text(text)
    new_source.write_text(text)
    old_header, new_header = stock / "numerics.cuh", facc / "numerics.cuh"
    old_header.write_text("int precision = 32;\n")
    new_header.write_text("int precision = 16;\n" if difference == "header" else old_header.read_text())
    old_object, new_object = stock_build / "kernel.o", facc_build / "kernel.o"
    old_object.write_bytes(b"compiled exact numerical object")
    monkeypatch.setattr(reuse, "ROOT", tmp_path)
    monkeypatch.setattr(reuse, "STOCK", stock)
    monkeypatch.setattr(reuse, "FACC", facc)

    def database(build: Path) -> dict[str, dict[str, Any]]:
        if build == stock_build:
            return {str(old_object): {"output": str(old_object), "command":
                    f"nvcc -O3 -DTORCH_EXTENSION_NAME=stock -c {old_source} -o {old_object}"}}
        optimization = "-O2" if difference == "flags" else "-O3"
        return {str(new_object): {"output": str(new_object), "command":
                f"nvcc {optimization} -DTORCH_EXTENSION_NAME=facc -DPYBIND11_STDLIB=facc -c {new_source} -o {new_object}"}}

    monkeypatch.setattr(reuse, "compdb", database)
    dependency_record = f"{old_object}: #deps 2, deps mtime1 (VALID)\n    {old_source}\n    {old_header}\n\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout=dependency_record, stderr=""))
    result = json.loads(reuse.certify(facc_build).read_text())
    assert bool(result["approved"]) is approved
    assert bool(result["rejected"]) is not approved
    if approved:
        assert result["approved"][str(new_object)]["inputs"] == {
            str(new_source): reuse.evidence(str(old_source))[0],
            str(new_header): reuse.evidence(str(old_header))[0]}
