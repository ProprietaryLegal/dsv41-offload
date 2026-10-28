"""worker/p2p/prebuilt.py: the production unit loads only the pinned build of the p2p.cu it runs (S10 P-1), CPU only.

A stand-in extension module named dspark_p2p (built here with the system C compiler against this interpreter's
headers) plays the role of the nvcc build, so the load path itself is exercised, not mocked.
Run: python -m pytest -q worker/tests/test_p2p_prebuilt.py   (from offload/dspark)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

from worker.p2p import prebuilt

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")

STANDIN = r"""
#include <Python.h>
static PyObject* tag(PyObject* self, PyObject* args) { return PyUnicode_FromString(TAG); }
static PyMethodDef methods[] = {{"tag", tag, METH_NOARGS, ""}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "dspark_p2p", NULL, -1, methods};
PyMODINIT_FUNC PyInit_dspark_p2p(void) { return PyModule_Create(&mod); }
"""


def _build(tmp_path, tag: str) -> str:
    c = tmp_path / f"standin_{tag}.c"
    c.write_text(STANDIN)
    so = tmp_path / f"standin_{tag}.so"
    subprocess.run(["cc", "-shared", "-fPIC", f"-DTAG=\"{tag}\"", f"-I{sysconfig.get_paths()['include']}",
                    "-o", str(so), str(c)], check=True, capture_output=True)
    return str(so)


def _install(root, src: str, so: str, pin: str | None = None) -> str:
    d = os.path.join(root, prebuilt.source_key(src))
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, prebuilt.SO_NAME)
    shutil.copy(so, dst)
    with open(dst + ".sha256", "w") as f:
        f.write((pin or hashlib.sha256(open(dst, "rb").read()).hexdigest()) + "  dspark_p2p.so\n")
    return dst


@pytest.fixture(autouse=True)
def _fresh_module():
    saved = sys.modules.pop(prebuilt.MODULE, None)
    yield
    sys.modules.pop(prebuilt.MODULE, None)
    if saved is not None:
        sys.modules[prebuilt.MODULE] = saved


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "p2p.cu"
    p.write_text("// p2p source v1\n")
    return str(p)


def test_loads_the_pinned_build_of_this_source(tmp_path, src):
    root = str(tmp_path / "p2p")
    so = _install(root, src, _build(tmp_path, "v1"))
    mod = prebuilt.load_prebuilt(root, src)
    assert mod.tag() == "v1" and mod.__file__ == so
    assert prebuilt.load_prebuilt(root, src) is mod          # loaded once


def test_refuses_without_a_build_for_this_source(tmp_path, src):
    root = str(tmp_path / "p2p")
    _install(root, src, _build(tmp_path, "v1"))
    with open(src, "a") as f:
        f.write("// edited after the build\n")          # another p2p.cu: another key, no build
    with pytest.raises(prebuilt.PrebuiltError, match="build-p2p"):
        prebuilt.load_prebuilt(root, src)


def test_refuses_a_build_that_is_not_the_pinned_one(tmp_path, src):
    root = str(tmp_path / "p2p")
    dst = _install(root, src, _build(tmp_path, "v1"))
    shutil.copy(_build(tmp_path, "other"), dst)             # replaced after pinning
    with pytest.raises(prebuilt.PrebuiltError, match="is not the pinned"):
        prebuilt.load_prebuilt(root, src)


def test_refuses_an_empty_pin(tmp_path, src):
    root = str(tmp_path / "p2p")
    dst = _install(root, src, _build(tmp_path, "v1"))
    open(dst + ".sha256", "w").close()
    with pytest.raises(prebuilt.PrebuiltError, match="empty"):
        prebuilt.load_prebuilt(root, src)


def test_ext_uses_the_prebuilt_and_never_compiles(tmp_path, monkeypatch):
    import torch.utils.cpp_extension as cpp

    import worker.p2p as p2p

    root = str(tmp_path / "p2p")
    _install(root, p2p.SRC, _build(tmp_path, "prod"))
    monkeypatch.setenv("DSPARK_P2P_PREBUILT", root)
    monkeypatch.setattr(p2p, "_EXT", None)

    def no_compile(*a, **k):
        raise AssertionError("compiled despite DSPARK_P2P_PREBUILT")

    monkeypatch.setattr(cpp, "load", no_compile)
    assert p2p.ext().tag() == "prod"
