"""Production loading of the p2p extension from a prebuilt, pinned build (S10 P-1; R2-RUNTIME).

The development path (p2p.ext() without DSPARK_P2P_PREBUILT) compiles p2p.cu with torch.utils.cpp_extension at the first
use. A production unit must not run the compiler at start, nor pick up a build it was not qualified with. With
DSPARK_P2P_PREBUILT=<dir> the worker instead loads <dir>/<src16>/dspark_p2p.so, where <src16> is the first 16 hex digits
of the sha256 of the p2p.cu it runs, and only when the file's sha256 equals <dir>/<src16>/dspark_p2p.so.sha256 (written
by run/prod_tp2.sh build-p2p when it built that source). Anything else is a refusal naming the fix, never a fallback
to compiling.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys

SO_NAME = "dspark_p2p.so"
MODULE = "dspark_p2p"


class PrebuiltError(RuntimeError):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_key(src: str) -> str:
    return _sha256(src)[:16]


def prebuilt_path(root: str, src: str) -> str:
    return os.path.join(os.path.expanduser(root), source_key(src), SO_NAME)


def load_prebuilt(root: str, src: str):
    """The extension module from the pinned build of this p2p.cu under root; PrebuiltError otherwise."""
    so = prebuilt_path(root, src)
    pin = so + ".sha256"
    if not os.path.isfile(so) or not os.path.isfile(pin):
        raise PrebuiltError(f"no prebuilt p2p extension for {src} (source key {source_key(src)}) under {root}: "
                            f"build it with offload/dspark/worker/run/prod_tp2.sh build-p2p (expected {so} + .sha256)")
    with open(pin) as f:
        words = f.read().split()
    if not words:
        raise PrebuiltError(f"{pin} is empty")
    want = words[0].lower()
    got = _sha256(so)
    if got != want:
        raise PrebuiltError(f"{so}: sha256 {got} is not the pinned {want} ({pin})")
    if MODULE in sys.modules:
        mod = sys.modules[MODULE]
        if getattr(mod, "__file__", None) != so:
            raise PrebuiltError(f"{MODULE} already loaded from {getattr(mod, '__file__', None)}, not {so}")
        return mod
    loader = importlib.machinery.ExtensionFileLoader(MODULE, so)
    spec = importlib.util.spec_from_file_location(MODULE, so, loader=loader)
    if spec is None:
        raise PrebuiltError(f"cannot load {so}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[MODULE] = mod
    return mod
