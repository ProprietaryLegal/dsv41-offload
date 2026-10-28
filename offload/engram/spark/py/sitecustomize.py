"""DSV41-OFFLOAD (lane O1): with DSV41_ENGRAM_OVERLAY=<path to offload/engram/spark/engram.py>, the module
vllm.models.deepseek_v41.common.engram is loaded from that file instead of the image's, in every Python process of the
container (PYTHONPATH points here; vLLM's worker processes inherit it). Without DSV41_ENGRAM_OVERLAY nothing changes.
Afterwards the distribution's own sitecustomize (Ubuntu: /usr/lib/python3.12/sitecustomize.py) runs as it would have.
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm.models.deepseek_v41.common.engram"
_OVERLAY = os.environ.get("DSV41_ENGRAM_OVERLAY", "")


class _EngramOverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        if not os.path.isfile(_OVERLAY):
            raise ImportError(f"DSV41_ENGRAM_OVERLAY={_OVERLAY} does not exist")
        return importlib.util.spec_from_file_location(name, _OVERLAY)


if _OVERLAY:
    sys.meta_path.insert(0, _EngramOverlayFinder())

# chain to the next sitecustomize on sys.path (the one this file shadows)
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    _f = os.path.join(_p or os.getcwd(), "sitecustomize.py")
    if os.path.abspath(_p or os.getcwd()) != _here and os.path.isfile(_f):
        with open(_f) as _fh:
            exec(compile(_fh.read(), _f, "exec"), {"__name__": "sitecustomize", "__file__": _f})
        break
