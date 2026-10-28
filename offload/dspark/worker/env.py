"""Process environment the worker's numerics depend on; imported by the package before any exllamav3 call.

The Spark serve runs with EXL3_INT8_GEMV=0 (exllamav3 1.6.0 otherwise sends one-row mul1 GEMVs through a plain-int8
activation path, ~0.7 % output error measured on these tensors, offload/results/o2/d1_exl3_identity.json) and
EXL3_MOE_COOP_WIDE=1 (the wide coop tile; already exllamav3's default below Blackwell). exllamav3 reads both with
getenv on first use, so they are forced here, at import, before the extension runs anything.
"""

import logging
import os

FORCED = {"EXL3_INT8_GEMV": "0", "EXL3_MOE_COOP_WIDE": "1"}

log = logging.getLogger("dspark_worker")

for _k, _v in FORCED.items():
    _old = os.environ.get(_k)
    if _old not in (None, _v):
        log.warning("%s=%s overridden to %s (the Spark's setting)", _k, _old, _v)
    os.environ[_k] = _v
