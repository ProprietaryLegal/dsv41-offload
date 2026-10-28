# SPDX-License-Identifier: Apache-2.0
"""Offline tests of the remote DSpark proxy (Controller, CPU only, no vLLM install).

    cd offload/dspark/spark && python3 -m pytest -q tests
"""

import os
import sys

# the fused Triton gather is checked on CPU through Triton's interpreter
os.environ.setdefault("TRITON_INTERPRET", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
SPARK = os.path.dirname(HERE)
if SPARK not in sys.path:
    sys.path.insert(0, SPARK)
