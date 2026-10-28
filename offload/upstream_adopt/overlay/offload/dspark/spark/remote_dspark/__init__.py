# SPDX-License-Identifier: Apache-2.0
"""Remote DSpark drafter, Spark side (DSV41-OFFLOAD lane O3).

Selected only by VLLM_DSPARK_REMOTE (see config.py); with it unset the image's own
speculator runs unchanged. Contract: offload/dspark/CONTRACT.md. Modules:
  wire        message codec (no torch)        state     window/slot bookkeeping (no torch)
  transport   TCP stub + transport API        o1_adapter  lane O1's ofl library
  client      session / HELLO / STEP          config    environment
  gather      rank-0 row gather, ring, staging (torch)
  speculator  the vLLM speculators (capacity / resilience / shadow)
  mock_worker reference worker for tests (and for lane O2)
"""
