Status 2026-10-23: Capacity + async process proxy in production since W5 (main 3c1dc6b / prod-async-20261023); earlier sync/thread controls below are historical. See [live roadmap](../../../documents/plans/roadmap.md) and [measured record](../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Remote DSpark — Spark side (lane O3)

This directory is the engine overlay that sends each decode round's draft inputs to the DSpark worker on x86host and
feeds the returned proposals to the Sparks' unchanged verifier.
- Contract: `../CONTRACT.md`
- Coordinator checklist: `../INTEGRATION.md`
- Lane report: `/opt/dsv41/campaigns/DSV41-OFFLOAD/O3.report.md`

| path | what |
|---|---|
| `overlay/vllm/v1/worker/gpu/spec_decode/__init__.py` | the image file plus one guarded branch; inert unless `VLLM_DSPARK_REMOTE` is set |
| `remote_dspark/wire.py` | message codec (CONTRACT §6); numpy-free for HELLO/PING/STATS |
| `remote_dspark/state.py` | window range rule (CONTRACT §7.2) and slot table, shared with the reference worker |
| `remote_dspark/transport.py`, `o1_adapter.py` | the contract's TCP stub and lane O1's `ofl` (RDMA/TCP) |
| `remote_dspark/client.py` | session, seq, HELLO, background reconnect, STEP exchange |
| `remote_dspark/gather.py`, `fused_gather.py` | rank-0 row gather (trailing 128 rows, rejected rows dropped), device ring, pinned staging; one-launch Triton variant |
| `remote_dspark/round.py`, `streamsync.py` | one round: rank 0 RPC + TP broadcast; async variant with a stream flag |
| `remote_dspark/breaker.py` | round-count circuit breaker (CONTRACT §10): trip after 3 failed attempts, 256 rounds without network, one probe |
| `remote_dspark/speculator.py` | vLLM speculators: capacity / resilience / shadow |
| `remote_dspark/mock_worker.py` | reference worker (fake drafter) over the TCP stub or `ofl` |
| `tools/identity.py`, `tools/probe.py`, `tools/bench_round.py` | identities from S2's receipt; pre-flight probe; overhead bench |
| `launch-configs/*.env` | configs for `offload/launch/start_exp.sh` |
| `tests/` | offline tests (CPU only): `python3 -m pytest -q tests` |

To run the reference worker on x86host (fake drafts, for plumbing checks only; it never replaces O2's worker for
parity or timing):
```
python3 -m remote_dspark.mock_worker --listen 10.202.28.1:18620 \
  --identity be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd \
  --markov-identity 0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a \
  [--ofl-dir ~/dsv41-offload/o1/bin-with-ofl.py]
```
