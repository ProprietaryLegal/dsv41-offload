# PP4 on WIN2 (R-PP4) — fairer, deeper PP4 measurement and the production comparison

NDM 2026-10-27: "Fairer win rule is essential", "Do better deeper testing and measuring", "go window after window".
Method: Fable review row 2 ("OFF/ON/OFF, pooled accepted-per-round … and pooled ms/round; short and 32K ×3 each; keep the
#61161 RNG test as a gate") and WIN2.md. Code: `pp4_measure_v2.py`, `pp4_window.py --measure-v2`, `pp4_rng_gate.py`,
`PP4-production-policy.env`, `pp4_v2_stage.sh`.

## Two registered modes (one OFF/ON/OFF HOLD each)

| Mode | Starts | Question |
|---|---|---|
| `controlled` | s1-A / s2-B / s3-A | fixed5 greedy drafts vs fixed5 sampled drafts on the PP4 runtime (Round5/6's question, re-decided) |
| `production-policy` | s1-P / s2-B / s3-P | does the PP4 promotion configuration (B = fixed5 sampled, `PP4-sampled.env`) beat **production's own policy** (adaptive K, `[[1,2,5],[3,4,4]]`, greedy drafts)? |

Role P (`PP4-production-policy.env`) is `PP4-sampled.env`'s greedy role A with production's policy left in place: the
five fixed-policy tokens are removed and `SPEC_OVERRIDE` is unset, so `serve_exp.sh` applies production's schedule and
`VLLM_DSV41_ADAPTIVE_K=1` / `AK_*` exactly as production `serve.sh` does. `runtime()` checks the P container env equals
production's plus only the PP4 worker identities, the greedy switch and the round-trace mount, and that its speculative
config is production's. **P is production policy on the PP4 runtime (PP4 K2 worker + verifier overlay), not the
production binary**; the exact-binary comparison is the promotion window's G6.

## Measurement

- Decision cells `short-b1` and `32k-b1` (B1, T=1, top_p=1, forced 512 tokens, fresh cache salt per stream). `32k-b4` is
  dropped so three repeats fit the HOLD.
- Per start and cell: six waves = three repeats × two waves (one per public source family, opinions/filings), seed
  42+wave matched across starts. Prompt order per family uses existing frozen cuts only (`ORDERS`).
- A repeat pools its two waves' server counters (`pp4_metrics.pool`): rate = (rounds + accepted) / decode seconds,
  accepted/round, ms/round; plus mean TTFT and client decode tok/s.
- Greedy (pp4-r14 1.1-nat; Round15 near-tie locator) on s2-B, and on s1-P in production-policy mode.

## Decision (win2-v1, paired by repeat)

Control = mean of the two bracketing starts in repeat r; candidate = s2-B in repeat r.

| Series per decision cell | Role |
|---|---|
| `pooled_rate_tps` | primary (WIN: point > 2 % and the 95 % paired-t interval excludes 0) |
| `accepted_per_round`, `decode_tps` | guards (can only cause LOSS: worse than control spread in ≥ 2 of 3 repeats) |
| `ms_per_round`, `ttft_s` | reported with their own win2-v1 verdict; never decide |

WIN2.md prefers ms/round as the primary for decode cells. PP4 is the change that deliberately spends ms per round
(+0.31 ms MEASURED) to buy accepted tokens (+6.9 % MEASURED offline), and the production-policy mode compares adaptive
depths 2..5 with fixed 5, so ms/round alone can never show PP4's effect and would read its cost as a loss. The rate is
(1 + accepted/round) / ms-per-round, so it charges that cost in full. `decision.checklist_literal` records what
ms/round-primary would have said. Status turns INCONCLUSIVE when the starts ran at different GPU clock caps (parsed from
each start's `clock cap N MHz` lines), at a cap other than production's configured one (each node's `/etc/environment`
GPU_CLOCK_CAP, else dsv41.env's 2150 default; 2300 since CLOCK-PERSIST), or THERMAL-WATCH dialled the clock back
inside the HOLD: the control is always a same-clock control. The archive's day-to-day spread for the primary (same
cell, max_tokens 512, production commit and clock cap only; WIN2-R2 rows carry `clock_cap_mhz` per start from that
start's own log, both Sparks agreeing) is reported next to the in-window OFF/OFF spread; the
2 % bar is never raised silently.

## Gates before any host contact

1. `--qualification` + `--qualified-commit`: the QUALIFIED admission receipt of the qualified commit, carried to HEAD only
   if every file outside the measurement layer is byte-identical (`CARRIED_FILES`/`CARRIED_PREFIXES`; that includes
   `pp4_cells.py` and `PP4-sampled.env`) and the runner's remote/runtime units are AST-identical (`CARRIED_UNITS`). The
   v2 budget must fit: total ≤ 85 min and measurement end ≤ 73 min.
2. `--rng-gate`: a PASS `pp4_rng_gate` receipt at HEAD (AUD1's six #61161 tests + four PP4 sampled-q multi-round tests,
   CPU interpreter, ~3 min) whose generated PP4 kernel sha equals the admission's prepared verifier. Every start's
   speculative config must be standard verification (`standard_spec`).

## ESTIMATED HOLD (registered round6 inputs; not measured v2 blocks)

| Mode | 333 s pair reserve | measurement ends | 600 s dry bound |
|---|---|---|---|
| controlled | 80.1 min | 68.1 min | 84.5 min |
| production-policy | 80.8 min | 68.8 min | 85.2 min (refuses; fits while the measured pair reserve ≤ 588 s) |

## Chain mode (WIN2-R2 `chain.py`)

- **Preflight.** `chain.preflight(serving, hold, placeholders_present=…)`. The placeholders are probed on both Sparks over
  an ordinary ssh (default tmux socket) only while the marker is present.
- **Chained start.** Production comes from `chain.production_snapshot()` (sha-checked against the marker) unless
  `--chain-production` names a capture; commit/tag are checked; both Sparks are proved quiet; the HOLD opens with
  `allow_unreachable=chain.allow_unreachable(mode)`.
- **End.** `chain.end_action(stop, hold)`:
  - `chained` (measurement window, no stop, not last): `chain_close` — default-socket hold placeholders started on both
    Sparks before the first fence (PP4's own placeholder lives on its hold scope's private tmux socket and dies with the
    fence; the shared `chain.CHAINED_CLOSE` cannot see it from inside a scope), then fences, pair close, experiment
    stop, quiet proof, production-DOWN proof, placeholder proof, scoped HOLD stash + commit-close, marker. Summary
    `chain: CHAINED`, restore status `CHAINED: production left down`.
  - `restore`: the strict W8 restore + watch. After the restore that ends a chain the runner takes the default-socket
    placeholders down (`window.sh close`'s job, which cannot reach them from PP4's scope) and clears the marker only on
    watch PASS.
  - `abort` (marker present and no HOLD of ours, e.g. a chained start that failed before its HOLD): `chain.abort`, whose
    summary is the window's `restore`; decision REJECT.
  - Admission and pair-only windows never chain (their receipts need the restore and watch).
- Gate the next window with `chain_prev_ok.sh` (it aborts the chain itself on any refusal while the marker is present).
- Run the PP4 test suites with `DSV41_CHAIN_MARKER` pointing at a scratch path so a live chain cannot change them.

## Coordinator launch (after an admission at the qualified commit is QUALIFIED)

```bash
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
H=<offload/pp4-win2 HEAD, full sha>; Q=<admitted commit, e.g. 85b0d96ddf5bd3c2870eae7af092d679b00a6c26>
A=<QUALIFIED admission window dir at $Q>
cd <clean worktree at $H>
# 1. #61161 gate (CPU only, once per commit)
PYTHONPATH=$PWD:$PWD/offload/dspark CUDA_VISIBLE_DEVICES= TMPDIR=$C/scratch/p4v2 /home/operator/miniconda3/bin/python3 \
  -m offload.launch.windows.pp4_rng_gate --execute --out $C/scratch/PP4-coord/rng-gate-${H:0:12}.json --basetemp $C/scratch/p4v2/rng
# 2. stage the sparkhead checkout at $H (between windows; copies $Q's prepared verifier)
WINDOW_COORDINATOR=Deepseek/coordinator bash offload/launch/windows/pp4_v2_stage.sh --execute $H $Q
# 3. dry plan, then the window (production-policy first; controlled may follow in the chain)
bash offload/launch/windows/pp4_window.sh --measure-v2 production-policy
bash offload/launch/windows/chain_prev_ok.sh "$PREV_TMUX" "$PREV_FILE"
WINDOW_COORDINATOR=Deepseek/coordinator WINDOW_CHAIN=1 bash offload/launch/windows/pp4_window.sh --execute \
  --measure-v2 production-policy --qualification "$A/admission-receipt.json" --qualified-commit $Q \
  --rng-gate $C/scratch/PP4-coord/rng-gate-${H:0:12}.json --expected-experiment-commit $H \
  --remote-repo dsv41-deploy-pp4-v2-${H:0:12} --worker-tree /opt/dsv41/x86host-data/pp4/admission-${Q:0:12}/repo \
  --worker-commit $Q --worker-identity be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd \
  --worker-markov-identity 0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a \
  --data-dir /opt/dsv41/data/bench-data --out $C/PP4-v2-prodpolicy-W-$(date -u +%Y%m%dT%H%M%SZ)
```

### Before any admission: the API rehearsal against production (queue line, between windows)

PP4-90f admission cmd/0132 stopped on HTTP 400: the R16 terminator probe forced its token with `logit_bias`, which the
image refuses whenever speculative decoding is on (image `vllm/sampling_params.py:1103-1116` `_validate_spec_decode`,
from `SamplingParams.verify` :850 via `v1/engine/input_processor.py:128`; 400 at
`entrypoints/serve/exception_handling/error_response.py:41-44`). Production runs the same check, so the probe, not the
candidate, was wrong; it now forces with `allowed_token_ids`. Every admission HTTP error now logs and carries the
server's body. Rehearse every admission API probe against production (tiny: ≤ 8 generated tokens each, priority 10, no
idle wait; one control request replays the R16 probe and must come back 400), and queue an admission only on PASS:

```bash
cd <clean worktree at the admission commit>
WINDOW_COORDINATOR=Deepseek/coordinator PYTHONPATH=$PWD:$PWD/offload/dspark CUDA_VISIBLE_DEVICES= \
  /home/operator/miniconda3/bin/python3 -m offload.launch.windows.pp4_api_rehearsal --execute \
  --url http://HEAD_IP:8000 --eos 1 --out $C/PP4-API-REHEARSAL-$(date -u +%Y%m%dT%H%M%SZ)
```

Exit 0 = PASS. `rehearsal.json` holds the verdict, `cause_confirmed`, the reading (4xx = probe wrong for this image;
transport/5xx = no answer, rerun) and every probe's status, error body, finish/stop reason, usage and hashes;
`responses.jsonl` (0600) holds the replies. `--eos 1` is the model's EOS from PP4-90f `admission-eos-source.json`.

Use `--measure-v2 controlled` and an `…-controlled-W-…` out dir for the other mode; add `WINDOW_CHAIN_LAST=1` on the
chain's last window. Every v2 window appends its finished control starts to `$C/control-archive.jsonl` (cells
`pp4rt-fixed5-greedy-*` / `pp4rt-production-policy-*`, `max_tokens` 512).

## Promotion

`PP4-PROMOTE-GATE-V2.md`: the LOSS semantics already match `offload/pp4-promote`; its `measurement_gate` needs four
changes to accept an R-PP4 measurement (metric set, carried-qualification binding, production-policy mode only, new
recorded fields).
