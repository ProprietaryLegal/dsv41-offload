# UP2 offline rehearsal and failure injection

MEASURED on Controller CPU, with synthetic responses and owned local subprocesses;
**no inference, Spark access, GPU access or service changes**. Default dry plan
and all execution refusals print the restoration commands. Rehearsal uses the
same `sequence()` as live operation, plus raw-command startup fakes and an HTTP
response fake for the actual request driver. No fake is exposed as a live CLI
adapter or accepted as native evidence.

```bash
source ~/.config/ndm/paths.sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TRITON_INTERPRET=1
UP2_SCRATCH=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/UP2-SCREEN-R2
mkdir -p "$UP2_SCRATCH"
export TMPDIR="$UP2_SCRATCH"
# Use a fresh clone of the committed reviewed lane; strip ALL inherited GIT_*.
(
  while IFS='=' read -r key _; do
    case "$key" in GIT_*) unset "$key" ;; esac
  done < <(env)
  git clone --shared --quiet "$PWD" "$UP2_SCRATCH/repo"
  cd "$UP2_SCRATCH/repo"
  "$HARNESS_PYTHON" -m pytest -q -p no:cacheprovider \
    --basetemp="$UP2_SCRATCH/pytest" \
    offload/launch/windows/tests offload/upstream_adopt/tests offload/prefill_trace
)
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases --follow-imports=silent \
  --cache-dir="$UP2_SCRATCH/mypy" \
  offload/launch/windows/up2_{checks,request,proxy_stats,low}.py \
  offload/launch/windows/tests/test_up2_low.py offload/launch/windows/tests/test_s30_probes.py
bash -n offload/launch/windows/up2_low.sh
bash offload/launch/windows/up2_low.sh --dry-run
bash offload/launch/windows/up2_low.sh --dry-run --screen --phase diagnostics
bash offload/launch/windows/up2_low.sh --dry-run --screen --phase measured --contexts 300000
bash offload/launch/windows/up2_low.sh --dry-run --screen --phase measured --contexts 150000
# Delete only this lane's scratch after saving evidence outside it.
```

MEASURED UP2-SCREEN-R2 final integration:341 tests +4 subtests pass; strict
mypy passes on6 files; shell syntax and offline plans pass. Historical
UP2-SCREEN integration had310 tests +4 subtests; S30's fixes added5 tests. This run includes the full prefill suite, including the CPU
trace tests (the earlier310-test run included only bounded-chunk/window tests).
Immutable42-file build/zero-fuzz composition and source pins remain unchanged.
Isolated test children may create bytecode in the scratch clone's overlay; use a
fresh clone or remove only those generated files before repeating immutable
verification. Never delete a shared tree's artifacts. No source pin/assertion is
relaxed. Logs: campaign UP2-SCREEN-R2.tests.log and UP2-SCREEN-R2.types.log.

SCREEN rehearsal uses the actual missing-packet implementation inside the fake
sequence. The fake is synthetic, not a live CLI adapter. An explicit flag is
required in every window; diagnostics produce PASS-SCREEN/native NOT RUN when
evidence is absent, and every per-request result, needle result, ledger,
checkpoint and summary carries screen true/adoptable false/SCREEN level.
Raw command/memory logs remain provenance records, not qualification receipts.
Current audit counters provide round counts but no chunk q; missing q therefore
records ESTIMATED4092 and unverified tails, never a manufactured MEASURED q.
A supplied measured q packet is used and cited even without a complete packet.
Supplied invalid native receipts stop in both modes, including when rechecking
prior qualification beside a NOT RUN record.

| Injection / scope | Expected outcome; exercised test |
|---|---|
| Missing/conflicting coordinator identity | STOP before remote operations; dry/abort and conflicting-coordinator tests |
| Invalid CLI | Restore commands still printed; invalid-CLI test |
| Old/future HOLD timestamp | STOP; `test_deadline_comes_from_hold_and_never_renews` proves runner reinvocation cannot renew it |
| Worker revision, target metadata, image, engine, env/K1, argv drift | STOP; raw-command start and runtime fixture tests; worker mismatch refuses before start |
| Cap !=327680, KV !=2502600, hash verifier failure, health failure | STOP; `test_real_startup_route_hard_failures_stop` |
| Replay startup guard absent/substituted/SO2/duplicate/shadowing/hash drift | STOP; existing16 replay-guard tests included in integration |
| Engine exception, invalid provenance, missing final taps, PARTIAL_WINDOW, NaN | STOP; error-log and every-hard-counter tests |
| Proxy torch, wrong process/capacity/mlock, dead proxy, watchdog not running | STOP; proxy gate tests |
| Async watchdog fire (late reply, target-only round; 2/5 fresh K1 starts fire once at round 1) | RECORDED in proxy-stats.jsonl, never STOP; stale/provenance/suffix gates stay hard (S30 FIX-2) |
| MemAvailable below3GiB on either rank, missing/EOF/stale monitor | STOP, owned child reaped; local monitor fake and memory-floor tests |
| SwapFree decrease (every GB10 model load swaps out ~4 GiB: W8/W10 MEASURED) | RECORDED in memory-<host>.jsonl, never STOP (S30 FIX-3) |
| Measured-only context subset; defaults both |9 cells/12 submissions for300K or150K;18 cells/21 submissions for both; context-subset tests |
| Mixed/absent window identity, unexpected contexts, duplicate cells | STOP; `test_summary_refuses_spliced_arms_and_contexts` |
| Stage B without300K or without all3 usable pairs | INCOMPLETE record; context-subset and Stage-B tests; timing never STOP |
| Submission cutoff in request child; hard reservation bound | STOP names exact submission/hard deadline; deadline propagation, actual request refusal and hard-bound tests |
| Unrelated failed child after deadline machinery added | Original failed-rc error retained; unrelated-child control |
| External STOP, owned command failure/timeout | STOP, only owned group signaled; supervisor tests |
| Stats layout mismatch or absent flush ACK | STOP; actual owned memfd with housekeeping thread; offset constants checked against reviewed source |
| Native source/container/evidence hash/q/tail/audit mismatch | STOP; native identity and artifact-tamper tests |
| Missing native packet or bounded checkpoint arrival | Default NOT RUN/HOLD; supervised wait accepts only a timely packet; checkpoint tests |
| Each of12 mandatory native safety checks is NOT RUN | NOT RUN/HOLD; parameterized test removes each in turn |
| `--screen`, missing native q and complete packets | Continue all API gates and 18 timing cells; all 12 native checks NOT RUN, PASS-SCREEN, adoptable false; `test_screen_missing_native_continues_all_cells_and_labels` |
| `--screen`, supplied bad native schema/check/hash/q/tails | STOP through unchanged validator; `test_screen_supplied_bad_packet_still_stops` |
| Supplied measured q but missing complete packet | Use measured q and cite q.json; native remains NOT RUN, tails unverified; measured-q screen test |
| PASS-SCREEN consumed without `--screen` | STOP before remote operations; prior qualification test |
| Screen prior artifact/source drift or bad supplied receipt beside NOT RUN | STOP; prior artifact and supplied-receipt tests |
| Screen result files, checkpoints, request and needle ledgers | `screen: true`, `adoptable: false`, SCREEN label; full-sequence and HTTP-driver tests |
| Screen health/cap/worker/hash/KV/engine/identity and sequence failures | STOP as on default path; parameterized startup/sequence tests |
| Cold cache hit, wrong cap/uncached count, paired prompt/salt hash drift | STOP; cold/pairing tests |
| Numerical common mass<.99 or KL beyondmax(.005,2*A/A) | STOP for full-logit diagnosis; numerical test |
| Needle5/6 or foreign code | STOP; actual needle driver with HTTP fake; passing6/6 also exercised |
| Timing100x slower, batch text differences or large batch numeric deltas | RECORDED, never STOP; complete-sequence fake |
| Missing/invalid timing sample | RECORDED, ratio null; no STOP |
| Sampled-first numerical rows | Accepted independent of list order; numerical and HTTP-driver tests |
| Best-effort shortfalls0/1/2 controls | Counts equal actual ledger events; no assumed3-request gate |
| Partial matrix failure after first cold cell | Durable checkpoint contains1 cell;31 attempts/30 completions/1 failure in synthetic full-sequence rehearsal |
| Transport attempt without opened HTTP response | Unresolved/dispatch-uncertain, never fabricated completed or canceled |
| Stream missing DONE/usage/finish | Failed ledger event, no successful result file; real request driver HTTP fake |

The full synthetic sequence has5 starts,18 measured-matrix cells and49 inference
submissions including warmups/diagnostics/needles. Those are MEASURED CPU fake
counts, not native inference counts. Only sent events enter accounting; unsent
optional probes enter `not_sent` records. `send_intent` is a transport attempt;
`http_opened` proves response headers arrived. A crash between them is explicitly
uncertain and requires coordinator reconciliation, never an invented cancellation.
Native apparatus supplies its own request ledger in the qualification packet.

The selected single-context measured sequence has3 starts,3 warmups,9 cold cells
and12 submissions, for2,796,000 prompt tokens /13,596 max output at300K, or
1,446,000 /13,596 at150K. These are **MEASURED CPU synthetic schedule counts**.
Default both-context schedule remains18 cold cells/21 submissions; diagnostics
remain28 submissions. `window_id` binds cells to their fresh output directory;
paired hashes also remain hard. No arms or contexts are combined across windows.

**ESTIMATED** S30 measured-input budgets replace the older PF1 broad estimates:
diagnostics62–68min,300K55–59min, optional150K43–48min, legacy combined measured
75–81min. Add ~7min ESTIMATED restoration plus checks: upper-end totals75/66/55min
leave15/24/35min to the90min bound before extra checks. The legacy combined upper
81+7=88min leaves2min and exceeds75min submissions; do not schedule it.
Diagnostics+300K117–127min before restores cannot share one90min HOLD.
See UP2-LOW.md for measured source inputs and all75/90min margins. Audit overhead
is unmeasured. Fixed bounds never renew. Each window needs a NEW HOLD, same
reviewed commit/config/worker/diagnostic qualification and fixed salt prefix.
No prompt cache was added; repeated tokenization cost remains in the estimates.
Native wait is not included in SCREEN. No local inference gain is claimed.
