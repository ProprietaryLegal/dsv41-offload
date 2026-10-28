# 0075 positive successor reservation (UP2, default OFF)

The installed-image scheduler plus the process-aware UP1 composition now protects
admitted work before spending opportunistic rows. This replaces the stateless
zero decision; small chunks without a skipped predecessor are allowed.

With replay enabled, each ready running text request reserves min(demand,129,remaining model-length allowance)
rows and its draft slots at the start of the turn. Decode demand is included.
Earlier requests retain their order and may spend only the budget above later
reservations. A long-prefill threshold or later time prediction is a soft bound:
the protected minimum wins. A request with a skipped predecessor bypasses cadence
deferral for its next ready turn. Thus a ready admitted successor progresses in
one turn under unchanged resources, even when earlier greedy work would leave
64 or 128 rows. Running order and legal priority numbers are unchanged; ready resumed owners
retain their reserved admission capacity ahead of new work.

Before ANY skippable admission, both engine budgets must support all configured
active slots at 129 rows (plus draft slots in the input budget). Unsupported
capacity/alignment fails explicitly before a skip. The configured four slots and
4096-row policy support this reservation. The admission itself still passes real
KV allocation. After scheduling a potentially skipped chunk, its Request records
the expected end position as its successor obligation. This conservative record
also covers logprobs that do not actually skip. It cannot survive cancellation
into a new Request, even if an ID or engine slot is reused.

If KV allocation fails, ordinary preemption retains the obligation. If computed
positions change behind an obligation (backtracking/KV loss), the actual installed
preemption path retires the old worker request; ordinary hit lookup rewinds and
publishes replay_start on resume. A waiting request with a retained obligation
and a ready restored prefix also rewinds a real 128-row suffix, including after
async restore. Rebuilding may be chunked: reads are bounded by replay_start and
only computed rows are published. This safely replaces the old >128 obligation;
a smaller chunk cannot consume the skipped history. No cached-token counter or
zero-row ring rollback is treated as reconstructed taps.

Pauses, incomplete restores, pipeline completion fences, unavailable physical KV
and permanent resource loss cannot promise unconditional token progress. Peers
continue; existing completion/restore paths own readiness. Configuration failures
raise a named error rather than returning replay-policy zero indefinitely.

CPU evidence runs actual Scheduler.schedule, _preempt_request, _mark_prefix_replay
and sync/async bookkeeping through offload/dspark/d2/scheduler_cpu.py. KV allocation,
model completion and dependency construction are fake. This proves scheduler
statements and state boundaries, not native kernel parity or hardware liveness.

Reproduce on Controller, with no network or GPU:

```bash
source ~/.config/ndm/paths.sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TRITON_INTERPRET=1 \
  "$HARNESS_PYTHON" -m pytest -q -p no:cacheprovider \
  --basetemp=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/UP2-R2/pytest \
  offload/upstream_adopt/tests offload/prefill_trace/test_bounded_chunk.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  "$HARNESS_PYTHON" -m offload.upstream_adopt.build_overlay
```

The baseline had four pin-related failures: speculator.py had gained the merged,
default-off decode-capture hook (b42c0e6). Its refreshed pin preserves that exact
source; the capture hook imports its existing engine module only when explicitly
enabled. All other source pins, K1 bytes, zero-fuzz refusal and async package
contents are retained. The generated overlay is always rebuilt and hashed.

Final review follow-up: ready waiting owners with an outstanding obligation also
reserve a suffix-rebuild floor and an available admission slot before running
extras are allocated. New/unadmittable waiters cannot consume those reservations
or hide the resumed owner behind an allocation failure. The normal readiness
handler completes async restores before reservation; incomplete ones reserve
nothing and do not block ready peers. Running rollback recovery respects the
installed connector free fence and stale-output policy before preemption.

If a ready waiting owner proves physically unadmittable, its unused floor and
available slot pass to the next ready owed owner in that turn. This prevents
a missing-KV waiter from freezing a peer even with only one admission slot free.


Replay startup now requires the exact UP2 scheduler OVERLAYS entry and its
manifest hash, before any SSH, service stop or staging in start_exp.sh. Missing,
image/SO2, duplicate, ancestor-shadowing and unreviewed raw Docker mount substitutions fail
loudly. OFF does not invoke the guard. Both ranks still require the runtime hash
check after staging; this source preflight is not proof of a deployed mount.

SO2 and UP2 do not compose without one hand-composed scheduler and a fresh
independent check. SO2 quanta/pauses must exempt owed successors and its reserve
proof must cover the 129-row floors. Never pair a skipping model/worker with the
image or SO2 scheduler. Default-OFF and textually clean merges do not establish
semantic compatibility.

The successor suite runs both without drafts and with production DSpark K=5,
four extra drafting slots, KV lookahead 5, prefill lookahead 1 and async
placeholders. Decoder peers carry draft IDs. S27's K4/K5 near-model-limit probe
asserts ON == OFF under the installed clamp in sync and async scheduling.
