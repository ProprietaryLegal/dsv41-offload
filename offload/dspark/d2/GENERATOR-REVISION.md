# D2 reviewed generator revision boundary

D2-FIX1 supersedes S14 report §8.4 step 1's byte-equality rule for the NEXT reviewed restart only.
The currently running e2b2186 generator is not modified or stopped by this lane. This branch is a
candidate revision, `d2-priority20-v1`; independent review and coordinator integration precede use.
Pin the accepted commit and SHA256 of generate.py, prompts.py and this file in the restart receipt.
Never restart from a moving checkout; the old e2b2186 files remain a reproducible historical control.

MEASURED (CPU, image-pinned `d2/tests/test_priority.py`): the actual Request, priority queue,
Scheduler.schedule and AsyncScheduler classes from src-image order waiting work 0, 10, 20.
The former 5 would precede 10. The image scheduler already includes patch 0031c; do not apply it twice.
With the documented priority floor 5, priority0 demand defers priority20 prefill on throttled steps;
priority10 does not. Running requests are visited before waiting admission; numeric priority does
not instantly preempt an active prefill. No source statements are changed by the CPU fixture.
Fake unlimited KV and simulated completion cannot establish real latency or memory headroom.

Three auxiliary prefills at the production 4096 input budget / 1024 chunk limit leave 1024 for an
arriving priority10 request in the fourth slot without speculation (sync and async CPU cases).
With DSpark K=5, four additional draft slots per request leave 1008 legal input tokens; admission
still succeeds. The fixture tests both budgets explicitly. Cap and default are
therefore both three; lower via --streams if production contention warrants it. Legal 0/10 remain
unchanged. There is no SO1 exclusion or training-as-foreground pause rule. Priority20 alone does
not protect priority10; the driver's foreign-demand cancellation is required.

On the first observed positive foreign-demand poll, stop submissions and abort all own streams.
/metrics blindness, PAUSE, HOLD, stale HOLD, time-cap exhaustion and TERM/INT also cancel own work.
Detection uses aggregate gauges, with the existing 3-second ended-stream grace; a foreign arrival
inside that grace can be masked until it expires. This is bounded polling, not instant preemption.
Close the HTTP socket, join own threads, append received committed token IDs and the segment
(start, n, finish, usage, priority, generator revision), then atomically save the sequence. Keep
partials; resubmit only after 120 seconds quiet, using original prompt IDs + received generated IDs.
Do not re-render a partially generated prefix or reuse uncommitted draft proposals.

Checkpoint: every 30 seconds save received IDs and the time budget atomically. Clean cancellation
saves immediately; process death may lose the suffix since the last checkpoint, which is never
claimed as committed training data. A restart changes running to partial/pending and retains the
saved Spark-hours cap. Legacy segments without priority/revision are historical e2b2186 priority5;
new segments record 20 explicitly. Never relabel an earlier segment as 20.

Coordinator apply boundary: PAUSE using the existing window protocol, confirm aborted events and
server drain, end the old driver cleanly, record its budget and file hashes, then restart from the
reviewed immutable revision with the same prompts/state and existing cap. Only the coordinator
may perform this on the live state. This lane issues no Spark requests or state writes.

## Decode-capture second pass (2026-10-24)

The W9 equivalence gate failed. This candidate is `d2-decode-capture-v1`, based on
the reviewed priority20 revision including bounded admission and grace hardening.
The request tag is now `d2c.<pid>.<seg>.<max(0,len(prompt_ids)-128)>`. On resume,
`prompt_ids` includes the committed partial; the overlay retains the original
storage origin and first computed rows while the resume's tail is recomputed.
Priority20, cap3, first-positive abort, quiet/backoff, budget/checkpoints and
sampling remain as reviewed. No generation-logprob option is enabled by default;
use the top-20-only replay path until the coordinator's three-prompt smoke verifies
option (ii). Do not infer support from an HTTP 200.

This is NEW generation, never a restart of the historical first-pass state.
The CLI refuses `/opt/dsv41/data/d2/gen`, including symlink aliases. The
coordinator reserves a fresh private state directory such as
`/opt/dsv41/data/d2/gen-decode-REVIEWED_SHA-RUN_ID`; verify it does not exist
before first launch. Never copy the old seqs/budget into it. A restart may reuse
only that second-pass directory, with the same capture tree and immutable source.
The first command explicitly supplies the approved `--spark-hours` cap; subsequent
restarts preserve it. Record the code commit and generate.py/prompts.py/this file
hashes plus overlay/capture hashes and the production apply receipt.

A fresh serving restart usually empties prefix cache, but does not prove every
prompt tail was computed: shared prefix hits can still omit rows. Ledger coverage
must cover the original r0 through L_gen-2; a gap is a refusal. Do not fill it with
replay taps after a failed equivalence gate. Freeze/copy the capture only after all
tagged streams are drained and a new FLUSH nonce is acknowledged. See
DECODE-CAPTURE-APPLY.md for the coordinator boundary.

## Option-ii collector revision (2026-10-25)

`d2-decode-capture-option-ii-v2` adds explicit `--top20 generation` for the
coordinator's second pass. Default requests remain byte-identical. See
[OPTION-II.md](OPTION-II.md) for the pinned completions-format proof, censored
outside rank, atomic token/row checkpoints, fail-closed PAUSE, frozen exporter
and stronger generation-source verification receipt. New mode cannot be added
to a state that already contains off-mode generation. Historical revision
instructions above remain context; this boundary governs the next reviewed start.
