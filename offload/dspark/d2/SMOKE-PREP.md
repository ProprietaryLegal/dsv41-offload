# W10 R2 smoke preparation

R2 supersedes the original one-sample gate. The new qualified input is
`/opt/dsv41/campaigns/DSV41-OFFLOAD/W10/private/smoke-input-r2-qualified.json`.
Its SHA-bound v2 receipt is `W10/private/r2-baseline/k1-request-receipt.json`.
The historical inputs, single-sample receipt and consumed send marker are retained.
Never re-run either reference collection: R2's durable marker is also consumed.
The sanitized measurement summary is `offload/evidence/D2-SMOKE-R2-baseline.json`.

Preparation selected two existing public D2 book/dev prompts, tokenized via the
idle-gated production endpoint, plus original W5/K1 row 4. No historical driver is
executed; its body is recovered from AST and its invocation/capture SHA verified.
Private files are mode 600 under mode-700 W10/private; raw prompts, token IDs,
responses and capture data never enter Git or reports.

The original greedy body omits top_p and priority (pinned defaults 1 and 0).
Keep that exact model/messages/template/sampling identity. Token-ID output and
stream serialization fields are permitted tracing differences. Tagged requests
retain top_p=1, priority=20 and their original 128-token budgets. Control budget
remains the historical 300; all three prompt lengths remain below 8192.

The R2 baseline collector (`greedy_reference --execute-r2`) has already sent six
original priority-0 requests: one untimed warm-up, then five uncontaminated repeats.
It refuses busy readiness snapshots, monitors foreign demand while each owned
HTTP child runs, cancels only that child on contamination and permits at most ten
attempts including warm-up/readiness failures. Missing/reset metrics or output
drift fail immediately. Attempts/checkpoints, raw metrics, responses and hashes
are private. The medians are TTFT 0.1146697998046875 s and mean engine-output step
0.037329789555062436 s; cached prompt tokens 0; nine output steps per request.

The collector's final runtime comparison initially refused unordered Docker mount
lists. Canonical ordering and raw initial runtime persistence are now fixed and
covered by tests. The saved six samples were qualified offline, with no resend.
For this actual collection the initial raw runtime snapshot was not retained:
initial K1 preflight passed, saved/fresh postflights agree after canonicalization,
and the metrics process start identity stayed constant in every interval and the
recovery check. That limitation is recorded in the receipt/runtime-recovery.json.
`qualify_saved` rechecks response hashes/UTF-8 bytes, raw metrics, cache conditions,
attempt ordering and all five timings; it has no HTTP/SSH implementation.

Candidate measurement:

1. Refresh HOLD, both-rank runtime/file identities and model root/context for the
   two tagged requests and three informational controls before group dispatch.
2. Run the two tagged requests together, keeping their nonstreaming bodies exact.
   Start each of three sequential informational greedy controls only while both
   futures remain in flight, both capture ledgers have grown, and fresh metrics
   show two running and zero waiting. This keeps occupancy at three (K1 has four
   sequence slots). Refuse overlap failures; never label a late control active.
3. Informational controls use streaming serialization, reassemble exact text/raw
   IDs and retain usage/prompt echoes. Report median client TTFT and mean interval
   between token-ID delivery chunks. These include transport/parser effects and
   are NOT engine steps or part of the latency gate. Never attribute overlapping
   aggregate histogram deltas to an individual request. The aggregate TTFT count
   must still match the exact successful group request count to reject foreign
   work. Chat top20 support also requires request-local draft-token statistics or a
   validated multi-position chat decode step (not prefill); another concurrent
   request's drafting cannot establish chat support. The prior live response has
   counters disabled, so the capture-ledger proof is required there.
4. Preserve existing response, top20, prompt-identity and capture checks. After
   both tagged requests finish, request the durable nonce-bound writer flush and
   validate zero errors and the exact captured sequence set.
5. With the writer idle, run one untimed identical control warm-up, then five timed
   isolated controls, refreshing HOLD/runtime/model before every request. Bracket
   each with idle metric snapshots and reject contamination, cache mismatch or
   output drift. Compare medians of TTFT and mean engine-output step against the
   five K1 repeats; BOTH must remain within the unchanged 5% bound.
6. Copy capture files via O_DIRECT, validate the original capture rows/ledger,
   detect zero/missing/extra rows or sequences, and recheck both container IDs.

There are three unique input types and eleven candidate HTTP generation calls:
two tagged, three informational, one control warm-up and five timed controls.
All nine candidate controls must match K1 UTF-8 bytes. If either tagged request
ends too early (including HTTP rejection), active-writer qualification refuses;
missing top20 or unproven chat drafting still select the existing replay
fallback when the active group otherwise completes.

The wrapper is installed byte-for-byte at
`/opt/dsv41/campaigns/DSV41-OFFLOAD/W10/smoke_window.sh`. Default invocation
remains entirely offline. `D2_SMOKE_REPO` selects a reviewed clean checkout. R2
qualified inputs are now the default. Scratch uses the campaign disk, never /tmp.

```bash
# Offline only; safe for implementation lanes:
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W10/smoke_window.sh
bash offload/dspark/d2/decode_smoke.sh --dry-run
```

Only Deepseek/coordinator may use `--execute`, after independent source review,
pausing generation drivers and saving/aborting their own streams. The existing
D2-GEN.PAUSE marker, SHA-bound inputs, exact clean source and NEW run/output are
required. The runner refuses an existing HOLD; records 300 seconds of continuous
zero gauges at gaps <=10 s; writes window.json; opens the established HOLD window;
exports an advertised-HEAD Git bundle, verifies its SHA at sparkhead, then fetches
locally into a clean detached reviewed checkout. No push or third-party fetch.

Both ranks receive three SHA-verified immutable source files and NEW empty
host-owned mode-700 capture directories. The original remote speculator prefix,
free capture disk >=1 GiB, and both-rank MemAvailable >=3 GiB remain mandatory.
Transfers/start/smoke retain STOP/memory supervision and bounded timeouts. The
candidate starts through start_exp.sh with D-decode-capture.env and the NEW run.
These service/window operations were NOT executed by the R2 implementation lane.

On completion/failure, cleanup requires a different-from-original full container
ID, pinned image and exact run mount, and rechecks that ID before stopping only
the candidate. Low-memory/STOP does not prevent owned-candidate cleanup. A
mismatched identity refuses cleanup. Partial staging requires a NEW run ID.
No x86host worker, unit or GPU operation occurs in this wrapper.

HOLD remains open. Restoration stays separate and is printed, never invoked:
W8/restore.sh start, check, close in that order, with coordinator review. An
unchanged original container is left alone. The real candidate latency, live
capture durability and informational overlap remain coordinator measurements.
