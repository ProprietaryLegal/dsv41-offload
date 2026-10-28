# PP4 Round 13 coordinator sheet

## Round 13 — NDM's 0.9-nat near-tie

NEW PP4 experiment checks register `pp4-r13-near-tie-0p9`. NDM's instruction
"You can loosen the nats to .9" sets the limit to 0.9 nats, with the same
1e-3 rounding allowance as NBW. This replaces Round 12's grid-step rule;
NBW/TV1, the worker and strict W8 remain unchanged.

At the first differing token, both compared tokens must be in the fresh-cache
top20. Their absolute logprob difference must be <=0.9+1e-3 nats, and the
BETTER of the two must be within 0.9+1e-3 nats of the top. Both conditions
are required independently. The experiment token need not itself be top.
Length/prefix differences and malformed/missing evidence still STOP.
No logit-grid or divergent-prompt-count condition remains: four or all six
near-ties may pass if every divergence satisfies the registered conditions.

Keep the full returned top20, both token IDs, first-divergence index, token
counts, unrounded margin, better-token-to-top gap and experiment-to-top gap.
Generation and teacher-prefix probe retain DISTINCT per-request cache_salt
values, with output-path/run/arm namespace plus fresh invocation nonce.
Both generation endpoints enforce nonempty salts before transport; retries
cannot reuse generation/probe KV. Prompt/output text remains memory-only.

The real admission margins 0 and 0.5 nats satisfy the numeric limit; 1.0 nats
STOPS. Tests use explicitly synthetic supplemental top20 because the original
receipt lacks it. Historical failed admission remains FAILED; no retroactive
qualification follows. Fresh coordinator admission, all remaining gates,
six DISTINCT EXACT W8 IDENTICAL results and PASS post-close watch remain
required. This source change claims no live speedup or promotion.

## Historical Round 12 coordinator sheet — superseded by Round 13 above

## Round 12 — PP4-only greedy registration

NEW PP4 checks register `pp4-r12-logit-grid-tie`. This supersedes the inherited
0.501-nat experimental allowance below for PP4 only. NBW/TV1 and strict W8 are
unchanged. Read `offload/dspark/probabilistic/pp4/LOGIT-GRID.md` for source
evidence, exact historical limitations and the next-window causal discriminator.

At the first differing token, PP4 saves the complete returned top20 by token
ID, both compared IDs, unrounded margin, experiment-to-top gap, observable step
and all-pairs residual/FP32 rounding bounds. The smallest positive returned gap
must witness a binary-power grid <=1 nat; every one of its 190 pairwise gaps
must lie on that grid within the two operands' FP32 ULP sum. That rounding
bound must itself be <step/16. No GCD rescue or 0.501 threshold remains in PP4.
Both tokens must be present, the experiment token must equal the top exactly,
and its gap to the reference must be <=one step plus FP32 subtraction rounding.
Off-grid, clamped/nonfinite/missing values, an unidentifiable/all-equal grid,
unequal token lengths/prefix-only differences, or >3 divergences among the six
prompts STOP. Partial failure diagnostics persist before the STOP.

Each invocation derives a run/arm namespace from its output path and adds a
fresh nonce. Every greedy generation has its own nonempty cache_salt; each
teacher-prefix probe has a DIFFERENT salt, so it cannot reuse the generation's
KV. PP4 Http enforces nonempty salts before either generation endpoint's POST.
The six prompts, target settings and reference bytes are unchanged; prompts
and generated text stay in memory. No additional live request or runtime
instrumentation is added beyond the existing first-divergence probes.

Historical admission `PP4-429-admission-W-20261027T133630Z` remains FAILED:
it lacks top20 evidence and never checked prompts3–5. The real receipt's
0/.5/1 cases are tested using explicitly synthetic supplemental top20 values,
not retrospectively certified. X86host probe source remains applicable because
Round12 changes NO file below offload/dspark/worker. A fresh coordinator-only
admission still requires all gates, strict six distinct W8 IDENTICAL results
and PASS post-close watch. No measurement or promotion is authorized here.

## Earlier Round 9 coordinator sheet (retained)

## Round 9 — per-family public cuts

**Ready for independent source review.** Continue from 399d8b7; Round 8 is parked
on offload/pp4-r8-reserve and is NOT included. This round changes only public
cut allocation, its metadata/validation, cold-salt assertions and regressions.
No staging, precheck, live request or hardware work was performed by this lane.

The coordinator reports fresh physical PAIR_QUALIFIED followed by admission
PP4-399-admission-W-20261027T082144Z stopping at public-cuts, then strict 6/6
restoration/watch PASS. Those are coordinator observations, not new lane
qualification. Earlier failed-pair statements below describe historical work.
This repair is not a successful admission receipt or a performance result.

Both admission and measurement use the SAME pp4_cells.build/texts_from layout
and file/body pins. The old shared offsets reached 5,250,000 while the filings
body has only 1,592,473 characters. Round 9 registers
`pp4-r9-per-family-public-cuts-v1`: four short keys followed by 20 long keys.
All 24 keys are retained for EACH family, including currently unused keys.

| Family | Body characters (header excluded) | Reserved spans | Final start |
|---|---:|---|---:|
|opinions|10,195,729|Disjoint; cumulative 8,192 short/256,000 long reservations|4,896,768|
|filings|1,592,473|Overlap required; uniform 58,107 character start spacing|1,336,461|

Every start is strictly increasing and at least 4,096 characters apart. A full
layout reserves 5,152,768 characters, so disjoint filings cuts are impossible
under the conservative bound. Overlap is recorded explicitly and permitted
only in that case. The final filings reservation ends at 1,592,461, inside its
own body. No corpus is repeated/padded. Text hashes remain distinct.

Each target reserves **8 characters/token**. MEASURED offline on the real public
files with `/opt/dsv41/data/d2/tokenizer/tokenizer.json`, SHA256
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`:
ALL 319 opinions/50 filings nonoverlapping 32,000 character chunks have maxima
4.656577415599535/4.5577553055120354 characters/token. Eight reserves more than
1.7 times the observed maximum. Every proposed full reservation was separately
tokenized offline: the smallest 32K reservation contains 60,929 opinions or
63,505 filings plain-text tokens. All 48 actual offline cuts converged within
the unchanged four-call search and token band. This is a conservative planning
bound with CPU evidence, not proof of serving chat-template token counts.
The real builder still counts the complete served chat and stops outside its
2% band or if any retry would exceed its reservation; it never crosses it.

Frozen source file SHA256:

```text
opinions_cl.jsonl  80461203f88a6b837eed8fdf9b7e77a40660acb36dfa9887e00a0ae37483b6a1
filings_recap.jsonl b1f2744634cddb0716074c451df5f46e2da3d851324ab794f0caff05ab7202a9
```

Shuffled body SHA256 (seed 42, same public header/order as before):

```text
opinions ce1a8a50d0e5e490bcc72e8f444da874e0c10e9d25aa76c18e012b27a5517812
filings  f92ce1677d0f523fb037ef858dbb69b1e2a6ca2b868b4e097c4fb82bd762170a
```

cuts.json records each family's source/body hashes, layout registration,
reserve/spacing/overlap, and each cut's offset/characters/tokens/text hash.
Re-rendering refuses source, plan, offset, reserve or text-hash drift. Old cuts
must be rebuilt with the reviewed source. Cold per-stream cache_salt is required
before HTTP submission and seeded forwarding, including warmup; actual driver
tests assert distinct salts across all 48 OFF/ON/OFF streams plus warmup. Existing
cached-token-zero and cross-start salt-uniqueness checks remain.

`tests/test_pp4_round9.py` reads the REAL hash-pinned files at
/opt/dsv41/data/bench-data for both families and EVERY prompt key.
It checks the entire 8 character/token span without a tokenizer dependency/server;
missing or changed files FAIL, never skip. Other tests cover independent builds,
refusals and real shared request threads with a fake API. No corpus/prompt or
generated text is committed. Bounds, budgets, sampler/capture, ownership,
strict W8 and post-close watch are unchanged from 399d8b7.

## Earlier qualification and measurement contract (retained)

**Ready for independent source review; hardware HOLD.** The capture repair and
bounded measurement design are CPU-tested. The coordinator's previous physical
pair HOLD FAILED and remains failed. No repaired physical-pair, admission or
performance result is claimed. No staging/precheck/window was executed by PP4.

Round6 supersedes Round5's105.55min measurement STOP. Admission still reserves
ESTIMATED75.82min. Measurement now fits conditionally from a successful admission
HOLD's actual pair duration, with deterministic B4 omission when necessary.
Missing admission evidence refuses before host contact; no timing is invented.

## Round8 hang diagnostics and hypothesis

The399d8b7 pair-only correctness child timed out at240s with an empty log.
The coordinator's later `diag-20261027T062939Z` selector diagnostic returned0
with all32cases/ten changed-input replays/both ranks exact, without a traceback.
It does not pin the earlier hang or supply the complete source-bound physical
qualification/admission receipt. Details and primary hashes: `PAIR-HANG.md`
in `offload/dspark/probabilistic/pp4/`.

The probe now emits flushed BEGIN/END/FAIL phases from imports through output,
including each eager/captured position, each replay rank and idle/result checks.
A native30s repeating faulthandler timer starts BEFORE torch import only in
the probe CLI; its parent60/240s bounds and all HOLD/restoration rules remain.
Keep both stdout/stderr and the last phase/traceback in the next reviewed run.
`pair_diagnostics.py` is included in the primary source hashes.

Separate **HYPOTHESIS** correction: compile-only Triton warmup leaves CUDA
launch handles lazy. The sampled host path now resolves each compiled `.run`
handle on its own device before launching either peer. Reading the property
does not launch a kernel. Either-rank load failure refuses before any launch;
repeated primed calls avoid the setup work. CPU ordering regressions pass, but
this is not a confirmed cause/repair of the earlier hardware timeout. Full
fresh physical-pair qualification under the new reviewed commit is required.
No measured kernel, peer timeout, greedy/T0 implementation or performance
rule changed. Historical GPU receipts remain historical and unchanged.

## Capture repair and fresh sources

The pair probe previously omitted `stream=` on `torch.cuda.graph`. PyTorch's
class-level default stream can remain on rank0, leaving rank1 capture empty
while its selector executes eagerly and times out waiting for its peer. This
source mechanism matches the coordinator's warning/sticky-error symptoms; the
lane has not reproduced or cleared them on physical hardware.

The probe now allocates an explicit capture stream on each device. The real
K2 worker already allocates and passes rank-local cap_streams. Both sampled
capture paths now declare capture intent independently of current CUDA state;
select refuses before launch if the expected rank's current stream is not
capturing. CPU tests execute the real worker capture method with two distinct
simulated devices and intentionally inject a shared rank0 stream. Solo emulation
is not hardware qualification. Greedy selector, original graph streams/pools,
T0 dispatch, wire q and the entire Triton producer kernel remain unchanged.
Historical GPU receipts retain their original hashes; tests reconstruct exact
historical producer bytes after removing only host capture-intent and launch
handle priming additions.

The staged5fdc17a Sparkhead tree and X86host
`/opt/dsv41/x86host-data/pp4/source` are untouched. Fresh staging is WRITTEN,
NOT EXECUTED in `pp4_admission_stage.sh`. After independent review, its inputs
must name the NEW full reviewed commit and matching source/worker identities.
It derives `~/dsv41-deploy-pp4-admission-<commit12>` and
`/opt/dsv41/x86host-data/pp4/admission-<commit12>/repo`, refuses existing paths,
prepares separate ordinary/admission overlays, and reads the reviewed existing
runtime/tune only. Do not use legacy `pp4_stage.sh` against the queued root.

The coordinator's NEXT physical-pair-only HOLD uses these fresh sources, its
reviewed staging receipt, both GPU locks, shared fencing, W8 and post-close
watch. Physical B1–B4/depth4–5, changed-input graph replay, mixed T0,
peer-timeout/sticky teardown and whole-worker archived identity remain required.
Any failure restores before target experiments. Never synthesize QUALIFIED
from the solo receipts or the failed earlier HOLD.

## Controlled comparison and qualification

Production is still596ba1cf4e3b11be29b41a57e7f69a5d8efa4799,
prod-k1-327-ops1-20261026, legal-v2a/v5, K1 327680, TP2 K2 fb68c52,
REQUIRED0 and async process proxy. PP4 compares rebuilt CPython3.13/Torch2.12.1,
fixed5/adaptation-disabled OFF greedy versus ON top32/tau.7/dyadic24 on the same
target/weights/pair. This controlled comparison cannot establish a win over
adaptive production. Reviewed K1 runner and both None guards remain intact.

The admission HOLD retains pair qualification FIRST, then one ON start, public
cuts, warmup, six indexed experimental greedy checks, and all six primary probes:

- Either-rank q failure becomes target-only through the actual installed verifier,
  with exact same-logit/seed/position coupling on both ranks.
- Grammar, penalties, stop strings and actual sampled EOS on both ranks.
- Tool calls, JSON mode and reasoning parser.
- Cancel/reset/reconnect/timeout/late reply through owned experiment seams.
- Both-rank image/source/ABI reconciliation.
- Complete-round instrumentation: separate512-token companions for every one of
  the THREE admission cells and both public families, both ranks and at least16
  matching complete full-batch rounds. Measurement's B4 omission does not cut this.

The six hash-bound reviewed seed gates remain required. All12 live gates must
pass with their evidence hashes, strict W8/deadline and PASS post-close watch.
The receipt remains NOT QUALIFIED after any failure. The measured-block gate is
explicitly retired, not silently passed. Admission contains no calibration or
headline measurements. Its bounds remain original HOLD+73 abort/+85 restoration.

Round7 persists `admission_run` from the backend tag before the HOLD starts,
overriding any seed identity. Finish validates the physical primary against that
parent run and the expected commit. Measurement requires a nonempty parent run
matching both the evidence index and the primary; older receipts without this
binding refuse. An intact same-commit artifact from another admission cannot
price the new HOLD, even if its path/hash/run bookkeeping is updated.

Admission also records `physical_pair_evidence`: path, hash and run for its own
`physical-pair-qualification.json`. Measurement requires that same-directory
artifact (including its resolved directory), its admission artifact hash, exact
physical schema/coverage/source hashes and source commit, positive finite measured driver seconds, successful
strict restore/deadline and PASS watch. The budget records both primary and
admission-receipt SHA256. Old pair-only pointers, missing/failed/solo receipts
and changed files cannot provide a shorter reserve. These checks validate local
evidence consistency; they do not authenticate deliberately fabricated files.

| Admission component | ESTIMATED seconds |
|---|---:|
|Setup / physical-pair bound|120 /600|
|Historical worker / Spark startup inputs|333 /404|
|Warm / six greedy / cuts|30 /42 /120|
|Semantic/lifecycle probes / profiles / transitions|1800 /320 /60|
|Strict restoration|720|
|Total|4549 =75.82min|

Only historical startups are MEASURED inputs; these are planning allowances,
not observed PP4 duration guarantees. Larger reviewed admission allowances may
not cut any reserve and must still total<=85min. Failed probes STOP and restore.

## Measurement HOLD and reading rule

Exactly OFF_before / ON / OFF_after, three independent worker AND engine starts
in ONE HOLD after physical pair qualification. Closing OFF cannot be skipped.
Each start retains warmup, immutable failure baseline, runtime/source checks,
headline cells and final failure/log checks. ON retains six NBW exact-or-qualified
near-tie greedy checks. W8 still requires six DISTINCT EXACT IDENTICAL indices0–5.

| Cell | Prompt tokens/stream | Batch | Whole groups/start | Requests/start | Outputs/request | Role |
|---|---:|---:|---:|---:|---:|---|
|short B1|1024|1|4|4|512|Decision, always retained|
|32K B1|32000|1|4|4|512|Decision, always retained|
|32K B4|32000|4|2|8|512|No regression only; omitted only by budget rule|

T1/top_p1/high/priority10, ignore_eos, exactly512 outputs/length finish. Matched
42+group seeds/cut hashes, fresh salts per start/group/stream, alternating public
families. B4 streams stay one simultaneous group. Exact output/settings/server
completion gates remain. No private PP2 prompts/captures are used.

**Measurement has NO separate profile companions and NO full-round-p95 guard.**
Complete-round instrumentation is proven in admission. Retain5% TTFT and server
decode ms/speculative request-round guards, both pooled and per public family.
Client text tokens/s remains report-only. Headline failure-counter growth/reset
and final snapshots remain strict; startup failures may remain unchanged.

Pool R=speculative request-rounds, A=accepted tokens, D=server decode milliseconds
per cell/start. Decision rate=(R+A)/D; report A/R and D/R separately. Reference is
the mean of the two OFF rates; spread=abs(OFF_before-OFF_after). A B1 wins only
if its ON-reference point AND paired95% lower delta strictly exceed
max(2%reference,spread). Every measured cell, including B4 if retained, may lose
no more than its own spread at point and lower bound. Exact win equality fails;
exact allowed-loss equality passes. At least one B1 win is required.

Bootstrap2000 paired whole-group draws, seed4103; never split B4 streams or
resample individual speculative rounds. Passing is CANDIDATE FOR REVIEW only.
One ON start has no restart replication. Sparse intervals are conditional;
server counters omit zero-proposal rounds and include terminal effects. No
synchronized engine-step or general performance bound is claimed. Omitted B4
supports no B4 performance/no-regression conclusion. New metric registration is
`pp4-r6-512-aba-no-measurement-profiles-v1`; old results cannot silently qualify.

## Conditional budget and deterministic B4 fallback

`round6-budget.json` preserves Round5's nine numeric-only/hash-pinned NBW-W3 K
artifacts and DEP1 scaling: retain max cold TTFT; scale ONLY max decode span by
511/1499; add2s publication+5s overhead; round UP to5s. Group allowances remain
20/35/105s. B4 remains explicitly ESTIMATED from historical B3 by4/3. No sampled
speed gain, startup overlap or smaller workload cost is assumed.

Measurement pair reserve=ceil(max(180s,1.5*MEASURED admission pair driver seconds)).
The50%/floor is an ESTIMATED allowance for variability and external transitions;
it is not a measured guarantee. Admission/pair-only retain their600s bound.

| Component | All three cells | Two B1 cells |
|---|---:|---:|
|Headline/start|430s|220s|
|Measurement profiles/start|0s|0s|
|Worker + Spark + warm + runtime/fences/start|333+404+30+60=827s|827s|
|First OFF block, including pending120s cuts|1377s|1167s|
|ON block, including42s greedy|1299s|1089s|
|Closing OFF block|1257s|1047s|
|Setup + restore|120+720=840s|840s|
|Total excluding pair reserve|4773s|4143s|

Try all3cells first. If their total exceeds5100s/85min, drop ONLY32K B4 before
HOLD, record the omission and use both B1 cells for every start. If even those
exceed85min, refuse locally before output/host/process creation. No override.
The selected cell set is hash-bound through the budget and saved in the summary;
missing B4 in an all-cell result cannot be reinterpreted as a completed fallback.

Conditional examples, NOT actual admission measurements:

-180s floor reserve: all3cells4953s=82.55min; retain B4.
-600s reserve: all3cells5373s=89.55min fails; two B1cells4743s=79.05min fits.
-957s reserve: two B1cells5100s=85min exactly; larger reserve refuses.

Thus B4 fits only if the actual admission pair driver<=218s; both B1 cells fit
only if driver<=638s under these allowances. There is no successful new-source
admission duration yet, so neither the eventual matrix nor its actual time is
claimed. Startup/probe/prefill/remote variation can still stop a window incomplete.

## Immutable restoration and coordinator-only commands

All modes retain original HOLD+73 abort/+85 restoration, no renewal. Before any
start, all remaining COMPLETE blocks including closing OFF and pending cuts must
fit. All remote mutations, including pair-open, nested starts/stops, probes,
pair-close and W8, use the shared receiving-host scopes. Fence/reap/prove absence
on all hosts before restore and close. Unknown compute/container/JIT state or
failed proof retains HOLD. Final close reserves90s. W8 remains unchanged at
SHA256d1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49.

The shared15min post-close watch remains mandatory after verified restoration,
including stopped runs, outside HOLD. At the85min ceiling ordinary attendance
is100min including watch; exceptional recovery uses its separate existing lease.
Failed restoration/deadline/watch cannot yield QUALIFIED or a review candidate.

Inert lane reproduction:

```bash
source ~/.config/ndm/paths.sh
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.pp4_window --dry-run
bash offload/launch/windows/pp4_admission_stage.sh
```

After independent review ONLY, the attended coordinator stages the new commit,
then runs fresh pair-only, then admission, then measurement if every gate and
budget passes. Use the separate versioned trees, current worker/Markov identities,
full reviewed commit, public corpus/reference inputs and fresh campaign outputs.
The following is a recipe, NOT a lane execution or clearance:

```bash
source ~/.config/ndm/paths.sh
: "${PP4_REVIEWED_COMMIT:?full independently reviewed commit}"
: "${PP4_ADMISSION_SEED:?new-source reviewed seed file}"
WINDOW_COORDINATOR=Deepseek/coordinator \
 bash offload/launch/windows/pp4_admission_stage.sh --execute \
 "$PP4_REVIEWED_COMMIT" "$PP4_ADMISSION_SEED"
```

The runner uses `--execute --pair-only --qualification <staging-receipt>` first;
then `--execute --admission --qualification <seed>`; measurement uses
`--execute --qualification <successful-admission/admission-receipt.json>`.
All require matching `--expected-experiment-commit`, `--worker-tree/commit/identity/
markov-identity`, fresh `--out` and separate `--remote-repo`. No command here
changes production sources, images, units, queued paths or promotion state.

## Round 10 — service graph dispatch (CPU-reviewed; hardware pending)

The sampled-only `graphs=False` override in the actual service builder is
removed. Default sampled/OFF callers capture; explicit `--no-graphs` still
selects eager dispatch. TP2 readiness now logs the resolved graph mode.
`server.py` joins the physical probe source manifest: historical receipts
remain historical and cannot attest to the changed entry point.

This corrects the demonstrated qualification/service mismatch. It does not
establish the cause or cure of the observed slow wider-batch step. Next use
the separately reviewed X86host-only diagnostic before another Spark admission.
No timeout, kernel, launch order, numerical rule, admission budget or recovery
change is part of this correction. The existing S78/S76 greedy-request salt
blocker remains open for admission/measurement; this diagnostic clears none
of those gates.

The Round10 X86host-only driver is documented in
`offload/dspark/probabilistic/pp4/SERVER-PROBE.md` (inputs, exact coordinator-only
recipe, primary evidence, immutable bounds and failure/restore interpretation).
It is WRITTEN, NOT RUN. It executes real `worker.server --tp 2` in separate
owned reference/baseline/candidate processes, actual direct frames and native
X86host-local OFL callbacks. It uses prod_window pair/close only; no Spark SSH,
API, HOLD, staging or image operation. Startup600s/request30s bounds are
unchanged existing limits; no new server graph startup duration is measured.
ESTIMATED allowance <=72min includes the fixed60min probe/12min close ceiling;
slow work truncates rather than extending those limits. Historical physical
qualification remains historical; the driver binds both server.py hashes.

## Round 11 — S80 initial request-sequence correction

The diagnostic now starts each direct/OFL session with HELLO and a bounded
window-only STEP: four full128-row generation1 slots, no draft flags, zero
masks. Its first two draft calls are positive B1/two rows, then positive
B4/28 rows, with no intervening RESET, resync, width or draft. Slot0 advances
128->130->137, other slots128->135; every consumed window remains full.
Only then does the existing repeated width/mask/reset matrix run. Both fresh
sessions in both transports follow this order. OFL still follows direct/model
work and is not cold-model evidence. The model helper likewise executes its
first positive B1/B4 pair before repetition, separately at depths4/5.

Actual DraftCore/codec CPU regressions decode every submitted frame, record
backend draft widths/inserted-row counts, and assert the first pair, same
session/generation/contiguous windows and absence of intervening controls in
direct and fake-OFL paths. Model API ordering is checked separately. S80's
prior graph/lifecycle review remains a CPU/source finding; this sequence fix
adds no hardware qualification. Driver ownership, numerical checks,30s frame
bounds and close behavior are unchanged. Written, NOT RUN; independent review
and coordinator-only X86host execution remain next.
