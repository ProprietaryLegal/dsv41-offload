# D2 replay preparation (D2-REPLAY-PREP)

Coordinator execution only. The preparing lane sends no Spark requests or service commands.

## Input staging — MEASURED 2026-10-24 UTC

`stage_replay_inputs.sh` copied the completed generation state (excluding LOCK), acap exports,
prompt manifest, tokenizer and pinned encoder into private directories under x86host
`/opt/dsv41/x86host-data/d2`. All 135 staged files pass INPUTS.SHA256 there. Original and staged
sequence/acap/tokenizer/manifest bytes were also compared on Controller. Directories are mode 700.
No prompts.jsonl is required by replay or copied to x86host.

V1 is external to this repository. Its existing x86host `~/dsv41-offload/d2-data/v1/verify`
reference was copied to private `d2/v1/verify` and every Python file hash compared with the
original. V1.SHA256 itself is pinned by the runner to
17109d736c217ea9c459be1f7851a2b7cfe057b762ae6b60ee86a7dc7f063684.
MEASURED CPU loading of the existing x86host captures: four nonwarmup requests, 542 every1
equivalence blocks and zero capture plumbing errors. No target inference was involved.

Actual starting branch HEAD was 90310af (brief commit after f2f4ee6). Its immutable code snapshot is
`source/90310af87fc143a72bb2177fc1cbaafbf31eb866/repo/offload/dspark` under that private root.
Archive SHA256: f77ac228eeeb5ef57477a763e2321ce3319afeba982c43ce8c0b3f3390aa8ee9.

CPU `d2.replay plan` and `estimate` ran on x86host using the existing o2 Python, CUDA hidden.
Plan: **128 entries, 0 skipped, 55,700 bytes**, SHA256
cd95f20702607d9d62349b0d9e7a57e7de4e76c5d5139dd584b9e390cadb4e53.
All 121 done + 3 partial generations are supported by replay.load_seq and included. Four acap exports
are included. Generation has 958,822 saved tokens; combined span positions are 960,223.
There are **1,957,785 prefill tokens and 1,000,218 tap rows** (30,726,696,960 tap bytes).

## ESTIMATED window cost

`plan-summary.json` includes acap; `estimate-generation.json` excludes it.
Combined replay: 14.8 minutes at pure-prefill 2,200 tok/s; 30.8 minutes at the prior Phase-0
end-to-end 1,060 tok/s rate. Neither is a measurement of this run; inflight remains **2**.
Top-20 arrays ~165.2 MB; serialized prompt-logprob responses ~2.8 GB in total (streamed per request).
Copy arithmetic at the configured 1 GB/s cap is 0.51 minutes; the historical runbook allows
~9 minutes for the guarded copy. Use that conservative operational allowance, not the cap as an
observed throughput. Start ~8 minutes + replay ~15–31 + copy allowance ~9 = **32–48 minutes**
before equivalence. All-block three-arm equivalence is ~90 minutes CPU (historical estimate,
not measured on x86host here). Coordinator change 2026-10-24: equivalence runs as a separate x86host-only
phase AFTER production is restored, so the production outage is the **32–48 minute** window phase plus
~8 minutes of restoration; a failed gate needs the decode-capture fallback window either way. No runtime
guarantee is implied.

Private stage log: campaign `D2-REPLAY-PREP.stage.log`. No GPU, dependencies or services changed.

## Provenance preparation

`replay_evidence.py` writes a private hash-indexed evidence inventory and an explicitly incomplete
`d2.sequence-context.v1` scaffold. Neither constitutes coordinator approval. The scaffold contains
the verified tokenizer marker map and renderer hashes, known request settings and separate acap
seed-42 contexts. Target/engine receipts remain null; effective defaults are marked unverified.
Do not rename the scaffold to make the builder accept it.

MEASURED read-only rank0 file hashes match x86host's S2 candidate manifest and quantization plan:
candidate 7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae;
EXL3 plan ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d.
Tokenizer c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b and
encoder abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b match the
frozen prompt manifest. The current rank0 generation_config.json is absent. Small model files
were read with O_DIRECT; container inspection is read-only. No endpoint was queried.

The 127 recorded generation segments have no sampling/revision fields: the documented legacy
e2b2186 contract applies; no priority20 restart is inferred. ACAP-SAMPLING.md and its primary four
benchmark rows establish T=1/top_p=1, high thinking, seed42, priority10. The acap round-temperature
evidence remains the existing D2-FIX1-R2 receipt. OPS1 records image/engine/launch/mount state, but
explicitly disclaims runtime-byte/health attestation. None proves old effective processors by itself.

The coordinator must verify the historical target/quantization/runtime receipts for generation
and original acap, effective SamplingParams/generation-config defaults, absent tools/schema/custom
stops/processors and actual template selection. Null request options are not proof of effective
defaults. Identify any unrecorded generation segment gaps honestly. In the window, run
`capture_window_identity.sh` on BOTH ranks (the runner does this), save image id, full mount list,
command/env, mounted replay overlay hash, served tokenizer/config/encoder hashes, SamplingParams
and scheduler hashes, generation-config presence/hash and engine version. Check them against the
historical evidence; changes in target engine overlays need review rather than alias-based approval.
Then supply a fully verified final context with separate original acap overrides and SHA256-bound
generation/replay evidence receipts. The builder must still validate it before any tensor assembly.

Fresh window evidence cannot reconstruct missing historical attestation. If the coordinator cannot
substantiate a required field, real build stays refused and collection must be reconsidered.

## Coordinator window runner

Campaign entry point: `W9/replay_window.sh` (copy of the tracked shell wrapper). Default or
`--dry-run` exercises synthetic checks locally, advances a virtual five-minute clock, creates no
services or results and cannot call SSH/HTTP. It prints SYNTHETIC/DRY-RUN; it is not a live success.
Failure probes also run parsers on missing/busy/nonfinite gauges, memory below 3 GiB, wrong images,
bad runtime hashes, cached metadata, incomplete copies and truncated equivalence shards. A fake-SSH
supervision test proves low memory closes the remote command lease and leaves unrelated work alive.

Before live execution, stage a fresh immutable `offload/dspark` snapshot of the reviewed HEAD in
x86host `d2/source/FULL_SHA/repo`, with `SOURCE_COMMIT` beside it. Use the item-1 staging procedure
for a fresh snapshot (do not replace an existing one). Set D2_REPO to that same reviewed local
checkout. Local and remote code hashes and the frozen input/plan hashes are checked before opening.
Only the coordinator runs, in detached tmux:

```bash
source ~/.config/ndm/paths.sh
export D2_COORDINATOR=Deepseek/coordinator
export D2_REPO=/absolute/reviewed/checkout
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W9/replay_window.sh --execute --phase window \
  --source FULL_REVIEWED_40_CHARACTER_SHA \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/W9/run-NEW_ID
# then W8/restore.sh start, check, close (production back), then, x86host CPU only:
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W9/replay_window.sh --execute --phase equiv \
  --source FULL_REVIEWED_40_CHARACTER_SHA \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/W9/equiv-NEW_ID
```

The runner waits up to one hour for 300 seconds of zero running AND waiting production gauges;
busy resets the clock; missing/failed telemetry refuses. It then opens the window, starts A-replay,
checks rank0 overlay-active and absence of root-owned/refusal/error lines, captures both-rank identity,
runs with inflight2/check-remote, validates response metadata, copies/checks all planned files,
checks both container IDs and stops the experiment (phase `window`). Phase `equiv` (x86host only, no Spark
memory stop, production already restored) re-checks the copied population and performs three fresh
equivalence shards at every1, including completion/population checks.
The new checks supplement existing tools: replay check alone does not fail missing sequences, and
equiv report alone cannot distinguish a truncated-but-present shard. Gates use the real nested
`gate.pass`/`gate.verdict` schema. No override flags are provided.

Both Sparks' MemAvailable is checked before and throughout active steps; below 3 GiB, absent/stale
telemetry, external STOP, command failure or timeout stops the workflow. Remote active commands own
an EOF lease; loss of the local supervisor stops only their child process groups. A service started
by a launcher remains coordinator-owned on abort; the runner never claims an SSH disconnect restored
production. It does not call x86host's worker/unit/window commands or allocate either production GPU.
After a complete copy it checks BOTH captured container IDs before stopping the experiment. On any
failure it stops its submissions and leaves service inspection/restoration to the coordinator.
HOLD stays set. Run W8/restore.sh `start`, `check`, then `close` separately; do not delete HOLD first.

## Post-window script

Campaign `W9/after_replay.sh` defaults to offline dry-run. The coordinator uses the same D2_REPO,
D2_COORDINATOR and full source SHA as the replay window; `--execute --source FULL_SHA --out NEW_DIR`
executes only x86host commands. It never contacts a Spark. `--context` defaults to x86host
`d2/provenance/sequence-context.json`. The prepared INCOMPLETE scaffold is deliberately a different
filename. Final context must list accessible hashed `evidence_receipts` ({path, sha256}) for every
target/replay receipt digest, including original acap. Context validation runs before building.

Order: full dataset build with context + passed equivalence; dataset verify; exact plan/context/
manifest/gate binding with no skips/overrides; G1 plumbing; S25 aggregate; K2/score for acap, legal-dev,
general-dev and dev; binding acap PASS; fresh immutable pilot source; GPU-backed pilot prepare;
CPU once-only sample and plan validation. S25 writes only aggregate counts of resid==0, minimum
top_lp, max |sum(exp(top_lp))+resid-1| and counts over 1e-6/1e-4, with manifest identity and explicit
synthetic status. No actual S25 data measurement is claimed before replay/build.

G1 uses only solo GPU-bc62df2a, nonblocking its existing lock, checks the existing power logger and
225 W cap, and limits each hold to 840 seconds (780-second internal deadline). Its compiler/tuning
caches stay in D2. A deadline can return zero with partial output, so a separate sequence/block
coverage check refuses scoring until complete. If interrupted, `--phase g1` resumes K2's saved
sequences without rebuilding. `--phase build` stops after the CPU build/audits. `--phase pilot`
skips G1 generation but revalidates the exact shards/gate, binding acap PASS and both pilot K2
populations. Resume phases never rebuild shards or silently redraw an existing sample.

The fresh pilot root is x86host `d2/pilot/w9-FULL_SHA` (under this lane's write boundary). Its copied
source files are read-only, directories are private mode700, logs/build caches are siblings, and SOURCE_COMMIT pins the reviewed source.
RULE_VERSION/EVALUATOR must be amendment-6ab v2 before preparation. The existing 50-window solo225
parity receipt is required and validated by pilot prepare against its loaded frozen identity.
Pilot preparation uses the same permitted solo wrapper/lease; no arm training starts. Prior pilot
sources/plans are never reused. If prepare/sample is interrupted AFTER the new source root exists,
inspect that private root and resume the corresponding printed exact command: preserve any plan,
do not recreate the root or redraw a sample. A fresh --execute output directory preserves new logs.

These are future coordinator GPU commands, not GPU work performed by this preparation lane. No
production-pair UUID/unit/window command is invoked by either W9 script. Review and integration
remain with Deepseek/coordinator; the lane performs no merge or push.

## CPU verification record

MEASURED on Controller: full d2/tests + d2/train/tests **481 passed**, 232 pre-existing
PyTorch output-resize deprecation warnings from native/formats.py:44. Final focused runner,
post-window, replay and equivalence-control suite **76 passed**. Strict mypy is clean on all
six new Python modules and both new test modules (8 files); all four shell entry/probe/staging
scripts pass bash syntax. No new dependency was installed. X86host's existing o2 environment has
tokenizers 0.23.3, torch 2.10.0+cu126, safetensors 0.8.0 and numpy 2.2.6; CPU imports were checked
with CUDA hidden. Shellcheck is not installed; syntax and offline behavior were checked instead.

Logs are campaign `D2-REPLAY-PREP.{final-full,final-focused,mypy,dry-run,after-dry-run}.log`.
Dry runs are synthetic workflow checks, not evidence that an inference window succeeded. No live
replay, dataset build, G1 GPU pass, real S25 aggregate, pilot prepare or model training ran here.

Final response guard: replay.parse_logprobs now rejects explicit HTTP-200 refusal/error/filter/abort
responses and cache hits BEFORE publishing any arrays. The existing cached-response RuntimeError
contract is preserved; its regression test now also proves no rejected NPZ is left for resume.
