# Second matched pilot: draft Amendment 8 for coordinator review

D2-PILOT2 prepares recipe options only. This is proposed text, **not an amendment
adopted into the binding plan**, an approval to run, or a diagnostic conclusion.
The coordinator records NDM's authorization and a confirmed DIAG.md reading
before choosing a recipe, staging its immutable snapshot or seeing second-pilot
data. The diagnosis is the one on `offload/d2-diag`,
`offload/dspark/d2/train/DIAG.md`, including its clarifications and binding menu.
No pilot or diagnostic was run by this lane.

## Proposed Amendment 8

Use one matched REF / FAST / FAST-B pilot with exactly one change from the first
pilot. Freeze **one** recipe in advance. Do not run a menu and choose the best
result afterward. Proposed deterministic selection order:

| Confirmed diagnosis | Recipe to preregister | Single change |
|---|---|---|
| H6 calibration-only gain, including H1 + H6 | `ace` | Hard CE on the teacher top-20 mode; ties choose the lowest token id |
| H1 mass covering, without confirmed H6 | `tvd` | Existing top-20-plus-tail total variation objective, per OBJ1 §2 |
| H5 early-update shock, with neither H1 nor H6 confirmed | `lr` | KL with lr 5e-5 and linear warm-up for the first 10% of updates, then cosine |
| H0 only, no confirmed matching cause, or H2/H3/H4/H7 alone | No second pilot from this menu | Stop and report; these findings require a different approved experiment |

If H1/H6 and H5 are both confirmed, the objective recipe above takes priority;
this pilot does **not** combine an objective change with an LR change. The
coordinator may instead preregister `lr` with a written diagnostic rationale,
before data. Such a decision must be recorded as a change to this proposed
selection table, not selected from second-pilot outcomes. `kl` is the default
compatibility/control recipe, not a proposed useful rerun after H0.
Component freezing, teacher rescaling, position reweighting, reverse KL and
new training families are outside these implemented options.

All other settings stay matched: native 1017 zero-adapter initialization, seed
23, FAST-B order seed 24, raw taps, r32, batch ceiling 128, chunk 8192, AdamW
weight decay .01, clipping 1, 20,000 unique blocks, 159 updates, checkpoints
2,500/5,000/10,000/20,000, and unchanged QDQ/STE arithmetic. TVD/ACE retain lr
2e-4 and the original cosine. The `lr` recipe uses ceil(.1 * 159) = 16 warm-up
updates: applied lr for zero-based update s < 16 is 5e-5*(s+1)/16; thereafter
5e-5*.5*(1+cos(pi*(s-16)/(159-16))). Scheduler steps follow optimizer updates;
it reaches zero after update 159. All three arms use the same applied rates,
which the plan and evaluator check exactly.

ACE uses **only** the teacher's highest-probability retained token, never the
committed sampled label or a CE/KL mixture. Every valid D2 scored position has
a top-20 row (DIAG Audit (a)); absent/empty top-20 rows are skipped for ACE,
including in its weight denominator. The generic KL/TVD missing-target .1 CE
fallback remains unchanged. Position weights are (1,.8,.65,.5,.4), normalized
across active positions. An entirely unscored batch refuses. FAST uses the
same objective with explicit q-onehot gradients; REF remains streamed autograd.

The **sample rule and pass rule are unchanged**. Redraw from the same MANIFEST
and exact K2 files using seed 20261024, at most 1,000 complete fixed-K2 rounds
per development split, before training. The complete sample object (keys,
ordered key hash, source hashes, population counts/hashes, seed and limit) must
equal the first pilot's sample. `pilot sample --previous-plan` asserts this;
the stager repeats it before train/evaluate/finish. The first plan is read-only
and pinned to SHA256
`c4732bc224944af7df7de1d1fe67dbf6efd2d6bcd9d49bdc94af584085074402`.
The baseline is recomputed on this identical sample, not copied: first pilot
MEASURED E[A] dev 1.505 and heldout_legal 1.855 are reproduction checks, not
substitute evidence. Keys stay private.

`pilot_verdict.py` is unchanged: at 10K and 20K on dev and dev_legal, REF and
FAST each gain >=.05 E[A] over zero-adapter with the 90% family-clustered lower
bound >0; FAST-minus-REF lower >=-.03; FAST baseline deficit <=.01; each
position's FAST median KL deficit to REF <=.01 nats. The exact Amendment
6/6a/6b FAIL/INCONCLUSIVE/extra-seed rules remain authoritative. E[A] is the
fixed-K2 0/1 prefix metric excluding bonus/correction; EA_RB is informational.
Mean KL need not improve for ACE/TVD. Evaluation still reports reference KL
and the historical KL loss, so the same metrics and verdict remain comparable.
Both judged splits remain development data; no reserved TEST family is used.
A PASS here does not pass full G2/G3, admit the current full runner, or deploy.
See [PILOT2-ADMISSION.md](PILOT2-ADMISSION.md).

The plan, each arm receipt/checkpoint contract and STAGED.json bind the recipe.
The run tag and recipe derive new directories such as
`decode-d2pass2-r1-p2-tvd` and sessions
`d2pilot2-tvd-decode-d2pass2-r1-*`. Never alter the first pilot's namespace,
source, receipts or sessions. The stager refuses overlap, existing destinations,
changed source/inputs/commands, and changed recipe/run tag after staging.

## Coordinator commands (local x86host only; not executed by this lane)

First record NDM authorization, diagnostic receipt hash/readings and the chosen
recipe in the campaign preregistration. Wait for the diagnostic and other jobs
to release **both** solo leases; do not stage/run into the D2-DIAG snapshot.
Keep the existing power logger alive, confirm no swap and sufficient RAM for
the existing two-job limits. Leases remain REF GPU-240876ad and FAST/FAST-B
GPU-bc62df2a, 225 W, individual nonblocking flock, <=128 GiB RSS, no production
pair. This procedure neither changes services nor sends target requests.

Review and transfer the integration commit using the existing coordinator
process. Run from its clean checkout, with HEAD equal to the reviewed full SHA.
The placeholders deliberately refuse until replaced. These five `--execute`
commands are **coordinator-only**, not lane rehearsal instructions.

```bash
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
export PYTHONPATH="$PWD:$PWD/offload/dspark" CUDA_VISIBLE_DEVICES=''
export D2_COORDINATOR='Deepseek/coordinator'
P2_COMMIT=REVIEWED_FULL_40_HEX_COMMIT
P2_RECIPE=CHOSEN_RECIPE_FROM_PREREGISTRATION
P2_TAG=decode-d2pass2-r1
P2_PY=$HOME/dsv41-offload/e/venv/bin/python
[[ "$P2_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'replace reviewed commit'; exit 1; }
case "$P2_RECIPE" in tvd|ace|lr) ;; *) echo 'record and set chosen recipe'; exit 1 ;; esac
[ "$(git rev-parse HEAD)" = "$P2_COMMIT" ]
[ -z "$(git status --porcelain)" ]
pilot2_args=(
  --source "$PWD" --commit "$P2_COMMIT" --recipe "$P2_RECIPE" --run-tag "$P2_TAG"
  --assembly /opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2
  --parity /opt/dsv41/x86host-data/d2/train/parity50-solo225/parity.json
  --g1 /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.json
  --k2-dev /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.k2_drafts.jsonl
  --k2-legal /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/heldout_legal.k2_drafts.jsonl
  --previous-plan /opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1/plan.json)
# Dry render, then CPU input admission check (no launch):
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}"
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --check-inputs
# Stage creates NEW snapshot/output only; prepare starts the leased prepare job,
# then draws and checks the identical sample in that job before it exits.
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --execute --phase stage
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --execute --phase prepare
```

Wait for `d2pilot2-$P2_RECIPE-$P2_TAG-prepare` to disappear and inspect
`$HOME/dsv41-offload/d2/pilot/$P2_TAG-p2-$P2_RECIPE/logs/prepare.coordinator.log`.
Require successful prepare/sample, complete sample equality and no running
prepare lease. Preserve the sampled plan thereafter. Then:

```bash
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --execute --phase train
```

Wait for both `...-ref` and `...-fast` sessions to disappear; inspect their
coordinator logs and all three complete arm receipts. FAST-B runs after FAST
on the same solo, while REF runs independently. Follow PILOT-STAGE.md's
boundary-resume procedure for an interruption, using this new snapshot and
new session names; never rerun prepare/sample or a completed arm. Then:

```bash
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --execute --phase evaluate
```

Wait for both `...-eval-dev` and `...-eval-legal` sessions to disappear with
successful logs and both complete part files. Baseline E[A] must reproduce the
first pilot, and every part uses its identical frozen sample. Then:

```bash
"$P2_PY" -m offload.dspark.d2.train.pilot_stage "${pilot2_args[@]}" --execute --phase finish
```

Wait for `...-finish` to disappear and inspect `finish.coordinator.log` plus
`/opt/dsv41/x86host-data/d2/pilot/$P2_TAG-p2-$P2_RECIPE/receipt.json`.
Report the exact outcome; no automated extra seed, full launch or changed gate.
Do not infer completion from an echoed exit marker alone.

## Wall and allocation estimate

MEASURED source: binding D2 plan, Result 17:16Z 2026-10-25: first pilot REF
6,999 s, FAST 206 s (FAST-B 201 s), 159 updates each; evaluation 104.9 min per
card, 13 passes x 1,000 rounds. No second-recipe timing is measured.

ESTIMATED using 206 s conservatively for **each** FAST arm: training wall
max(6,999, 2*206) = 6,999 s = 116.65 min; then parallel two-card evaluation
104.9 min; total **221.55 min = 3.6925 h wall**, plus staging/hash checks,
loads, checkpoint I/O and scheduling overhead. Training allocation
(6,999+2*206)/3,600 = 2.0586 GPU-h; evaluation 2*104.9/60 = 3.4967 GPU-h;
combined approximately 5.5553 GPU-h. ACE/TVD may change the rates: these are
planning estimates from the KL pilot, not measured second-pilot performance.
Full training, full-set G2, new TEST and worker qualification are separate.
