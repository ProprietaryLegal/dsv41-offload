# PP4 promotion gate × R-PP4 v2 measurement — patch note for `offload/pp4-promote` (R-PP4, 2026-10-27)

Written by R-PP4 on `offload/pp4-win2`; `offload/pp4-promote` is not edited here (S91 checks it, PP4-PROMOTE round 2 owns
it). Read against `offload/pp4-promote` c258296 (`pp4_promote_window.py`).

## Already consistent: LOSS semantics

The promotion's two no-loss checks already use `decision_v2` LOSS semantics, as the coordinator asked:

- **G6, in-window** (`regression`, 67af145): same-HOLD exact production control vs PP4, 3 unpaired repeats per arm,
  `decode_tps` primary + `ttft_s` guard, `decide_window`; any decision-cell `LOSS` (worse than the control's own spread in
  ≥ ceil(2n/3) repeats) or `NOT A CANDIDATE` refuses; WIN and INCONCLUSIVE pass. No change needed.
- **Pre-promotion evidence** (`measurement_gate`, fbbe129): recomputes `decide_window` from the measurement's recorded
  `decision_series` and refuses any decision-cell `LOSS`. R-PP4's summary is built for this: `decision` carries
  `decide_window`'s own top-level fields (`rule`, `status`, `min_gain` 0.02, `decision_cells`, `cells`, `groups`) and
  `decision_series` is top-level, so the recompute matches exactly (tested:
  `test_decision_recomputes_from_its_recorded_series`). A clock-guard override (starts at different GPU clock caps, or
  a thermal dial-back inside the HOLD) sets `status` INCONCLUSIVE, which deliberately does **not** recompute, so the gate
  refuses that measurement as evidence.

## Four changes `measurement_gate` needs to accept an R-PP4 measurement

### 1. Metric set: rate primary; ms/round reported, not a guard

`measurement_gate` today: `ORIENTATION` has no `pooled_rate_tps`, and `MEASUREMENT_METRICS = ('accepted_per_round',
'ms_per_round', 'ttft_s')` must all be in `decision_series`, where every series after the first is a LOSS-capable guard.

R-PP4 writes, per decision cell (`short-b1`, `32k-b1`), `decision_series` = `pooled_rate_tps` (primary,
(rounds + accepted) / decode seconds per repeat), then `accepted_per_round` and `decode_tps` (guards, per WIN2.md), and
`reported_series` = `ms_per_round`, `ttft_s` (each with its own win2-v1 verdict in `decision.reported_metrics`, never
deciding).

Why ms/round cannot be a guard here: PP4 deliberately spends ms per round (+0.31 ms MEASURED for the fused selector) to buy
accepted tokens (+6.9 % MEASURED offline). In the fixed5-vs-fixed5 controlled mode, ms/round barely moves with content, so
the control spread is well under 1 % and +0.8 % would read as a LOSS in every repeat even when tokens per second rise.
In the production-policy mode, adaptive depths 2..5 and fixed 5 draft rounds of different shapes. The promotion file
already says this for G6 ("per-round metrics are not like for like"). The rate charges the ms/round cost in full: it is
the rate's denominator. `decision.checklist_literal` records what ms/round-primary would have said, for review only.

Patch (`pp4_promote_window.py`):

```python
ORIENTATION = {'pooled_rate_tps': True, 'accepted_per_round': True, 'ms_per_round': False, 'ttft_s': False,
               'decode_tps': True}
MEASUREMENT_METRICS = ('pooled_rate_tps', 'accepted_per_round', 'decode_tps')   # decision_series, primary first
REPORTED_METRICS = ('ms_per_round', 'ttft_s')                                    # reported_series: present, non-deciding
...
    for cell in MEASUREMENT_DECISION_CELLS:
        got = [s.metric for s in series if s.cell == cell]
        require(got == list(MEASUREMENT_METRICS), f'PP4 measurement {cell}: decision series {got} differ (primary first)')
    reported = measurement_series(data.get('reported_series'))
    for cell in MEASUREMENT_DECISION_CELLS:
        missing = sorted(set(REPORTED_METRICS) - {s.metric for s in reported if s.cell == cell})
        require(not missing, f'PP4 measurement {cell}: reported series {missing} absent')
```

### 2. Source binding: carried qualification

`measurement_gate` today: `experiment_commit == switch.PP4_COMMIT` (941a3c3 at c258296; its admission stopped on the
near-tie locator defect, so the admitted commit will be PP4 Round15 85b0d96 or later, and `switch.PP4_COMMIT` moves with
it). An R-PP4 measurement runs at R-PP4's own commit (the v2 runner exists only there; it merges 85b0d96) and carries
the admitted commit's QUALIFIED receipt only when every file outside the measurement layer is byte-identical (that
includes `pp4_cells.py`, so the near-tie code is the admitted one) and the runner's remote/runtime units are
AST-identical (`pp4_measure_v2.carried`). Its summary records `experiment_commit` = R-PP4 HEAD, `runtime_commit` = the
admitted commit,
`carried_qualification` = {qualified_commit, head, changed_files, rule} and `qualification_sha256`.

Patch: merge `offload/pp4-win2` (brings `pp4_measure_v2`) and replace the commit check with

```python
from offload.launch.windows import pp4_measure_v2 as mv2
...
    carried = record(data.get('carried_qualification') or {})
    require(data.get('runtime_commit') == switch.PP4_COMMIT and carried.get('qualified_commit') == switch.PP4_COMMIT
            and carried.get('head') == data.get('experiment_commit'), 'PP4 measurement does not carry the admitted source')
    # Re-prove the carried rule now (no trust in the recorded file list): same git diff allowlist + runner unit ASTs.
    mv2.carried(admission_path, switch.PP4_COMMIT, str(data['experiment_commit']), str(data['measure_v2']))
```

(`measurement_gate` needs the admission receipt path for that call; `evidence_gate` already has it.)

### 3. Which measurement: production-policy mode

R-PP4 has two registered modes. `controlled` (s1-A/s2-B/s3-A) is the old fixed5 greedy vs fixed5 sampled comparison;
`production-policy` (s1-P/s2-B/s3-P) compares the PP4 promotion configuration (fixed5 sampled, `PP4-sampled.env` role B)
with production's own adaptive policy (`PP4-production-policy.env`) on the same PP4 runtime. Only the second says
anything about "no loss against production". Patch:

```python
    require(data.get('measure_v2') == 'production-policy', 'promotion evidence must be the production-policy measurement')
    require(data.get('config_sha256') == pf.digest(SAMPLED_CONFIG), ...)   # unchanged: B is PP4-sampled.env
```

### 4. Recorded fields R-PP4 adds (no change needed; listed for the reviewer)

`qualification_sha256` (sha of the `--qualification` receipt), `runtime_commit`, `carried_qualification`, `rng_gate`
(the #61161 verifier-RNG gate receipt validated before host contact; its PP4 kernel sha equals the admission's
prepared verifier), `measure_v2`, `production_policy_config_sha256`, `decision_series`, `reported_series`, `archive`.

## HOLD budget implication

None for the promotion HOLD: the R-PP4 measurements are separate windows before it (ESTIMATED 80.1 min controlled,
80.8 min production-policy with a 333 s pair reserve; measurement ends at 68.1 / 68.8 min). The production-policy
window refuses before host contact if the admission's measured pair reserve exceeds 588 s (600 s dry bound: 85.25 min).
G6 inside the promotion HOLD is unchanged.

## Caveat the reviewer should keep

P is production POLICY on the PP4 runtime (PP4 K2 worker in greedy mode + PP4 verifier overlay), not the production
binary. The exact-binary comparison is G6 inside the promotion HOLD. A production-policy measurement with no LOSS is
necessary evidence, not sufficient; G6 remains the final no-loss gate.
