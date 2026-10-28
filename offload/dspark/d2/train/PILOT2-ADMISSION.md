# Second-pilot PASS: full-run admission work still required

This lane changes no `full_*.py` file. A second-pilot PASS would qualify its
matched recipe outcome, but the current `full_stage` cannot admit that new
source lineage. Do not edit a plan/source map, copy a PASS onto an old plan,
change a pin to whatever happens to be on disk, or invoke a verifier without
its original immutable source to work around this refusal.

## Exact current blockers

`full_stage.admission` calls `require_full_pilot` with `--pilot-source`.
`full_admission.verify_pilot_source` currently requires:

- plan source `d2/train/loop.py` = `PILOT_LOOP`,
  `273477509c033da9e6a41d7d4629634816f859bac6d9b7742b7665c54861fed1`;
- current full-source loop = `FULL_LOOP`,
  `74c6c16c1d5376c0ac9ab57d40468f92e8c21ef1d5b56b8bc18cd7c54fc9dab1`;
- immutable pilot file set and every recorded source hash exactly match its
  plan, and every current source other than the loop matches that pilot map.

MEASURED candidate loop SHA256 in this lane:
`6c5cad94195581efc25452df28d8884f0511a04df7e7a65b5390cb1a61ce9858`.
It adds recipe support and the opt-in LR schedule; it deliberately matches
neither historical pin. The second pilot also changes `loss.py`,
`fast_loss.py`, `pilot.py`, `pilot_eval.py`, `pilot_stage.py` and
`pilot_units.py`. These changes must be qualified together in the new pilot;
a first-pilot source map is not evidence for them. The actual new plan's
source map, not this descriptive list, is authoritative.

Passing a new plan to the old immutable-source bridge fails with
`unsupported pilot source lineage`. Passing an old plan to the candidate
full loop fails with `full loop changed beyond reviewed admission patch`.
Both are intended refusals, even if a receipt says PASS. The first pilot's
actual FAIL remains failed.

There is a second blocker after source qualification: `full_stage.commands`
and `full_train`/`loop.main` still construct the old KL run (lr 2e-4, no
warm-up) and expose no recipe selector. Merely relaxing the source pins could
silently train a different objective/schedule from a TVD/ACE/LR PASS. The
current recipe-aware `fast_policy.require_pilot` can verify such a receipt's
pilot evidence; that alone does not prove that a full Run uses its recipe.
The supported full_stage path requires the reviewed original-source bridge.

## Required coordinated change, after an actual PASS

1. Review the exact second-pilot integration commit and immutable snapshot,
   chosen recipe, source map and PASS receipt. Add an explicit supported
   lineage in `full_admission.py`: pinned qualified pilot loop/source identities
   and the narrowly reviewed full-candidate differences. Preserve the
   historical lineage if still supported. Verify every original source file
   and the exact file set, then run **that pilot's unchanged verifier** in its
   own original namespace. The new map includes the full-run modules that
   already existed when the second pilot was prepared. Do not blanket-ignore
   `full_*.py`, arbitrary additions or changed loss/scheduler modules.
2. Bind recipe into full staging, commands, checkpoint/resume and full-evaluation
   receipts. Select the qualified Run from the verified PASS, or expose a
   recipe argument that must equal it; reject a mismatch before loading or
   writing. Extend `full_stage`, `full_train`/`loop.main` and any necessary
   admission/evaluation contracts together. For TVD/ACE propagate divergence;
   for LR propagate lr 5e-5 and the 10% warm-up-plus-cosine policy. Keep other
   qualified settings fixed. Define the full-run schedule with its full update
   count explicitly; do not reuse the pilot's literal 159-step LR list.
3. Stage a NEW full namespace including recipe/run tag and a NEW reviewed source
   snapshot. Bind its source map, pilot snapshot, receipt and recipe without
   mutating the pilot snapshot, STAGED.json, sample, checkpoints or receipts.
   Start fresh from the native base and seed 23; do not resume a pilot adapter.
4. Preserve every real-data/G1/parity/frozen-identity/shape/sample/checkpoint/
   source admission and the unchanged pilot verdict calculation. Full G2 uses
   all complete rounds of dev, heldout_legal and heldout_general, not the pilot
   sample. The reserved fresh TEST and G3/worker/deployment gates remain
   separate. A full-run bridge is not permission for those later operations.

## Required tests for the admission follow-up

- Both explicit reviewed lineages are admitted only with their exact original
  source trees; unknown loop hashes, changed/additional/missing source files,
  changed original sources and unreviewed full-source differences refuse.
- Genuine nonsynthetic PASS evidence is reverified in the second pilot's own
  source namespace, including sample redraw, matched units, every arm recipe,
  every checkpoint hash, interval/threshold computation, evaluation identity,
  population report and input hashes. FAIL, INCONCLUSIVE, synthetic, old-rule,
  forged/mutated receipts and a borrowed recipe still refuse before model load.
- Full dry rendering remains side-effect-free and preserves the old CLI's
  shared Settings contract. Every supported recipe is propagated exactly;
  missing/mismatched recipe, changed resume contract and wrong namespaces refuse.
- Tiny CPU full-versus-qualified-pilot arithmetic tests for every recipe,
  permitting only the declared epoch/update-count schedule differences. Audit
  actual applied rates, fresh initialization, optimizer state, cancellation,
  archived checkpoint immutability and exact interruption/resume. Keep this
  lane's independent bit-identical KL check against b6f79e9 and ACE dense-oracle
  plus FAST/REF gradient checks. GPU FAST qualification still comes from the
  real matched pilot, not CPU synthetic evidence.
- Update the old full-admission fixtures to archive actual pinned source rather
  than reconstruct an old loop by reversing a few text replacements. Replace
  the S37 "only main changed" structural contract with an explicit reviewed
  new-lineage comparison; do not broadly waive arithmetic checks or hide the
  new Run metadata. Re-run full trainer, D2, S33–S37 and strict typing. Classify
  existing real-input skips separately from synthetic tests.

The five trainer `test_full_admission.py` failures on this branch arise at
`full loop changed beyond reviewed admission patch`, before the later gates
those tests intend to exercise. They are expected evidence of the blocked
full-run lineage, not failed ACE/KL arithmetic. The S37 static/dynamic lineage
checks likewise need the coordinated update above; their exact red nodes are
listed in the lane report. No assertions or pins are weakened here.
