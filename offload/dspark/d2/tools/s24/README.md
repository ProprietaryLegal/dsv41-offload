# S24 probes — independent merge-check of D2-PILOT-R3 (`offload/d2-pilot` @ e06a1e6)

CPU only, synthetic data only; no real plans, receipts, tensors, prompts or generated text. Run from the repo root:

```bash
export CUDA_VISIBLE_DEVICES='' TMPDIR=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/S24
python3 -m pytest offload/dspark/d2/tools/s24 -q -p no:cacheprovider -W ignore::UserWarning
```

| file | brief item | what it checks independently |
|---|---|---|
| `test_s24_item1_amendment6.py` | 1 | Exact `Fraction` oracle of Amendment 6 written from the amendment text, re-implemented paired family→request bootstrap (2,000, seed 20261024, exact rational 5th/95th); 150-case differential fuzz; exact boundaries with no epsilon; early checkpoints non-gating; amendment 6b regression: definitive failure remains FAIL alongside an unrelated straddle |
| `test_s24_item1_policy.py` | 1 | FAIL / INCONCLUSIVE receipts refused by production even when forged with every digest recomputed; old rule, synthetic, interval and dev-label tampers refused |
| `test_s24_item2_units.py` | 2 | FAST-B interval populations at 20,000 / 23,017 / 2.4M; LR list against torch's own AdamW+LambdaLR at 159 updates; unit tamper and dataset recheck; resume keeps counters |
| `test_s24_item3_audits.py` | 3 | Zero-init signal reaches every A at step 2; A=B=0 dead-adapter counterexample; projection-cache entry points; REF autograd and FAST analytic CE/KL/TV vs FP64 full-vocab unfused; whole-model gradients vs plain autograd with canonical STE |
| `test_s24_item4_sharding.py` | 4 | `install()` TP2 deltas vs the training `Dense` / grouped wo_a math with uneven vocab split and two heads/groups per rank; mis-sharding counterexamples; FP32 norms; guard scope; service-CLI unreachability; lease and launcher refusal of the production pair |
| `ref_train_dump.py` + `evidence/ref-train-bitwise.txt` | 5 | Reference/default training bit-identical to d5a103e in five cases |
| `run_suites.sh` + `evidence/suites.txt` | 6 | Every CPU test directory on exported trees; red sets compared with offload-x86host's own |

R4 imports these files without merging the checker branch. FIX-1 now passes without xfail; the source-family probes cover transitive bridges and shared-prompt A-captures. The exact oracle now applies amendment 6b; archived evidence remains the original S24 record. R4 measurements live in the campaign logs.
