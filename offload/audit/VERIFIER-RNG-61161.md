# AUD1 item 1: production's verifier vs vLLM #61161 (random draws reused across verification rounds)

**Verdict: NOT AFFECTED.** Production verifies the remote drafter's proposals with *standard* token-by-token
rejection sampling (not block verification), with one-hot q (draft_logits=None) at T=1. On that path each
acceptance uniform is read at most once for a position that is not committed in the same round, so no random draw
is reused after it has influenced a decision. #61161 fixes block verification (and the draft-side noise of locally
sampled drafts). Neither runs in production. No backport is needed, and none was made.

The image runs the code from before the fix (`u = tl_rand32(seed, pos)` with no step mixing). A CPU test runs that
same image kernel through Triton's interpreter. On production's path the output distribution is correct. Switched
to block mode, the same kernel shows #61161's bias, and the PR's one-line change removes it (MEASURED below). The
bug therefore exists in the image. Production cannot reach it unless someone sets
`rejection_sample_method="block"` (or enables locally sampled drafts, see "What would make it AFFECTED").

Sources: the image's read-only copy `/opt/dsv41/data/src-image/` (vLLM 0.31.1.dev0+g6e517b15.dsv41gb10)
plus production's overlays at sparkhead main **21bed31** (`git show 21bed31:dsv41.env` OFFLOAD_DOCKER_ARGS: K1
`kvpack/overlay` incl. `v1/worker/gpu/model_runner.py`, `remote_dspark/`, `spec_decode/__init__.py`). The
verifier files are image originals; no overlay replaces them. PR: vllm-project/vllm#61161, merged 2026-10-15,
merge commit 9edcc7fa7ec858e13d4dc10ed968b741e2c95d6a (read with `gh pr view/diff`, nothing executed).

## What #61161 changes and why

Block verification (Sun et al.) reads the acceptance uniform of *every* draft row and takes the last row with
`u <= h` as the accepted length. So the committed outcome carries information about uniforms at positions past the
accepted prefix. In the next round those positions are verified again, keyed by the same `(request seed,
position)`, which reuses randomness already conditioned on the earlier decision. The PR's measurements (UPSTREAM
CLAIM, 10,000 requests): deterministic draft + block 0.4171 vs 0.5000 expected; probabilistic + block
0.3558 vs 0.3000. Standard verification: deterministic 0.5039 both before and after, with an *unchanged output
hash*; probabilistic 0.3083 before, 0.3055 after (the target distribution was already retained). The fix mixes the
draft step into the seed's high 32 bits (`spec_step_seed(seed, i)`), **only when USE_BLOCK_VERIFICATION**, plus the
same step key for cached draft Gumbel noise (local probabilistic drafting / DFlash2 / watermark fallback).

## Trace of the path production reaches (file:line, image copy unless marked)

| Step | Where | What it establishes |
|---|---|---|
| Spec config | 21bed31 `serve.sh:34` `SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}'` | no `rejection_sample_method`, no `enable_adaptive_verification` |
| Default method | `vllm/config/speculative.py:530` `rejection_sample_method = "standard"`; `:552` `enable_adaptive_verification = False` | standard verification |
| Verifier object | `v1/worker/gpu/spec_decode/rejection_sampler.py:163-175` | `use_block_verification` is True only for `"block"` |
| Remote speculator | 21bed31 `remote_dspark/speculator.py:56-58` refuses `draft_sample_method != "greedy"`; `:65-66` refuses adaptive verification; `:272` `self.draft_logits = None` (capacity class, `VLLM_DSPARK_REMOTE_POLICY=capacity`) | one-hot q; no local draft sampling |
| X86host drafts | 21bed31 `offload/dspark/worker/model.py:420,428` (Markov argmax / `argmax`) | proposals are deterministic functions of committed context and taps |
| Call site | 21bed31 K1 overlay `v1/worker/gpu/model_runner.py:1714-1720` `rejection_sampler(logits, input_batch, self.speculator.draft_logits, invalid_drafts)`; `:1689-1700` invalid drafts = grammar and (if a depth policy is on) dead rows | `draft_logits=None`; invalid rows become -1 at `rejection_sampler.py:433` |
| Depth policy | `depth_policy.py:140-141` is on only if `VLLM_DSPARK_DEPTH_POLICY` is set; 21bed31 dsv41.env/serve.sh never set it | no dead rows in production (the test covers them anyway) |
| Seeds | `rejection_sampler.py:278` passes `sampling_states.seeds`; `sample/states.py:54-58` uses the request seed, or `np.random.randint` per request when unseeded | one 64-bit seed per request, fixed for its life |
| Row positions | `rejection_sampler.py` `_gather_draft_sampled_kernel` copies `positions[logits_indices]` to `pos`; positions come from `prepare_pos_seq_lens` on GPU `num_computed_tokens` (K1 `model_runner.py:1506-1511`) | row i of a round is keyed by position `start + i` |
| Commit | `input_batch.py:610` `computed_delta = query_len - num_rejected` (GPU `num_computed_tokens` is authoritative; `states.py:61` the CPU copy is only an optimistic upper bound) | the next round starts exactly after the last committed token |
| Acceptance draw | `rejection_sampler_utils.py:566-569`: `verifying &= is_valid_draft` (non-greedy), then **only `if verifying`** `u = tl_rand32(seed, pos)` (Philox, `gumbel.py:67`) | a uniform is read only while verification is still running |
| Standard test | `rejection_sampler_utils.py:662-663` `accepted = target_logprob > log(u) + draft_logprob` (draft_logprob 0 for one-hot); `verifying = accepted` | standard verification stops at the first rejection |
| Block test (not production) | `rejection_sampler_utils.py:595-627` `accepted_length = tl.where(u <= h, i + 1, accepted_length)` over all rows | the #61161 condition |
| Correction / bonus | `_resample_kernel` `:832-889`: one-hot residual = target with the rejected id masked (`:884-885`); bonus or -1 uses target logits (`:832`); `_seeded_resample_argmax` `:907/920` uses Gumbel `murmur3_uniform32(seed, pos, token)` (`gumbel.py:198`), at the resample row only | one correction draw per round, keyed by that row's position, a different generator from the Philox acceptance draw; draft noise is salted separately (`gumbel.py:18`) |

## Can an uncommitted position be verified twice? (standard, one-hot q, T=1)

A round's rows are keyed by `start .. start+k`. Standard verification reads uniforms for rows `0..a`, where `a`
is the first rejected row (or rows `0..k-1` if all are accepted). It commits `a+1` tokens: row i's decision
writes the token at position `start+i+1`. The next round therefore starts at `start+a+1`, beyond every row read.
Each case:

- **Partial accept.** The rejected row's position is committed in the same round with its correction token. Rows
  after it were never read. Nothing read stays uncommitted.
- **Async rounds (process proxy).** Round t's drafts are computed from round t's gather (`num_sampled`,
  `num_rejected`, `last_sampled`, `round.py:387-400`). Round t+1's verifier gets fresh rows from the GPU-side
  positions. A late or failed reply means zero drafts and zero confidence: token id 0 at every slot
  (`speculator.py:307-309`, `round.py:517`), still a deterministic proposal verified once. With no drafts in the
  batch, the plain sampler runs (K1 `model_runner.py:1707-1709`).
- **0047 adaptive window and the per-batch-size schedule.** The scheduler (`scheduler.py:2025-2036`) picks the
  draft count from the request's EMA of committed acceptances, so from history only. Fewer rows does not mean
  re-reading.
- **Placeholders, grammar-invalid drafts, dead rows (-1).** Verification stops at the -1 without reading its
  uniform (`:566`). The row before it is sampled from the full target.
- **Padded windows (CUDA-graph padding).** Padded rows are outside `cu_num_logits` and never reach the kernel.
- **Preemption / recompute.** The request returns with its committed tokens as prefill (`states.py add_request`),
  so verification resumes past them. A discarded in-flight result that is recomputed with the same explicit seed
  replays from the same history. Nothing committed depends on the discarded draw, so this is a replay, not
  conditioning.
- **Retries.** Client retries are new requests: a new random seed, or the same explicit seed replaying from the
  start.

In block mode (not production), rows `0..k-1` are all read and the next round starts at `start+a+1 <= start+k-1`
whenever `a < k-1`. That is exactly the reuse.

## CPU evidence (MEASURED, Controller CPU, Triton 3.7.1 interpreter, torch 2.12.1 CPU, Python 3.13.15)

`offload/audit/verifier_rng/`: `kernel_loader.py` imports the image's `rejection_sampler_utils.py` and
`sample/gumbel.py` unchanged under `TRITON_INTERPRET=1`, `CUDA_VISIBLE_DEVICES=` (only `vllm.triton_utils` and
the unused watermark kernel are stubbed). `multiround.py` drives multi-round generation through
`rejection_sample()` exactly as the V2 runner commits tokens (draft_logits=None, T=1, a random int64 seed per
request). It records the kernel's read set to count re-read keys. Synthetic toy targets only.

| Case (seeded, reproducible) | Trials | chi-square / dof (bound at alpha 1e-4) | Re-read keys | Result |
|---|---|---|---|---|
| Production path, #61161's case (p=(.5,.5), draft always 0, k=2, 6 tokens) | 2,000 | 48.3 / 63 (bound 113.7) | 0 | unbiased |
| Production path, context-dependent 3-token Markov target, drafter right about half the time, windows 1..5, dead rows | 3,000 | 74.5 / 80 (bound 135.9) | 0 | unbiased |
| Negative control: same kernel, block mode | 2,000 | 931.2 / 63 | 3,831 | **biased** (#61161 reproduced; second-token P(0)=.4325 vs .5) |
| Negative control, context-dependent case, block mode | 3,000 | 240.0 / 80 | 5,641 | **biased** |
| Positive control: block mode with #61161's kernel line applied to a copy | 2,000 | 61.8 / 63 | 0 (step-keyed) | unbiased |

These figures come from development runs. The committed tests use fixed seeds and assert the same outcomes:
`TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider --basetemp=<scratch>
offload/audit/verifier_rng/test_verifier_rng.py`. Result: **6 passed in 85 s** (16 CPU processes). Strict mypy and
ruff are clean. The tests show the kernel's logic; GPU float rounding is not part of the claim. They also show the
statistic detects the bug: the block-mode test fails on the image kernel and passes with the PR's change.

## Where production's T=1 / top_p=1 comes from

1. **The request.** Every client in this repo sends them explicitly: `RUNBOOK.md:94` (`"temperature": 1.0,
   "top_p": 1.0`), `functest.py:11`, `quality.py:11`, `offload/gb10_budget/capture.py:154`, the bench settings
   recorded in `offload/results/report-*-raw.md` and `BENCHMARKS.md:30` ("DeepSeek/author recommendation").
2. **If a request omits them,** the server applies `user -> server default -> OpenAI default`
   (`entrypoints/openai/chat_completion/protocol.py:662-675`). The server default comes from the model
   directory's `generation_config.json` (`--generation-config auto` is the default, `config/model.py:356`,
   `get_diff_sampling_param` `:1831-1851`; production's launcher passes no `--generation-config` or override). With
   no such file, the OpenAI defaults apply: temperature 1.0, top_p 1.0, top_k 0 (`protocol.py:626-633`).
3. **The model card** recommends temperature 1.0 / top_p 1.0 (`/opt/dsv41/data/upstream/hf-cards/
   deepseek-ai_DeepSeek-V4.1-Flash-1017.README.md:77-82`). The base repo at 1bae0e1 ships **no**
   `generation_config.json` (HF API sibling list in `hf-cards/...api.json`), and neither do the sudoingx EXL3 repos.
   Whether production's `/srv/dsv41/legal-v2a` has one cannot be checked from Controller. Coordinator, light read:
   `ssh sparkhead 'cat /srv/dsv41/legal-v2a/generation_config.json 2>/dev/null || echo none'`. If it exists and sets
   other values, requests that omit sampling fields use them. The verdict holds for any T > 0 and any top_p.
   T = 0 is the greedy branch, which draws no randomness.

## What would make it AFFECTED (keep in mind for PP2 and later work)

- Setting `"rejection_sample_method": "block"` in SPEC. Block verification on this image needs #61161's two-line
  kernel change first (and the test above as its gate).
- Locally *sampled* drafts (`draft_sample_method=probabilistic` on a local DSpark drafter) under block verification.
  For PP2's sampled remote drafts under standard verification, the reused-uniform bias does not apply. The q and
  draft RNG must still be keyed per round/step, never by `(seed, position)` alone. PP1's reference already uses
  round-owned streams. Any revisited-position draft noise must follow #61161's `spec_step_seed` rule.

## Coordinator check (optional, light, no restart)

Confirm the container's verifier files are the audited ones:
`ssh sparkhead "docker exec dsv41-serve sha256sum /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py"`
Expected: `ef1d40bf…d8d90`, `25759f98…2066a`, `c14a5fa1…868e`. The first two match PP1's hashes of the same image
copy.

Full SHA256 of the audited files: rejection_sampler_utils.py ef1d40bf31140eb644a8f3b7a003373209d6a8e297f059e70ea66cf4972d8d90;
rejection_sampler.py 25759f98e8aace9a556bc4252c206f246535f0536ebc919f97e097f92a72066a;
sample/gumbel.py c14a5fa1899a9ff320c62f29bd161be4a4f0fa124ecf26a28681f0f3d143868e;
sample/states.py 7dbccea6fb6e24830dafb447a43f93fd0539f33ef02f1b5bf7c2547234cbffa3;
input_batch.py 96163c486368f2cf0e539fe68eaa06f4df3ac814da8f7a92af2b04242e01bc6d;
config/speculative.py 49c67bc2595024e5412be9579b7e88139dcbe3976d47dad0ffe5c0f3a668d7f7;
chat_completion/protocol.py f23a9cb2ebc044fdfae3eef866ea7f8846d81d978f3f5303fa0d084fea97940e;
21bed31 K1 model_runner.py 682eb895b8fb3d68b9580954b6cda255a49f882bddb4b06342d7d96258eb6e62;
21bed31 remote_dspark/speculator.py 441be6b7d245114974c5dcb307c2efebd29516b958cd363bc70e24bcd73bdcf3.
