# Startup warm-up (OPS1, reviewed patch; not deployed)

The old helper requested 32 tokens but did not require them. MEASURED:
`/opt/dsv41/campaigns/DSV41-OFFLOAD/K1-production/start.log` records
5.629 s, 5,041 prompt tokens and **1 completion token**. Its raw completion
request has effective explicit max=32, default min=0, ignore_eos=false, no
explicit stop. The installed completion protocol merges generation-config
stop IDs. EOS or a default stop is a plausible explanation, **not a confirmed
cause**: the old helper discarded finish_reason, stop_reason, text and metrics.

Read-only `docker logs` on sparkhead (container started 20:46:08Z on October 23)
confirms deepseek_v41 tokenizer, reasoning and tool parsers, TP2 target,
max_model_len 327680, max_num_seqs 4, max_num_batched_tokens 4096, speculative
k=5 for B1/B2 and k=4 for B3/B4, process proxy and remote pair connection.
Raw `/v1/completions` does not apply chat templating or invoke chat reasoning/
tool output parsers. The old response's raw token IDs, EOS/stop reason,
reasoning count and attributable worker-round count are **unavailable**.
Completion usage is not visible-answer usage or a worker-round count. Do not
infer any of them from the nominal 32-token budget or unrelated request logs.

The local read-only engine snapshot supports min_tokens and ignore_eos in
`vllm/entrypoints/openai/completion/protocol.py:to_sampling_params`, and exposes
finish_reason/stop_reason from the actual RequestOutput in completion/serving.py.
Per-request speculative metrics are emitted only when enabled in the engine;
missing metrics and reasoning counts are reported as null, never zero.

## Patch and limits

`warmup_serve.sh` keeps the current start.sh interface; its new stdlib-only
`warmup.py` issues internal raw completions with **min=max=32**, ignore_eos=true,
no explicit stops, greedy sampling, priority 20. The current main start.sh
already invokes the wrapper after async startup. No start.sh/env edit or server
sampling-default change is needed. Its one-token response is now an error,
with sanitized counts and finish/stop reasons. Generated text/IDs are never
printed or saved. There is no separate hidden reasoning budget: at most 32
completion tokens per request, including any reasoning-like continuation.

Four bounded waves request B1/B2/B3/B4 (ten requests, 320 completion tokens
maximum). B1 has a unique synthetic prefix and roughly the previous 5K size
(ESTIMATED); returned prompt count must exceed4096 and reported cached count
must be zero. B2/B3/B4 use short synthetic prompts, so long serial prefills do
not consume the whole32-token decode interval before other streams can join.
Barrier-released clients target both live depth bands. Client concurrency is
**not proof of engine batch shape**: request completion skew, chunked prefill
and adaptive verification can produce smaller shapes or tail rows. Long-context
attention, every verification depth and future B6/B8 are not covered by this
startup sequence. Time bounds: each socket operation waits at most the time left
of a 90 s helper deadline (and never more than 60 s); a wave never starts after the
deadline and a run that finishes late fails; `warmup_serve.sh` ends the whole process
at 95 s (`timeout --kill-after=5s`, overridable by `WARMUP_HARD_S`) because a trickling
read or a waiting worker thread can outlive the helper's own deadline. Any of these is a
failure, which `start.sh` reports as its existing WARN; with the last K1 start at 432 s
the warm-up cannot reach K1's 600 s start guard.

Before every wave, both request gauges must exist and be zero, otherwise the
helper refuses. This is an additional guard, not a producer lock or a
preemption guarantee. Coordinator must keep D2/local-law and other producers
paused behind HOLD throughout startup and warm-up. An already submitted wave
has a bounded token budget; no later wave starts if other work remains.
The main caller preserves its existing WARN-on-failure behavior, so inspect
warm-up completion records before releasing HOLD; service readiness alone does
not establish warm-up success.

## Qualification / application (coordinator only)

1. Review/cherry-pick this item from offload/ops1 into the intended production
   branch. This lane did not deploy, request, restart or benchmark anything.
   Preserve K1's saved dsv41.env/start.sh hashes; this patch changes neither.
2. CPU: `CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s offload/launch/tests -p test_warmup.py -v`;
   `bash -n offload/launch/warmup_serve.sh`.
3. In the next exclusive startup window, save fresh x86host worker status and
   proxy counters **before** warm-up and immediately **after** it, with matching
   pid/session, timestamps and no foreign traffic. Worker `steps` delta measures
   actual worker RPC steps; `drafted_requests` delta counts per-request proposals.
   Engine per-request `num_spec_steps` is a verification boundary, not necessarily
   a worker RPC boundary. Do not mix them or rely on the proxy's 1024-round
   periodic stats to resolve a 32-token request. Missing/stale data leaves that
   check incomplete. Existing status is written on a timer; wait for a fresh
   timestamp before reading each snapshot.
4. Require all ten responses to return completion_tokens=32, finish_reason=length,
   stop_reason=null; record actual prompt counts and cached tokens separately.
   Correlate worker graph `(B,k,n_out)` and target verify-row dispatch/capture
   records for B1/B2 k5 and B3/B4 k4, including smaller acceptance tails. If shapes
   are not observed, add a bounded coordinator-only shape test; do not lengthen
   hidden reasoning or claim the wave size proves capture coverage.
5. Compare the first subsequent short/32K request and B3/B4 burst with later
   warm bursts (TTFT, visible-answer timing, worker rounds, failure/watchdog
   deltas). Existing semantics/greedy regressions still apply. Reject startup
   timeout regressions. Keep long-prefill first-use qualification separate.
6. Roll back only `offload/launch/warmup_serve.sh` and the added helper to the
   parent revision if needed; K1 manage.py restore and service changes are
   coordinator-only and not part of this patch.
