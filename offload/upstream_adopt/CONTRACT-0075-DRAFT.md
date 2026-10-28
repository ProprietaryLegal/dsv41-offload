# Proposed CONTRACT v1.9 suffix-provenance amendment (UP2)

DRAFT for coordinator integration review; CONTRACT.md v1.8 and wire version 1
remain the published contract. No frame, HELLO, STEP, descriptor or reply layout
changes. This amendment qualifies the forward-only completeness statements in
§1, §2 tap validity, §4.3, §7.2–7.5 and §13. It does not claim full causal target
recomputation or bit-identical draft results across EXL3 batch shapes.

## Replacement §4.3: only actual trailing computed rows

For an original request span of q rows, the engine may publish optional
replay_tap_lens_np (kept), in original batch order. None means the baseline
computation supplied every original row. With eager bounded replay, kept is
q for prompt-logprobs, min(q,128) normally, or exactly 1 for an eligible skipped
intermediate prefill. Kept is positive and no greater than q. Graph paths reset
this optional metadata on each preparation. Multimodal and prompt-logprobs
exclusions remain baseline. L20 source-KV writes still cover every original row;
the first SWA consumer preserves its original bound, later ones use real kept
suffix bounds. Intentionally uncomputed tap rows are never accepted history.

Candidates are the last min(q,kept,128) rows of the original span. Valid count
subtracts the rejected suffix; first_pos is the original position of the first
candidate, anchor_pos = first_pos + valid. The anchor token is last_sampled
when sampled>0, otherwise next_prefill at the request-state slot. Rejected rows
are excluded in both Torch and fused gather. An entry requesting drafts may
not discard any of its required newly computed candidate rows.

BatchView keeps max_q in its existing positional slot; optional tap_lens follows
it. This is engine metadata. Once gather has formed valid/first_pos/anchor, the
process descriptor needs no tap_lens field or wire change. Slot/generation,
request owner, verified depth and per-round metadata ownership are unchanged.

A skipped predecessor creates a per-request successor obligation only after
capacity for the next ready turn has been reserved. That successor trims
(q>128), unless the scheduler first arranges real suffix replay. The current
four-request 4096-row scheduler reserves a positive floor before earlier work
can spend it. Small persistent leftovers cannot silently yield zero forever.
KV loss, preemption, backward movement or restore invalidate the old promise:
retire/re-add via the installed prefix replay path or explicitly recompute the
hit's real suffix with replay_start. Reconstructed rows may arrive in bounded
chunks; their actual contiguous range governs readiness, never a token counter.

## §7 clarification: real history, generations and resync

The device ring and rank-0 range describe the SAME real rows. Gaps discard the
old contiguous range. Positive backward writes replace rows at/after first_pos;
a zero-row rollback may only shrink or empty the range. No operation widens lo
to anchor−128 to satisfy a caller. Each new slot owner changes generation and
clears its range. Generation wrap/echo and stale-reply validation retain v1.8.

With 0075 ON, before ANY NEED_DRAFT STEP the rank-0 ring must contain exactly
min(anchor_pos,128) contiguous real rows ending at anchor_pos−1. Otherwise no
remote draft is requested: fallback_replay_suffix_incomplete is raised/counted
and qualification stops. Do not accept PARTIAL_WINDOW as successful replay
qualification. Disconnection, timeout, watchdog or lost proxy marks dirty state;
NEW_GEN/RESYNC may resend only the current generation's actual contiguous rows.
A final real suffix can recover a disconnect before the last chunk; a disconnect
after it can recover from the real retained 128 rows. A stale buffer, late reply
or old owner can never supply missing history. Missing history below a skipped
gap requires actual target replay, not repeated RESYNC of the same partial ring.

The async proxy retains 64-slot round metadata ownership and private STEP header
and resync buffers. The owner of a completed round cannot write into newer engine
staging. Qualification must preserve watchdog/late-publication regressions.

## Optional two-rank qualification audit (default OFF)

VLLM_DSV41_REPLAY_AUDIT=1 constructs independent metadata on both ENGINE ranks
at the existing round entry: q, kept, first_pos, valid, anchor_pos/token,
slot, opaque owner digest and per-slot engine owner epoch. Both ranks exchange
these over the existing TP CPU group and fail together on any mismatch/local
validation failure. Local errors are exchanged before raising, so one malformed
rank does not simply omit its collective. Both ranks must set the same option
and load identical reviewed sources before startup. This is a diagnostic that
synchronizes device metadata; disable it for headline timings on every arm.

Engine owner epoch records observed slot ownership on each rank. It is NOT a
replacement for the worker STEP generation: disconnected/proxy-dead rounds can
skip worker delivery. Existing rank-0 STEP generation and reply-generation echo
checks remain authoritative. Require zero stale-generation/bad-reply events in
normal qualification, identical broadcast draft/fallback decisions on both
ranks, and explicit recovery during faults. No private metadata, anchor IDs,
tap tensors, prompt IDs or generated text is logged by the audit; only aggregate
audited-round counts and fixed error strings. Collectives are CPU-test injected;
real TP-group/native parity remains a GB10 gate.

## Proposed §13 entry

v1.9 (pending): optional real-suffix engine metadata; positive reserved successor
minimum; prefix-replay recovery; full real-window gate before drafts under 0075;
optional two-rank provenance qualification. Wire version remains 1. CPU evidence
validates state/statement boundaries with synthetic data, not hardware parity.
