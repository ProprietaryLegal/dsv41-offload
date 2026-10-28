# PP3 probability path (OFF, prototype)

The two explicit switches are `DSPARK_SAMPLED_TOP32=1` on the K2 TP2 worker
and `VLLM_DSPARK_SAMPLED_TOP32=1` on the experimental engine. No launch, unit,
image, production staging or default configuration enables them here. Old peers
refuse the HELLO mismatch. No use of the older v1/sample draft_probs API occurs.

The negotiated v1 extension has sample_mode2, capability8, STEP/reply flag2.
This uniquely selects top32/tau.7/dyadic24-floor-first-v1, standard verification,
fixed depth4/5. Existing target/draft/Markov identities are still checked by HELLO.
Each request adds a 16-byte pre-row record: signed user seed64, T>0 bit32, zero
padding32. Same-round gather metadata carries the seed and T>0 bit from engine
state slots to the CPU proxy without an engine host synchronization. Request
RNG key is BLAKE2b(session,slot,generation,user seed); round64 and absolute
position64 form Philox counters with a distinct proposal domain. Positive target
temperature only determines whether to sample; independent drafter tau is .7.

Reply trailer: `<4sIII>` = PP3Q/semantic1/n/depth; int32 mask[n]; int32 support
IDs[n,depth,32]; FP32 q of the same shape; CRC32 of the entire reply body before
the CRC. Original header/statuses/tokens/confidence precede it. Full B1depth5
reply including frame is 1416 bytes (1280 sparse payload); B4 is 5412 bytes.
The request extension is 16bytes/request before alignment. CRC is corruption
detection, not authentication. Existing frame sequence/session, STEP round and
per-slot generation echoes remain mandatory. Missing q, wrong masks, corrupt
CRC, noncanonical/out-of-range IDs, nonfinite/negative/nondyadic probabilities,
wrong mass or absent proposed-token mass fail closed. No receiver normalization.
Even unused/greedy IDs are range checked before dense GPU scatter.

ReplyLayout grows only in sampled mode: B4depth5 is 5320 bytes, below the
existing 64KiB hostflag page capacity. Thread and process proxies, shared region
metadata and watchdog failed buffers use the same negotiated layout. The
existing claim/publish/release/consumed ordering and metadata-ring ownership
remain in force. Probability buffers use engine slot mapping, not batch order.
Sparse q becomes full-target-vocabulary FP32 q on GPU; off-support q is zero.
No full-vocabulary softmax or logits is copied to the host. The eager worker
prototype does stage full Markov-conditioned vocabulary shards GPU-to-GPU;
candidate-only peer exchange and graph capture remain performance gaps.

`prepare_verifier.py` checks the pinned read-only image sources and writes two
experimental overlays to an explicitly new directory. It never deploys. The
sampler sees real probabilities (the legacy draft_logits argument is reused
only as an ABI carrier with an explicit new semantic marker). Acceptance uses
log(p)-log(q) for the standard min(1,p/q) test; rejected draws use the full target
residual (p-q)+. q is never divided by target temperature, softmaxed, truncated
to the proposal vocabulary or rounded to BF16. Target processors/grammar remain
in the image's sampler. Finite precision log/exp arithmetic is qualified
statistically on the solo; this does not assert real-arithmetic bit identity.

The runner overlay marks all drafts of a failed request invalid **before**
standard verification. The image then uses its existing -1 placeholder path to
sample the unchanged full target directly. This avoids failure-dependent
point-mass substitution and keeps failed/missing q from sampled IDs. Both ranks
receive the same status/q under the existing broadcast. Greedy rows use the
original worker selector and image greedy branch. Reset/new-generation updates
all q/valid entries for the current batch's state slots; absent slots are never
consumed without a subsequent proposal update.

The sampled startup guard requires both overlay ABI markers, capacity policy,
no margins, <=4 slots, standard rejection and a fixed depth. It refuses block or
synthetic verification, watermarking, adaptive verification, depth-file reload,
prompt lookup and dynamic schedules. A generic direct call to RemoteRound is a
test surface; production admission must enter through the guarded speculator.

[GPU receipt](gpu-qualification.json) contains100K toy requests, including
sparse/disjoint target tails and failed-proposal placeholders;100K paired T0
checks compare every emitted token and sampled length. Uninitialized padding
past num_sampled in the image's output tensor is outside that contract.
Real async CPU process/thread tests cover the expanded handoff. These checks
are not end-to-end production K2, native pair, grammar-object lifecycle or GB10
qualification. No live A/B is recommended until those gates and exposed costs
are resolved. The required CPython3.13 interpreter lacks production EXL3/facc;
this lane did not substitute a CPython3.12 worker or install/rebuild dependencies.

Coordinator reproduction: copy the experiment package and pinned pure sampler
sources into designated scratch; run `tools/pp3_locked.sh --qualify` with source,
overlay and output arguments, in detached named PP3 tmux on x86host's permitted
solo under its own flock. Use the required miniconda Python3 interpreter. See
[prepare_verifier.py](prepare_verifier.py) for offline overlay preparation.
The prepared files need separate coordinator review, source attestation and
read-only experimental mounts; ordinary stage_overlay.sh deliberately does not
deploy them. Both production switches remain OFF and rollback needs no action.
