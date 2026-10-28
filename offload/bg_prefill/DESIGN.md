# SO2: priority-correct bounded background scheduling (default OFF)

This is a new overlay on the pinned installed scheduler/0031c, not SO1. NDM 0 and
legal 10 remain foreground; priority >=20 is auxiliary background. No request,
sampling setting, token, queue ownership or worker is relabelled. CPU qualification
is not permission to deploy. The production profile is unchanged.

## Policy and limits

1. **Steady mixed decode:** when foreground RUNNING requests have small ready decode
   demands and all work is ordinary RUNNING work, visit foreground first in numeric
   priority/arrival order. Background decode/prefill waits only after foreground
   actually schedules. Every 24th overlay call, allow EACH admitted background a
   32-token prefill quantum (or its pending decode rows, at most 16). Reserve proof:
   at most four requests, each foreground <=16 rows, background <=32 rows, plus
   <=4 draft slots/request, use <=144 of the deployed 4096 input rows. Foreground
   therefore never queues behind a full background prefill chunk. A due background
   bypasses 0031 deferral; the installed cadence stays 24, never 100000.
2. **Foreground waiting admission:** if up to three RUNNING background prefills
   precede a foreground waiter, no KV holders/connectors exist and the 4096/4
   budget is present, reduce their chunks to 32, never exclude them. This is the
   safe part of 0073: the foreground admission pass has budget, but an unadmittable
   foreground waiter cannot prevent admitted background work from advancing.
3. **Solo background:** preserve the configured long-prefill threshold (1024 in
   production), even for one request, in both running and admission paths (0074).
   Foreground solo retains native budgeting. In-flight chunks cannot be recalled.
4. **Native dependency fallback:** holding waiters, connectors, encoder/LoRA work,
   capacity-bound states, large/blocked foreground demands, nonpriority policy and
   unsupported row budgets disable pause/admission optimization. Keep native KV
   allocation, readiness, holder-before-fresh draining, preemption and async logic.
   No allocation is speculated or rolled back. Solo capping remains independent.

The 24-call service guarantee is a **runnable-resource guarantee in the qualified
steady mixed envelope**, including continuous foreground decode: every resident
background with pending work and successful native allocation advances at least
one token per interval, and prefill advances min(32, remaining) tokens. The limit
is calls, not wall seconds. Preempted/blocked/externally loading requests are not
runnable and are not quietly counted as serviced. Counters expose fallback,
maintenance, pause and admission-quantum calls so an unsupported live workload
cannot masquerade as qualification. Waiting admission is still native and may
fail; there is no universal admission deadline.

An unconditional guarantee for *every* admitted request within N steps AND zero
foreground delay is physically impossible: background compute consumes the same
GPU, foreground can fill every budget, and KV/dependency constraints can prevent
any legal allocation. Even a foreground arrival during an existing async chunk
must wait for that work. Here the due quantum adds bounded compute; in the stated
linear model two background quanta add 64/2200 = 29.1 ms once per 24 rounds. This
is an explicit fairness price, not the upstream's indefinite idle-only policy.
Outside the qualified envelope this revision preserves native liveness but does
not promise a new N-step bound. If the brief requires the unconditional literal
wording, verdict remains HOLD; no benchmark should claim that stronger property.

The call bound also assumes deployed 0031 cadence 24 and SO2 period 24 start in
phase. Both installed engine schedule call sites advance 0031 once per scheduler
call; the CPU cadence regression checks those sites and equal counters through
native fallback and return to mixed mode. Arbitrary phase offsets do not carry
the same bound (S28 O3).

UP2/0075 is a separate scheduler replacement. A clean Git merge does not compose
it with SO2: one hand-composed scheduler preserving replay-owner exemptions and
reserved floors, plus a fresh independent check, is required (S27 O6 / S28).
SO2 remains default OFF and roadmap P3 until a non-aborting background client
exists. Slot/KV residency and the solo-cap fixed-step cost must be assessed
before that workload is introduced; see the recipe's blocking preconditions.

## Upstream mechanisms

- 0072 decode pause is bounded by the due quantum and conditioned on actual
  foreground scheduled work; never wholesale exclude a selected waiter.
- 0073 waiting throttle becomes small guaranteed background progress, with native
  admission/readiness. No empty-step retry of an infeasible winner.
- 0074 solo cap applies only at >=20, independent of deployed throttle floor 5.
- 0071 verify-depth cap is deliberately **not ported**: donor hooks target its
  local depth-policy worker, while production uses the remote DSpark proxy.
  Scheduler pause already removes those background rows on 23/24 mixed rounds;
  on the service round their ordinary valid drafts remain intact. A remote cap
  needs a separately reviewed per-request TP-consistent contract, not installation
  of the donor local speculator. No draft truncation or output-equality claim.

D2 priority20 keeps its existing abort-on-first-foreign-poll policy (including the
recorded ~3 s ended-stream grace and possible chained grace masking). SO2 is a
server-side backstop before cancellation and for other priority20 clients. Abort
may end a request before its due share; that is explicit client cancellation, not
scheduler starvation. Do not disable abort to manufacture overlapping production
traffic. The sustained-overlap simulation intentionally models background clients
that remain resident, separately from D2's cancellation case.

## Sequence and evidence

Item 1 commits this design. Item 2 builds one scheduler from pinned image source,
keeps 0031c in place and tests installed sync/async statements via D2-FIX1's CPU
loader, including S15 running and S19 real-KV-holder regression states. Item 3
models one/two 60K prefills with measured-source step costs and supplies a
coordinator-only GB10 off/on/off recipe. No live traffic, GPU or services here.
