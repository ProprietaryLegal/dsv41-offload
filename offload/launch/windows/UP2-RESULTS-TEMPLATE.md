# UP2 low-cap SCREEN result — review required

**SCREEN ONLY — ADOPTABLE: FALSE. Native qualification: NOT RUN / __.**
Production remains K1, replay OFF. PASS-SCREEN is not native qualification.

Decision: HOLD / Stage-B candidate only. Source/run paths: __. Production
21bed31/tag prod-k1-327-20261023; worker fb68c52. Actual identities verified: __.
Diagnostic qualification: PASS-SCREEN / FAIL / __.
Current measured window ID/output path: __. Measured contexts:300000 /150000 /both __.
Reviewed commit __; original diagnostic path __; NEW HOLD/deadlines __.
Report only this window's contexts below; mark the other context NOT RUN.
Never splice arms or contexts from different windows into one summary/criterion.
Native qualification missing/failed checks (all12 when absent): __. No adoption from this run alone.

**UPSTREAM CLAIM:** +4.9/+5.9% at478K; +6.5% at949K on the donor deployment.
These percentages are not local results. Our occupied prompts are150000/300000,
cap327680, output reserve1500. Exact usage/cold0, paired prompt/salt hashes and
same corpus/settings: PASS / FAIL / NOT RUN (__).

**MEASURED** (cite each run's source files; timing excludes audit/warmups):

| Occupied prompt | OFF-A TTFT median [min,max] | ON | OFF-B | Paired ON / mean(OFF-A,OFF-B), median [min,max] | Usable pairs / actual completed cells |
|---|---|---|---|---|---|
|150000|__|__|__|__|__/3; __/9|
|300000|__|__|__|__|__/3; __/9|

Ratio<1 means shorter end-to-end TTFT, not a measured GPU-kernel speedup.
OFF-A→OFF-B drift/noise: __. q source by diagnostic arm: MEASURED (packet path) /
ESTIMATED CPU4092 (__). Tails verified: false / __. Changed q/kept/work distribution: __. Timing anomalies
and missing samples: __. Partial cells remain missing; do not select repetitions.

| Safety/quality evidence | Result and exact source |
|---|---|
|Both-rank image,42 mounted hashes, target/worker/K1/proxy/Engram, cap/KV|__|
|Native q/tails, final real suffix/source-KV, owner/STEP/rank agreement|__|
|Near-cap K4/K5 sync/async and recovery/fault/progress/exclusions|__|
|A/A and ON/OFF coarse top20+OTHER KL≤max(.005,2*A/A), common mass≥.99|__|
|Isolated short greedy/finish, needles6/6 and zero foreign codes|__|
|Both-rank minimum memory/swap; engine/replay/watchdog/fallback errors|__|
|Completed representative task, visible answer counts and task time|NOT RUN / __|
|Per-stream mixed DELIVERY-EVENT p95/p99/max/count and accepted throughput|NOT RUN / __|

Submission accounting from actual ledgers: attempts__, HTTP-opened__, completed__,
failed__, canceled__, unresolved/dispatch-uncertain__, optional not-sent__.
Fault/native probes counted separately from cold matrix. Resource/telemetry gaps:__.
Restore: K1 START VERIFIED__, cap327680__, KV2502600__, greedy6/6__,
PRODUCTION HOLD OK__, reasoning/tool/structured smoke__, HOLD closed__.

Stage-B criterion at300000 in THIS window (all3 matched usable repetitions required):
`R_i = ON_TTFT_i / mean(OFF-A_TTFT_i, OFF-B_TTFT_i)`.
**MEASURED** paired ratios [__], median ratio__, gain `G = 1 - median(R_i)`__.
**MEASURED** paired spreads
`S_i = abs(OFF-A_TTFT_i - OFF-B_TTFT_i) / mean(OFF-A_TTFT_i, OFF-B_TTFT_i)`
[__], median spread `S`__. Exact raw-cell sources:__.
Stage B eligible only if `G > 0` AND `G > S`: PASS / FAIL / INCOMPLETE (__).
No repetition selection; missing/invalid pairs or a150K-only window mean INCOMPLETE.
Runner `stage_b` record and exact `window_id` source: __.
TTFT is an end-to-end prefill proxy; label any separately measured GPU timing.
A timing null/loss remains a valid SCREEN observation, never an API safety STOP.

Conclusion: local benefit / null / incomplete __. Build the native apparatus and
fully qualify only if the measured300K gain exceeds OFF-A/OFF-B spread above.
SCREEN never supports adoption, even when the criterion passes. Production stays
K1 with replay OFF. Before any adoption: native12 checks, decoder/mixed/task and
all source/safety/quality/restoration gates remain required.
**ESTIMATED** next-window time __, basis __; compare75min submission/90min hard
bounds and budget restoration. Current SCREEN estimates (S30 measured-input basis): diagnostic62–68min,
300K55–59min, optional150K43–48min; add ~7min ESTIMATED restoration plus checks.
Legacy measured-both75–81min exceeds75min at its upper end; use separate HOLDs.
Complete cells are not guaranteed; no deadline extension. STOP deadline/cause: __.
The **UPSTREAM CLAIM** at478K/949K is not local evidence and does not justify
Stage B after a null screen.400/450/500K phase not implemented; larger profile
remains HOLD pending explicit authorization and qualification.
