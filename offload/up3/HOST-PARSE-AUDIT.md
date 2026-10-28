# Round 6 remote parser audit — 2026-10-27 UTC

## Round 8 superseding poller proof

The CPU-pin-only overlay in `RDMA-PIN.md` brackets the exact image
`rdma.py:105` native start. `exl3_rdma.cu:566-570` creates the thread there;
`:266-272` performs its asynchronous self-pin. No native name setter is
assumed. The overlay itself now produces the unique `exl3-rdma-prx` name and
versioned JSON receipt, including both process/thread starttimes and
CLOCK_MONOTONIC nanoseconds. It does not alter production or the control arm.

`ExperimentReads.pin_receipt` permits only local Docker exec cat of
`/run/up3-rdma-pin/PID-STARTTIME.json` for the required worker discovered by
the existing argv/tree census. Numeric PID is bounded by the same namespace
ceiling and starttime must be decimal ASCII; no path or shell input is taken
from a receipt. The read is bracketed by both stable census/identity sweeps.
Duplicate JSON fields, missing proof, wrong lifetime/name/TID/affinity and any
unreadable required receipt refuse. Production probing does not read receipts
or attempt thread naming/pinning. Native/Engram attribution beyond this exact
poller startup remains unmeasured.

Only a receipt-matching current named nonleader may be on CPU14. Every other
required thread and all leaders must have5-9,15-19. The lost pin-init log stays
optional corroboration. Existing shared recovery is unchanged; new both-rank
actual-runner rehearsals assert no candidate timing and complete fake W8
start/check/close on failed receipt proof. This is CPU/source evidence, not a
live pinned-arm or restoration result.

## Round 7 superseding CPU-pin census and remaining blocker

The pin-init line is optional corroboration: S73 proved that `_pin_cpus()`
runs before logger handlers are attached (`__init__.py:53,60,69-75`). The new
`proc_census.py` discovers processes without that line. `ProductionReads.proc`
performs only `docker exec dsv41-serve cat` of finite, validated /proc paths.
It first reads the PID namespace's `ns_last_pid`, then reads numeric TID status
files, which Linux exposes even for nonleaders omitted by /proc directory
enumeration. Missing numeric slots are allowed only during discovery, with
exact ENOENT diagnostics. Permission errors and every missing required read
refuse. No remote Python, directory listing, process creation beyond the
authorized cat reader, model import or affinity write is used.

Two sweeps must agree on required process identity/argv/tree, all required TIDs,
comms, thread counts and affinities; scheduling state may legitimately change.
Every required TID's stat starttime is checked between those sweeps and again
afterwards. The leader must be present with its group's exact Threads count.
PID namespace values above16,384 refuse; cat batches contain at most512 paths.
More than eight process leaders refuse before any argv reads.
All stdout/stderr remains separate so an unexpected cat/SSH error cannot become
a valid status record. Final Docker identity is rechecked.

Real read-only observations on sparkhead and sparkworker-wired (2026-10-27):

| Role | rank0 | rank1 | Identification / source |
|---|---|---|---|
| Launcher | PID1, comm `vllm`,70 threads | PID1, comm `vllm`,79 threads | `/usr/bin/python3 /usr/local/bin/vllm serve ... --node-rank 0/1`; rank1 headless. |
| Engine | PID171, parent1,76 threads | absent | argv `VLLM::EngineCore`, comm truncated to `VLLM::EngineCor`; `vllm/v1/engine/core.py:1380`, `utils/system_utils.py:179-193`. |
| Worker | PID243, parent171,638 threads | PID187, parent1,635 threads | argv `VLLM::Worker_TP0/1`, comm truncated to `VLLM::Worker_TP`; `v1/executor/multiproc_executor.py:1083-1097`, same set_process_title. |
| Proxy | PID461, parent243,21 threads | absent | Exact isolated Python BOOT, package path and numeric region FD; `596ba1c:offload/dspark/spark/remote_dspark/proxy_proc.py:433-458`. |
| Runtime helper | resource_tracker170 | resource_tracker186 | Exact Python multiprocessing.resource_tracker argv; excluded from engine/plugin/proxy roles. |

The projected fixture contains all806/715 observed tasks, exact argv/comm/masks
and counts. These containers were on **experiment8001**, so this is observed
process-shape evidence, not a production8000-probe pass or CPU-pin qualification.
Stat starttimes in the replay fixture are explicitly synthetic test data.
All normal threads had default0-19 affinity, with one CPU14 thread per worker.

**Poller identity remains unavailable.** `vllm_exl3_moe/csrc/exl3_rdma.cu:267-272`
sets the native proxy thread's CPU; line570 constructs `std::thread(proxy_loop,
&g)`. This file contains no pthread_setname/prctl name setter. The observed
CPU14 thread has `VLLM::Worker_TP`, shared by hundreds of nonpoller threads.
That comm cannot justify a poller-only exception. The validator now refuses a
CPU14 thread even when it is a nonleader with that shared comm. The required
leader retains the exact registered mask check. No source-defined unique name
is invented; CPU-pin is **NOT CLEARED**, pending reviewed identity evidence.
The image, arithmetic/config arms, runner, shared modules and budget are unchanged.

Round7's two `--probe-production` attempts refused the actual8001 launchers
before further probe reads. Follow-up direct-census attempts saw an external
container replacement/removal and failed inspect/exec. They did not claim
success, retry a window, issue a port request or change production. Final live
results and timestamps are recorded in the campaign report. Final bounded
retries succeeded on BOTH production8000 containers with stable IDs and empty
error lists: rank0 launcher1 (67 threads), engine173 (73), worker245 (638),
proxy463 (21); rank1 launcher1 (79), worker187 (635). Masks were default0-19
except worker TID1150/rank0 and1010/rank1 on14, both with the same shared
`VLLM::Worker_TP` comm. Both receipts remain explicitly NOT QUALIFIED.

## Historical Round 6 audit

Scope: every remote result consumed directly by `host_gates.py` and
`up3_window.py`, including inherited capture/log adapters. Production reads were
limited to Docker inspect/logs/exec cat and sysfs. No inference request, HOLD,
preflight, GPU query, process census or mutation was run on a Spark.

## Source-derived log bodies

| Consumer | Exact producer | Result |
|---|---|---|
| RDMA initialization | `/opt/dsv41/data/src-image/vllm_exl3_moe/rdma.py:108-110` | Full rank/world/devices/GID/CPU/KB/slots/all-gathers message; auxiliary routing message at line 188 is separate. Prior shortened fixtures omitted the real suffix. Parser and fixtures now require its full shape. |
| CPU pin | `/opt/dsv41/data/src-image/vllm_exl3_moe/__init__.py:53` | Full `vllm_exl3_moe: pid ..., ... threads pinned to CPUs [...] (EXL3_CPU_PIN)` body; decimal PID/count and Python sorted-list representation. |
| Rank-0 proxy | `596ba1c:offload/dspark/spark/remote_dspark/speculator.py:152-155` | Full policy/endpoint/PID/device/shared-region message. This production bind-mounted module is **absent** from src-image; citing an invented mirror path would be wrong. Read-only cat on both live containers matched the pinned production file byte for byte. |
| Fallback | `src-image/vllm_exl3_moe/rdma.py:162` | `RDMA all-reduce off ... NCCL stays` remains a refusal. |
| Inherited rank log filter | `pf1_window.py:89-113`; source `src-image/vllm/v1/core/kv_cache_utils.py:2482`, K1 overlay `offload/kvpack/overlay/vllm/v1/core/kv_cache_utils.py:2507`; production `remote_dspark/client.py:336`, `speculator.py:152,216`; Engram overlay `offload/engram/spark/engram.py:1190,1402` | Capacity substring, proxy PID, worker-ready, async-mode and remote-table messages match their producer formats. No shared-filter edit. Live captured logs prove proxy and async parsing on rank 0; missing production startup fields are not fabricated as measurements. |

Both live containers' `rdma.py` plus `__init__.py` were byte-compared to the
read-only src-image files, without importing them. The logger template fixture
records exact source lines and SHA256s. Pin/ring-init positive cases are rendered
from those literal templates, **not** represented as observed production messages.
Production logs at capture contained neither pin-init nor rank ring-init; they
contained the distinct RDMA-routing line (two on rank 0, one on rank 1).

## Inspect, launcher and sysfs

Actual inspect on both ranks: `Id` is a full 64-hex string, `Image` has a
`sha256:` prefix, `State` is an object (`Running=true`, `Paused=false`,
`Status=running`), `Config.Env` is a list of `KEY=value` strings, `Config.Cmd` is
`["/serve.sh"]`, `Config.Entrypoint` is `["/bin/bash"]`. The `/serve.sh` bind
mount has Destination `/serve.sh`, RW false and a host-specific Source. The
projected fixture retains exactly those captured fields, not credentials or
arbitrary environment values. `Live.capture` already parses the correct `Env`
list and splits once at `=`; inherited capture parses two JSON objects and mount
Destination strings. No shape mismatch beyond item 1's Cmd-port assumption.

`runtime` reads its argv from actual `/serve.sh`, already joins continuations
and shell-splits quoted JSON. Item 1 repaired the separate host-gate assumption.
The real production launcher is a test fixture; only its port changes for the
positive experiment case. Full environment/source equality and source pins are
unchanged, including the previously corrected mounted-`.py` restriction.

Exact sysfs paths are `/sys/class/infiniband/DEVICE/ports/1/{gids/3,
gid_attrs/types/3,gid_attrs/ndevs/3}` on both Sparks:

| Rank/host | Device | Raw GID3 | Type / netdev |
|---|---|---|---|
| 0/sparkhead | rocep1s0f1 | 0000:0000:0000:0000:0000:ffff:0aca:1a02 | RoCE v2 / enp1s0f1np1 |
| 0/sparkhead | roceP2p1s0f1 | 0000:0000:0000:0000:0000:ffff:0aca:1b02 | RoCE v2 / enP2p1s0f1np1 |
| 1/sparkworker-wired | rocep1s0f1 | 0000:0000:0000:0000:0000:ffff:0aca:1a01 | RoCE v2 / enp1s0f1np1 |
| 1/sparkworker-wired | roceP2p1s0f1 | 0000:0000:0000:0000:0000:ffff:0aca:1b01 | RoCE v2 / enP2p1s0f1np1 |

The existing IPv6Address.ipv4_mapped conversion correctly yields 10.202.26.2 /
27.2 and 26.1 / 27.1. Device-name case matters. Tests now reconstruct this real
sysfs tree and retain both-rank rank/fabric/GID/type refusal cases. Discovery of
all functions also found unused roceP2p1s0f0 GID3 zero with EINVAL attributes;
that exploratory output was not valid structured evidence. Captured fixtures use
separate, successful `set -e` reads of only the two required devices.

## Remaining remote contracts (source audit, not live execution)

| UP3 path | Producer / consumed format | Audit disposition |
|---|---|---|
| preflight ledger/checkouts/pins | Shell exit status; ordered `sha256sum` lines | Exact reviewed source checks; no speculative stdout regex or service call in this audit. |
| host_check | `HOST_PARITY`: uname line, driver CSV line, anchored `MemAvailable: N kB` | Correct source-command ordering and proc format; GPU/kernel/memory values not queried under this read-only authorization. |
| model/health and worker | Existing JSON contracts, `X86HOST_PROBE`, `pf.worker` | Source/CPU-reviewed only; no port requests or X86host access authorized. No changes. |
| source/Engram/proxy stats | Existing emitted JSON, source hashes and `record/number/rows` checks | Source/CPU-reviewed only; model metadata and /proc not read remotely this round. No changes. |
| owned HOLD and close receipts | Shared admission/scoped_remote constants; timestamp+run marker | Shared contracts unchanged; not executed. |
| quiet/restore/W8 | Shared quiet marker, exact six distinct IDENTICAL 0..5 and pinned W8 markers | Source/CPU-reviewed only; no W8 or live restore. |
| greedy/cell/finish artifacts | Local drivers' JSON, full counter baseline and paired decision | Not a remote text parser; unchanged and outside Round 6 edit scope. |

The `/proc` TID/starttime/affinity parser uses stat field 22 (index 19 after the
closing comm parenthesis), stable-census checks and sched_getaffinity; tested
locally only. This audit does not claim actual pinned-worker/proxy census or
native RDMA/Engram thread attribution. Those remain experiment qualification.

The later item-3 production probes observed one complete **single-ring**
initialization message per rank on the same container IDs (initial log snapshots
above predated these messages). Both matched the new full-body regex, including
256 KB maximum, 3072 KB slots and 3072 KB all-gathers. Dual-ring and pin evidence
remain absent. The lane generated no requests to cause initialization; Docker
logs merely read the running production containers' history.

After final output-label refinement, a repeat first failed inspect during an
external container replacement. A later read found new IDs
`f3068c968fe664b629963890032af5228c6b11b00cdf34a3d7126cb994eb9bd1`
and `3e06c3bcc29cac93e10115c7d3e9fa146b315988ec2121533f3f279ed338217f`.
Both final production-probe attempts refused their actual `/serve.sh` port8001,
before logs/sysfs collection. No experiment result or new production pass is
claimed. The prior port8000 probes remain the observed production plumbing runs;
the final label-only changes are tested against those captured shapes.


## Round 8 read-only production overlay-absence check

The probe adds only cat of the fixed installed rdma.py path and consumes actual
Docker Mounts Destination fields. It requires the original image SHA256
`26e95c300a79aa1c33ceb759426343c06c1f2ce133b5f0bd4c673d1a158c3866`
and no mount covering rdma.py, up3_rdma_pin.py or /run/up3-rdma-pin.
`production-rdma-absence.json` projects only captured IDs/hash/destinations;
its provenance explicitly distinguishes this capture from the older
launcher/process shapes used in the fake transport. Docker returns Mounts in
varying order; complete before/after mount objects match when sorted by
Destination. No fields were discarded for that comparison.

BOTH final probes passed on production8000 with stable IDs and empty errors:
rank0 launcher1/67threads,engine171/73,worker243/638,proxy461/21;
rank1 launcher1/79,worker186/635. Both installed-source hashes match the
original and no relevant overlay/receipt mount exists. No receipt, Python
interpreter, affinity setter, port request, GPU or container/process change
was used. Probe receipts remain NOT QUALIFIED. This observes installed bytes
and mount configuration; it is not a CPU-pin or native GB10 qualification.
