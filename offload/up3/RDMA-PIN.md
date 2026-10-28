# Round 8 CPU-pin poller identity

Preparation for independent review, not a GB10 result or window clearance.
Only CPU-pin mounts `vllm_exl3_moe/rdma.py` and `up3_rdma_pin.py`; both are
rebuilt and SHA256-pinned through the existing UP3 inputs/manifest/mount checks.
Control still has zero overlay files and the exact production environment.
The common lock and manifest input hashes change because a new arm input is
registered; K5, dual-ring, split and VOR replacement bytes do not change.

The read-only image source is `/opt/dsv41/data/src-image`:

* `vllm_exl3_moe/rdma.py:145-166`: `ring_for` builds a TP ring lazily, on the
  first eligible eager operation. Capture returns without constructing a ring.
* `rdma.py:105`: `e.start(self.h)` calls the native extension. The overlay
  brackets exactly this call, after peer linking, without issuing a collective
  or a request of its own. All numerical/communication code stays unchanged.
* `csrc/exl3_rdma.cu:566-570`: `start` immediately constructs
  `std::thread(proxy_loop, &g)`. Thread creation is synchronous; execution of
  `proxy_loop` and its `pthread_setaffinity_np` at `:266-272` is asynchronous.
  There is no further lazy native start. A two-second monotonic bound waits
  for the sole newly observed TID to have affinity exactly `{14}`. Additional
  births or loss of an existing TID refuse. This interval is startup only.

The helper requires the exact `EXL3_CPU_PIN=5-9,15-19` mask and CPU14. It
renames the new nonleader through `/proc/self/task/TID/comm` to
`exl3-rdma-prx` (13 ASCII bytes), rechecks name/affinity/starttimes/membership,
then exclusively writes and fsyncs a JSON receipt. Zero/multiple candidates,
timeout, failed rename, failed write or a second ring refuse loudly. The
overlay stops its own native ring on identity failure and rethrows through
`ring_for`; the image's normal exception-to-NCCL fallback cannot hide it.

Receipt location: `/run/up3-rdma-pin/PID-STARTTIME.json` **inside the fresh
experiment container**, with a private owner-only directory. This is not a
host bind mount, shared cache, production path or cross-start artifact.
Existing launcher mounts do not cover `/run`; each start already requires a
new container ID and exact mount reconciliation. No launcher mutation or
receiving-host directory creation is added. Docker removes this private state
with the experiment. Receipt fields: version, PID/process starttime,
TID/thread starttime, old/new comm, exact affinity, CLOCK_MONOTONIC timestamp
in nanoseconds. Starttimes bind the proof to current process/thread lifetimes.

Lazy case: if greedy checks and OPS1 do not initialize the ring, no receipt
exists. The host gate must refuse before candidate timing; it must never
create traffic or waive proof to make the arm run. The receipt proves identity
at startup; the direct stable census must independently confirm the current
name, TID/starttime and affinity before timing. It is not a claim about future
thread affinity or measured hardware performance.

CPU reproduction uses the private environment in `UP3-WINDOW.md`:

```bash
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider \
 offload/up3/tests/test_rdma_pin.py offload/up3/tests/test_up3_build.py \
 --basetemp="$S/pin-startup"
```

The Linux subprocess test starts only its own CPU thread, delays its real
self-pin, writes its real proc comm, verifies the durable receipt and exact
leader/poller affinities, and joins the thread. Native/model imports are
stubbed in the separate test executing the actual patched image `ring_for`;
that test verifies lazy initialization, refusal propagation and native stop.

The real Linux regression also checks `/proc/.../status` Name exactly. A
newline supplied to a proc comm write becomes part of the thread name and
appears escaped in status; stripping comm on read can conceal that defect.
The helper therefore writes exactly COMM, with no newline, and requires a
comm read of exactly COMM plus the kernel's one newline. The strengthened test
fails behaviorally against the initial writer and passes after the correction.


Production absence is explicitly reported by --probe-production: the fixed
installed rdma.py source hash must match the original image and inspect must
show no mounts covering either overlay module or the receipt directory. The
probe never reads experiment receipts or invokes this helper. Both real
production8000 probes passed with stable container IDs; the lane performed no
experiment startup or live qualification.
