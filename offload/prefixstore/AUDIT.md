# RP1 Stage A conflict and safety audit

Audited against branch base 6b99476 (K1 tip 3472d3b through b31c8df), image v5 source snapshot, and donor
HeNryous afe4d5e41ec6fa4db17c5cc88758af439a106913. This is a CPU/source audit; GB10 certification remains required.

| Component | Shared mounted files | Interaction / disposition |
|---|---:|---|
| K1's 13-file overlay | 0 | RP1's 7 files are additive. Build rebuilds and verifies all K1 bytes first at fuzz 0, then applies 0056/0058/0077/0081 at fuzz 0. Both manifests and all 20 mounted bytes are verified. K1's full regression suite stays 27/27. |
| Remote DSpark `spec_decode/__init__.py`, `remote_dspark/` | 0 | Scheduler/worker connector metadata executes alongside remote drafting. Capacity mode has no draft KV group. RP1 never changes proposal/verification logic, taps, accepted lengths or remote-worker state. Restart identity at 24K/75K/150K, concurrent loads, and mixed decode remain target-side gates. |
| Engram `sitecustomize.py` / `engram.py` / JIT extension | 0 | Engram tables and RPC identity are unchanged. Prefix loads copy KV, then any remaining computed suffix follows existing Engram lookup. NIC/CPU/memory contention is a measurement gate. The RAM prototype never mounts into the engine. |
| Async process proxy `remote_dspark/proxy_proc.py` | 0 | Proxy is torch-free and separate from connector gloo group. No proxy ABI, file, signal or lifetime changes. Gloo synchronization happens before GPU submit_load; decode gating must measure that serialization. |
| Experiment launcher | 1 line (`serve_exp.sh`) | Existing `EXTRA` is extended with optional `EXP_EXTRA_ARGS`; when unset output is unchanged. Existing start_exp transports config exports. Test executes only pre-JIT portion and checks connector JSON reaches EXTRA; no launcher is run. Production serve.sh remains untouched. |
| v5 allocator / cache manager / scheduler | 0 direct edits | Read source to trace prefix-cacheable-group IDs, whole packed block stride and deferred external hits. OffloadingConnector scheduler receives an aligned per-request cap; ordinary KV allocation and prefix hashing are unchanged. |

## Exact donor rebuild

The four patches modify six source files; local `0100-rp1-safety.patch` also includes the filesystem manager,
bringing RP1 to seven. `MANIFEST.json` records every patch SHA256, base and output file SHA256 and the exact K1
manifest SHA256. `build_overlay.py` checks clean donor HEAD, each donor hash, `patch --batch --fuzz=0`, compile,
then byte identity. Upstream prose/launchers are never executed. Local patch is itself recorded in the manifest;
`--write` explicitly regenerates it, ordinary verification refuses drift.

0058 exempts only `OffloadingConnector` from expandable_segments incompatibility. It registers host buffers and
uses virtual GPU addresses at copy time. It does not authorize GPUDirect registration of movable CUDA allocations.
The gate test proves NIXL remains rejected with expandable segments.

0056 verifies every successful block, including the successful prefix of a partial I/O failure, on Python and native
wrapper paths. Missing/unreadable/bad sidecars delete the block and make it a miss. Native wrapper is tested using a
fake C loader; the installed GB10 extension and O_DIRECT alignment still require live verification. Sidecar-before-data
atomic rename is a process-restart mechanism; durability across an unclean host stop (stores do not fsync) and concurrent
cross-engine writers are not certified. CRC detects accidental bytes changing, not a correctly checksummed chunk filed
under a false identity.

0077 shares only eligible one-head attention groups with TP-only topology and copies rank 0's CPU chunks into the
other node before submit_load. RP1 refuses any multinode setup that cannot select this layout (upstream only warned
and retained the unsafe layout). One head is a structural gate, not numerical proof: compare independently stored
rank bytes with MN_VERIFY before enabling single-writer mode. The packed transfer includes full backing-stride bytes,
so distinguish unused padding and inactive aliased/window regions from the selected valid group payload during the
comparison. A mismatch in selected live payload is a hard stop. Rank 1's stale/mismatched CPU payload must be replaced
before its GPU transfer, and its prior contents must not affect answers.

0081's request cap mitigates eviction of promoted but unpinned CPU chunks. Local hardening uses the LCM of all chunk
token counts, validates finite fraction in [0,1], and clamps an explicit request max_load_tokens down to the safety cap.
Generated configs allow only <=0.2 with MAX_SEQS=4. This leaves nominal reserve; it is not a proof against every
scheduler/promotion race. Four concurrent file reloads and bounded completion are mandatory GB10 gates.

## Namespace and lifecycle

FileMapper already includes model, cache dtype, token-hash size, groups/layer order, topology, and replicated-layout
identity in its namespace. A mutable `/srv/dsv41/legal-v2a` path is insufficient to distinguish different weights.
RP1 requires `prefix_store_identity.checkpoint_sha256` and `layout_sha256` in multinode mode. Both lowercase 64-hex
digests are appended to the offloading model identity before FileMapper hashes it. The generated layout identity pins
all 20 mounted file hashes; FileMapper additionally pins runtime groups, block sizes, dtype and topology. Caller must
supply the exact verified checkpoint-manifest digest. A changed checkpoint manifest, overlay/layout or topology gets
a separate namespace. Never falsify this identity to reuse old files. Prefix salt/hash remains the engine's per-matter
isolation boundary. File sidecars are not authentication; local filesystem writers are trusted in Stage A.

CPU-region engine IDs change on restart, while disk content keys and salts intentionally persist. TieringSpec has no
unlink barrier, so old /dev/shm/vllm_offload_<engine_id>.mmap files can remain. Remove only paths tied to the stopped
experiment's recorded engine ID, after checking no process still maps them; never glob-delete shared memory. Never
reuse a live engine ID or count leaked tmpfs as free memory. A complete shutdown/empty CPU region is required to prove
a file hit rather than a surviving CPU/GPU hit.

RP1 refuses buffered filesystem fallback in multinode mode. This avoids a large page-cache population on the Sparks,
which would invalidate the 5 GiB MemAvailable requirement. The file tier has no configured disk quota/automatic
file eviction in this overlay: coordinator must use an experiment-specific namespace, monitor disk space, and expire
only inactive namespaces offline. Bounded operational disk eviction is a production integration requirement.

Gloo link/peer failure is not transparent compute fallback in Stage A: the existing collective can raise/hang until
its process-group timeout. Abort the experiment on the 300-second request watchdog and restore the known config.
Do not treat a half-loaded distributed prefix as usable. Stage B provides bounded per-RPC timeout/compute fallback
only in its fake client prototype; applying it inside the engine requires completion/invalidation tests.

## MC1 rebase and composition audit (2026-10-27)

RP1's seven commits (6b99476..9a5967b) were rebased onto production **596ba1c** (`prod-k1-327-ops1-20261026`) without
conflicts. Production's K1 manifest is the one RP1 pinned (559c5867…), so the v5 -> K1 -> RP1 fuzz-0 rebuild is
byte-identical; RP1 20/20 and K1 27/27 pass on the rebased tree. `compose_audit.py` re-runs the file and import audit
from git objects (no checkout), so the coordinator can repeat it against the refs current at window time:

```bash
python3 offload/prefixstore/compose_audit.py --json <campaign dir>/MC1-compose.json   # exit 1 on any shared mount
```

| Package (ref audited) | Shared mounted paths | Import adjacency | Interaction and disposition |
|---|---:|---|---|
| K1 packing, 13 files (596ba1c) | 0 | connector config/scheduler/worker and `config/vllm.py` import `kv_cache_interface` / `kv_cache_utils` | Read-only use of spec classes. K1 packing defines the packed stride the file tier stores; the layout digest covers K1's bytes. |
| remote DSpark, async proxy, spec_decode (596ba1c configs) | 0 | none | Proxy is a torch-free process; the connector's gloo group is separate. Engram loads through PYTHONPATH/sitecustomize, outside the vllm package. |
| 0075 replay skip, UP2-ADOPT (offload/up2-adopt 3e29dbc) | 0 | same kv_cache_interface/kv_cache_utils imports as K1 (0075's overlay re-ships those K1 paths) | The core scheduler (0075) talks to the connector through vLLM's interface, not imports. 0075 already has connector-aware paths: it waits on an in-flight connector fence (`replay_recovery_wait`) and re-runs the bounded replay after an async restore completes. 0075 was qualified without a connector (its CPU test fakes the fence), so M2 must run on 0075 if 0075 is production by then. Its `replay_guard` refuses raw Docker mounts: MC1 adds none (file tier, exclusion hashes and audit live under the launcher's existing `/root/.cache` mount). |
| 0091 upstream fixes, R-UPF (offload/upfix c1e711c) | 0 | `tiering/fs/manager.py` (MC1) imports `tiering/async_lookup.py` (0091) | #59096 changes `mark_miss`, the failed-promotion path every CRC/short-read miss takes: it no longer forces a newer PENDING/IN_FLIGHT probe to RESOLVED. That removes a phase-assert hazard on exactly MC1's negative tests. #59504 zeroes only the groups the connector loads, taken from `get_loaded_kv_cache_group_ids` or else `prefix_cacheable_group_ids`; the offloading connector loads `get_offloading_group_ids` = `prefix_cacheable_group_ids` (no hisparse), the same set, so it composes and stops stale SWA bytes in loaded ranges. #59880 changes spec-decode slot reservation only. 0075 and 0091 both mount the core scheduler; R-UPF ships a composed `overlay-0075` variant. MC1 needs neither. |
| PP4 sampled K2 (offload/pp4-promote 8548a7c) | 0 | none | Replaces K1's `model_runner.py` with K1 + a verify-mask block + `@execution`/`@sampling` timing decorators (`pp4_prepare.py`, PP3 `prepare_verifier.py`); the KV-connector hook calls are unchanged. Adds spec-decode utilities, a round trace and its own remote_dspark. |
| clock fold 2300 (prod/clock-fold-2300 420be24) | 0 | none | `dsv41.env` only; experiment configs source it and inherit `GPU_CLOCK_CAP`. |

**Layout digest (MC1 change).** The file-tier namespace used to pin only the K1 and RP1 manifests. A composed config
(e.g. MC1 + 0075 + PP4) would then have reused an old namespace across a promotion that changed mounted engine
bytes. `make_configs.layout_sha` now hashes every file every rendered `OVERLAYS` pair mounts (directories expanded:
remote DSpark, async proxy, spec_decode, K1, RP1/MC1). Any promotion that changes a mounted file therefore starts a
cold namespace, as F-MATTER §3.2 and §5 require. A composed arm must be rendered by a generator that calls
`layout_sha` on its final `OVERLAYS`; hand-editing a config's layout digest is never acceptable.
