# MC1 matter cache, Tier 1 (Spark NVMe): what is built and how to stage it

Plan: `documents/plans/lanes/F-MATTER.plan.md` §3; brief `MC1-MATTER-TIER1.md`. CPU-built and CPU-tested only;
nothing here has run on a Spark. The GB10 runs (M1 smoke, M2 window) belong to MC2 and the coordinator.

## The matter tier and its gate (item 2)

The only persistent tier in the generated configs is `Dsv41MatterTier` (`overlay/.../tiering/fs/dsv41_matter_tier.py`),
an out-of-tree secondary tier loaded through vLLM's `module_path`. Its gate is in `dsv41_matter.py` (standard library).

* **Default deny.** A request is written or read only when `kv_transfer_params` carries `matter_ns` (64 lowercase hex)
  and `persist: true` (the exact JSON boolean). Untagged traffic, `persist: false`, a string `"true"`, a malformed
  namespace: never persisted and never read, refused before any disk access. In-memory reuse is unchanged: such a
  request's chunks still land in the CPU tier and a later request with the same prefix loads them from there
  (`cases_reuse.py`: untagged, `persist: false` and excluded requests through the real connector, 0 bytes on disk).
* **Excluded namespaces.** The tier loads the hash-only exclusion file (one 64-hex line each, sorted, unique, LF),
  pinned by sha256 in the config. A request whose `matter_ns` is in it is refused for PUT and GET and logged once.
* **Fail closed.** Missing, symlinked, non-regular, group/other-readable, oversized, malformed or sha-mismatched
  exclusion file, or an audit log that cannot be opened or written: the gate is CLOSED and no request persists or
  reads. A later change of the exclusion file (any stat identity change) closes it until restart.
* **Unscoped tiers refused.** Patch 0101 makes the upstream `fs` tier refuse to start in multinode mode, and patch
  0105 makes the connector scheduler refuse, in multinode mode, ANY secondary tier that does not declare the
  matter-gated contract (below), so no config can add a tier that persists untagged requests.
* **Isolation.** Layout `<root>/ns/<matter_ns>/<FileMapper namespace>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin`
  (+ `.crc`, 0056). The FileMapper namespace keeps RP1's checkpoint and layout digests; the layout digest covers
  every mounted overlay file. The async lookup cache is keyed by (namespace, key): two matters never share a file
  or a lookup verdict, even with identical prompts. Root and namespace directories are 0700; one engine per root
  (flock).
* **Audit.** Append-only JSONL, 0600: `{ts, rank, op, ns, chunks, tokens, bytes, reason}` with `op` in `gate`,
  `put`, `get-hit`, `get-miss`, `refuse`, `purge`, `purge-sweep`. No names, no request ids, no content.
* **Purge by namespace.** `request_purge(ns|ALL)` refuses the namespace at once (tombstone), deletes its subtree in the
  janitor thread, and sweeps again when the tombstone expires (10 min default) to catch stores that were in flight.
  While an engine holds the root, a purge is requested by creating `<root>/purge-requests/<ns>.req` (or `ALL.req`).
* **Audit ops** also include `evict` (reason `ns-cap`/`tier-cap`) and `expire`.

## Bounds, TTL and purge (item 6)

* **Byte bounds.** `max_bytes` (tier) and `max_ns_bytes` (namespace) are hard admission limits: a store that would
  exceed either is refused (audited once per request) and the janitor is woken. Above 0.9 x a bound the janitor evicts
  to 0.8 x it: oldest files first inside a namespace, least recently used namespaces first for the tier (namespace
  recency = last put or hit). Usage is exact after each scan and an upper bound in between, so a bound is never
  exceeded by under-counting. Config `PREFIX_MATTER_MAX_GB=8`, `PREFIX_MATTER_NS_MAX_GB=4` (defaults sized for the
  window; set them from the measured free space before production).
* **TTL.** Files unused for `ttl_days` (`PREFIX_MATTER_TTL_DAYS=30`, F-MATTER's proposal for NDM to confirm) expire at
  start and every 6 h; a load refreshes its files' mtime, so mtime is last use. Stale temp files go too; a namespace
  left without blocks is removed.
* **At start.** The janitor scans the store before any store is admitted (`scan-pending` until then), purges every
  namespace that is now in the exclusion set (audit `purge`, reason `excluded`), expires and enforces the bounds.
* **Janitor.** One thread owns every deletion and also polls purge requests, expires tombstones and re-checks the
  exclusion file every 30 s, so an idle engine still honours a purge or a changed exclusion file.
* **Purge CLI** (`matter/purge.py`, on the Spark, with sudo): `list`, `purge --ns HEX|--all --audit A`, `expire
  --ttl-days N --audit A`. With no engine holding the root it deletes at once and audits (rank -1, `by: cli`); while
  an engine runs, `purge` files `<root>/purge-requests/<ns>.req` for the engine (which refuses the namespace at once)
  and `expire` refuses. Map a matter to its namespace on Controller with `matter/ns.py`; the CLI takes namespaces only.

Tier 1 under RP1's certified replicated layout lives on **rank 0's NVMe only** (sparkhead): the scheduler process owns
the tiers and rank 1 receives chunks by the 0077 gloo copy. sparkworker's disk is not used.

## Streamed landing (item 4)

Without it, a persistent-tier hit counts only once every chunk is promoted into the CPU tier, so the whole prefix
must be resident at once, and 0081 caps a request at 0.2 of the tier: 29,568 tokens at 256 MiB. `dsv41_stream.py` +
patch 0102 remove the cap without that residency:

* For a request a matter tier serves (gate ALLOW), a persistent-tier chunk is a lookup HIT where it lies (no
  promotion), so the hit covers the whole stored prefix and the scheduler allocates GPU blocks for all of it.
* The load is split into windows of at most `window` CPU chunks, `window` = half the per-request budget
  `min(PREFIX_STREAM_MB, load fraction x CPU tier)`. Each window is promoted (tier -> registered CPU region), pinned,
  and issued as an ordinary load job: the donor-proven 0077 rank-0 -> rank-1 gloo copy and CPU -> GPU handler, unchanged.
  The next window promotes while one lands, so at most two windows per request are resident (default 256 MiB tier,
  0.2: windows of 192 chunks, ~26 MiB, ~14.7K tokens; a 150K prefix lands in ~11 windows, ESTIMATED).
* Only the final window's GPU spec lets the workers release the request. If a window cannot land (a chunk evicted,
  purged or CRC-failed, or not resident within 60 s) or the request is aborted, one `fail` job makes every worker
  report the request finished and failed (`failed_recving`): the scheduler frees its blocks and recomputes. Multi-group
  layouts cannot report block-level errors, so a streamed prefix is never reported partially valid. After the final
  window is out, no fail job follows (no double release).
* A new GPU data path was deliberately not written: it could not be verified without a GPU, while this design changes
  only scheduler orchestration and the worker's completion reporting. CPU tests prove (a) the windows partition the
  single load exactly, by executing the real handler's descriptor code on both and comparing every (src, dst, size)
  copy; (b) the real connector scheduler/worker around a real tiering manager + matter tier land a 77-chunk prefix
  through a 32-chunk CPU tier byte-exactly into every GPU block of both groups, where the unstreamed path is capped at
  384 tokens; (c) corruption mid-stream and abort mid-stream each end in exactly one release and no leaked state.
* Knobs: `PREFIX_STREAM=1` (default) and `PREFIX_STREAM_MB=64`; env `VLLM_DSV41_OFFLOAD_STREAM[_MB]` in the container.

## Tier-agnostic connector: how Tier 2 plugs in

Nothing in the connector patches (0102-0105) or in `dsv41_stream.py` names a tier type. The matter tag travels in the
request's `kv_transfer_params` to every secondary tier, and a tier takes part only through this contract
(documented in `dsv41_stream.py`):

* class attribute `dsv41_matter_gated = True`, and a `MatterGate` from `dsv41_matter.py` deciding every PUT and GET
  (standard library only, no file-tier dependency: default deny, exclusion hashes, fail closed, audit, tombstones);
* `allows_get(req_context) -> bool`, True only when the gate would serve the request (streaming eligibility);
* the SecondaryTierManager API with its usual meaning: a refused request gets `lookup` MISS and a failed `JobResult`
  with 0 bytes for any store or load; a failed load is a failed `JobResult` (the stream fails the request, which
  recomputes).

A later Tier 2 (x86host RAM/NVMe over ofl, RP1 Stage B) is therefore one more `secondary_tiers` entry with its own
`module_path`; the manager cascades stores to every tier and promotes from the first tier that hits, and the stream
lands from whichever tier holds the chunks. `cases_tier2.py` proves it with a RAM-backed stand-in tier
(`tests/vllm_cases/mc1_ram_tier.py`, no files) through the unchanged connector: alone, it stores, streams a 77-chunk
prefix through a 32-chunk CPU tier byte-exactly after a restart and refuses untagged/excluded requests; next to Tier 1
it serves a prefix Tier 1 lost; a tier without the contract is refused at start in multinode mode. Tier 2 still needs
its own byte bounds/TTL/purge and the M4 link-contention and load check (plan §3.10, §4).

## Per-block payload hash dump (item 5)

`PREFIX_HASH_DUMP=/root/.cache/<name>` (diagnostic, off by default; `VLLM_DSV41_OFFLOAD_HASH_DUMP`) makes every worker
append `<name>_r<rank>.jsonl` (0600): one line per GPU block it moves, `{ev, rank, job, g, lb, key, sub, sha, seg, n}`.
`store` lines hash a store job's GPU source blocks on EVERY rank, writer or not, after the step's KV writes complete
(the in-memory pool bytes); `load` lines hash a load job's (or stream window's) GPU destination blocks once the copy
completed (the restored bytes). The scheduler labels each job's GPU spec with its offload keys in CPU-chunk order, so
lines join across runs and ranks on (key, group, sub-block), never on request ids. `seg` holds 8-byte digests of
`PREFIX_HASH_SEGMENT`-byte segments (default 4096), which localize padding or inactive regions when ranks differ.
It extends `PREFIX_MN_VERIFY`: the rank comparison no longer needs rank 1 to store, because every rank hashes its own
GPU blocks; MN_VERIFY stays the CPU-region identity check. It synchronizes the device per store step: never in timing runs.

`blockhash_compare.py restore --in-memory A_r<r>.jsonl --restored B_r<r>.jsonl` is the M2 byte-equality gate per rank
(PASS iff every restored block equals its in-memory bytes, none missing or unlabeled). `blockhash_compare.py ranks
--rank0 A_r0.jsonl --rank1 A_r1.jsonl` is the rank-identity check, with a histogram of differing segments.

## Namespaces and the exclusion hashes (item 3, Controller)

`matter/dsv41_matter_client.py` holds the one definition of a matter namespace (`matter/ns.py` and
`derive_excluded_ns.py` import it): `matter_ns = HMAC-SHA256(K, b"dsv41-matter-ns-v1\0" + canonical(folder))`,
where `folder` is the last component of the client-folder string and `canonical` is NFKC, casefold, whitespace runs
collapsed, stripped. A casing, spacing or path difference between the exclusion list and a folder name therefore
cannot defeat an exclusion, and a matter keeps its namespace when it moves to `!Closed`.
`python3 matter/ns.py < names` prints digests only.

`matter/derive_excluded_ns.py` reads `/opt/dsv41/data/private/matter-exclusions.txt` (owner-only,
pinned sha256: the operator's own, not published) and `matter-ns-hmac.key` (owner-only, 32 bytes), and writes
`/opt/dsv41/data/private/matter-cache-excluded-ns.txt` (+ `.sha256`), 0600, atomically, hashes only, in the
exact format the gate accepts. It prints only the count and the paths, never a name or digest; errors cite line
numbers only.
`--add-synthetic-ns HEX` adds a synthetic test namespace for a window's exclusion test, and is refused for the
production output path. Generated 2026-10-27 from the pinned list (3 namespaces); re-derived in memory after the
client-helper refactor (last-component folder names) and found byte-identical. The pin for a start is read from
the private `.sha256` file at start time; it is derived from real names, so it is never copied into git or a report.

## Client helper for the harnesses (local-law, the drafting harness, any client)

The cache lives inside the DeepSeek server, below any harness, and the request API is harness-agnostic: the matter
namespace and the opt-in travel as data in the OpenAI-compatible request (`kv_transfer_params`, `cache_salt`), so
any client can set them and the server has no harness-specific code. Each harness vendors one file,
`matter/dsv41_matter_client.py` (standard library only; no vLLM, no torch), when its owner adds the integration (MC3;
DSV41 lanes do not edit the legal repos):

```python
tagger = MatterTagger.from_env()                 # DSV41_MATTER_CACHE=1, DSV41_MATTER_KEY_FILE, exclusion sources
body = tagger.tag(client_folder).apply(body)     # or extra_body=tagger.tag(client_folder).body_fields()
```

* It derives the namespace from the client-folder string with the HMAC key, and enforces the exclusion list BEFORE
  sending, from the private name list (by canonical name and by namespace) and/or the hash-only file the server
  loads, each optionally pinned by sha256. A changed source is reloaded (a matter NDM adds is honoured at once); an
  unreadable, mis-pinned or malformed source sends every request untagged until it reads cleanly (fail closed).
* A persisting tag adds `kv_transfer_params = {"matter_ns": ns, "persist": true}` and `cache_salt = ns` (the
  layout rule). Anything else (excluded, `DSV41_MATTER_CACHE` off, not requested, invalid folder, exclusions
  unavailable) adds nothing and strips a stale `matter_ns`/`persist`: the request is untagged, never persisted,
  and keeps normal in-memory reuse (`cases_reuse.py`). Reasons are fixed codes; no name, key or excluded namespace
  is ever logged, raised or returned.
* Switching DeepSeek on or off is the harness's model choice; `DSV41_MATTER_CACHE` separately switches persistence
  for that harness. Untagged requests behave exactly as before MC1.
* `python3 matter/dsv41_matter_client.py selfcheck --key K --exclusion-list L --excluded-ns F --client-files
  "<client folders>" "<client folders>/!Closed"` prints counts only: whether every listed matter is in the hash file,
  and how many list entries match an existing client folder name. Run it before enabling tagging: an entry that
  matches no folder (e.g. a different name order) would exclude nothing on either layer.
* Tests: `matter/tests/test_client.py` (synthetic names and keys): one definition with `ns.py`; folder forms; the
  persisting fields are what the server gate ALLOWs; exclusion through either source and the names layer alone;
  fail closed at start and at runtime with recovery; `apply` merge/strip; env switch; CLI counts only; stdlib only.

## Staging for a start (coordinator)

1. On Controller: `derive_excluded_ns.py` (item 3) writes the hash-only file and its `.sha256` (both private).
2. Copy it to `~/.cache/dsv41-serve/dsv41-matter/matter-cache-excluded-ns.txt` on sparkhead (and sparkworker), mode 0600,
   directory 0700. In the container that is `/root/.cache/dsv41-matter/matter-cache-excluded-ns.txt`.
3. Start with `PREFIX_MODEL_SHA256=<checkpoint manifest sha>` and
   `PREFIX_MATTER_EXCLUDED_SHA256=$(cut -c1-64 <the staged file>.sha256)`. The config
   refuses to render without either. Store root `/root/.cache/kv_matter_mc1`, audit `/root/.cache/dsv41-matter/audit.jsonl`
   (override with `PREFIX_MATTER_ROOT` / `PREFIX_MATTER_EXCLUDED` / `PREFIX_MATTER_AUDIT`, all under `/root/.cache/`).

## Checks

`bash offload/prefixstore/check_all.sh` runs the fuzz-0 rebuild, config generation, composition audit, K1 suite, all
CPU tests (including the real-vLLM cases under the kvpack-build venv) and strict typing.
