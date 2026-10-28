# S2 exact checkpoint transfer

Ready markers on x86host are `/opt/dsv41/x86host-data/dspark/DSPARK_COPY_DONE`
and `/opt/dsv41/x86host-data/engram/ENGRAM_COPY_DONE`. A missing marker means
the corresponding package is not ready. Their JSON records contain file sizes,
SHA256 and timings. The Engram marker is written only after both whole shards
and the original index have been verified.

The drafter directory contains eight newly serialized safetensors files, with
the original tensor names, shapes and dtypes; its index references just the
4,755 inventoried dependencies. `TENSORS.json` includes each tensor's source and
destination byte range and SHA256 of the original packed range. The `source/`
subdirectory intentionally retains the eight hash-verified original source
shards as provenance (approximately another 5.38 GiB). Do not point a loader at
both directories. No EXL3 or FP8 conversion has been applied.

Configuration, EXL3 plan, tokenizer, candidate manifest and the live
`markov_top_tokens.json` accompany the tensors. The Markov cache was compared
byte-for-byte between both Sparks and against the delivered copy. Packed
payload size is 5,774,545,892 bytes; this excludes O2's runtime allocations.

The original source inventory and caches are retained in the lane worktree's
ignored `offload/dspark/.s2-artifacts/` directory on Controller and
`~/dsv41-offload/s2/` on x86host. No weights or caches belong in Git. Only progress
and the final report are lane-written files in the campaign directory.

## Reproduce

Run from this worktree on Controller, sourcing `~/.config/ndm/paths.sh` first:

```bash
mkdir -p offload/dspark/.s2-artifacts
ssh sparkhead python3 - < offload/dspark/inventory_source.py > offload/dspark/.s2-artifacts/S2.source-inventory.json
python3 offload/dspark/build_inventory.py offload/dspark/.s2-artifacts/S2.source-inventory.json offload/dspark
```

The Markov cache inputs must be fresh copies read from each Spark's
`~/.cache/dsv41-serve/markov_top_tokens.json`, named `S2.markov-a.json` and
`S2.markov-b.json`. Copy those two inputs, the inventory, source inventory and
`copy_to_x86host.py` to x86host's `~/dsv41-offload/s2/`. Execute on x86host in detached
tmux, with stdout/stderr captured in `~/dsv41-offload/s2/copy.log`:

```bash
python3 -u ~/dsv41-offload/s2/copy_to_x86host.py \
  ~/dsv41-offload/s2/TENSORS.json \
  ~/dsv41-offload/s2/S2.source-inventory.json \
  ~/dsv41-offload/s2/S2.markov-a.json \
  ~/dsv41-offload/s2/S2.markov-b.json
```

The copier requires the existing Spark host key under `HEAD_IP` and binds
SSH to x86host `10.202.28.1`, connecting to Spark A `10.202.28.2` with AES128-GCM.
Every source read uses `dd iflag=direct bs=64M`. Receive pacing is capped at
1,000,000,000 bytes/s; bounded pipe/SSH buffering and one dd block permit a small
bounded read-ahead. Production metrics are checked before each copy and every
0.5 seconds during it. A busy or unavailable metrics endpoint aborts the copy.
Spark MemAvailable is sampled every 0.1 seconds; below 4 GiB terminates the
copier's own dd process. X86host must have at least 16 GiB available before each
file. No listeners or CUDA processes are created.

The log's `S2_SOURCE_MEMORY` lines report the minimum sampled Spark memory per
file. Timings and rates include fsync and whole-file destination hashing;
fresh copies compare their streaming hash against a separate disk pass.
Source tensor-range hashes are computed from the whole-file-verified
source shards on x86host; every serialized destination range is independently
hashed before the drafter ready marker is written.

Verified existing source copies can be reused. Existing files whose expected
hash is unknown or wrong are refused. A `.s2-partial` is never presented as a
completed file. `copy(..., resume=True)` resumes an inspected partial with a
4 KiB aligned O_DIRECT read and discards the overlap already saved. It validates
the complete destination against the authoritative whole-file SHA256 before
publication, so a corrupt or stale prefix cannot pass. A complete checkpoint
is hashed and published without source traffic. Resumed-copy rates cover newly
transferred bytes and the final whole-file disk verification.

The actual campaign's production-request interruption occurred during shard 48;
`resume_engram.py` preserves shard 47's prior verification receipt and resumes
48 only in idle windows. It explicitly logs production/metrics interruptions
and waits 0.25 seconds before checking again; checksum, disk, source-memory and
other failures remain terminal. Run it in detached tmux on x86host:

```bash
python3 -u ~/dsv41-offload/s2/resume_engram.py \
  ~/dsv41-offload/s2/S2.source-inventory.json \
  ~/dsv41-offload/s2/copy.log
```

The production serve is never restarted by these tools.

## Checks

```bash
python3 -m unittest discover -s offload/dspark -p test_transfer.py -v
python3 -m mypy --strict offload/dspark/inventory_source.py offload/dspark/build_inventory.py offload/dspark/copy_to_x86host.py offload/dspark/resume_engram.py offload/dspark/test_transfer.py
```

The O2 loader owns runtime numerics and GPU fit validation. S2 validates packed
checkpoint identity, extraction and delivery only.
