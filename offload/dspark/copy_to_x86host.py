"""Run on x86host: capped direct-link O_DIRECT copy, exact extraction, receipts.

Only standard-library Python is needed; no CUDA or tensor library is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import NotRequired, TypedDict


class TensorRow(TypedDict):
    name: str
    shape: list[int]
    dtype: str
    resident_packed_bytes: int
    source_byte_range: list[int]
    source_file: NotRequired[str]
    representation: NotRequired[str]
    exl3_k: NotRequired[int | None]
    source_file_sha256: NotRequired[str]
    baseline_identical: NotRequired[bool]
    identity_evidence: NotRequired[dict[str, object]]
    provenance: NotRequired[dict[str, object]]
    source_byte_range_sha256: NotRequired[str]
    destination_file: NotRequired[str]
    destination_byte_range: NotRequired[list[int]]

ROOT = Path("/opt/dsv41/x86host-data")
RATE = 1_000_000_000  # decimal bytes/s; stricter than pv -L 1g
METRICS = "http://HEAD_IP:8000/metrics"
SSH = ["ssh", "-b", "10.202.28.1", "-c", "aes128-gcm@openssh.com",
       "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
       "-o", "HostKeyAlias=HEAD_IP", "rank0user@10.202.28.2"]
REMOTE = r'''
import json,subprocess,sys,time
path=sys.argv[1];offset=int(sys.argv[2])
minimum=2**63
def available():
    return int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
if available()<4*1024**3:
    raise RuntimeError('MemAvailable below 4 GiB before dd')
p=subprocess.Popen(['dd','if='+path,'iflag=direct,skip_bytes','skip='+str(offset),'bs=64M','status=none'])
while True:
    value=available()
    if value<minimum:
        minimum=value
        print('S2_SOURCE_MEMORY_SAMPLE '+json.dumps({'path':path,'min_mem_available_bytes':minimum}),file=sys.stderr,flush=True)
    if value<4*1024**3:
        p.terminate();p.wait()
        raise RuntimeError('MemAvailable below 4 GiB; aborted dd: '+str(value))
    code=p.poll()
    if code is not None:break
    time.sleep(0.1)
print('S2_SOURCE_MEMORY '+json.dumps({'path':path,'min_mem_available_bytes':minimum}),file=sys.stderr,flush=True)
sys.exit(code)
'''


def idle() -> None:
    with urllib.request.urlopen(METRICS, timeout=3) as response:
        body = response.read().decode()
    for name in ("running", "waiting"):
        values = re.findall(r"^vllm:num_requests_"+name+r"(?:\{[^\n]*\})?\s+([\d.eE+-]+)$", body, re.M)
        if not values or any(float(v) != 0 for v in values):
            raise CopyInterrupted(f"serve is not idle: {name}={values}")


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CopyInterrupted(RuntimeError):
    """A production-traffic guard interrupted this lane's transfer."""


def copy(source: str, destination: Path, expected: str | None, size: int | None,
         *, resume: bool = False) -> dict[str, object]:
    if destination.exists():
        actual = sha_file(destination)
        if expected is None or actual != expected or (size is not None and destination.stat().st_size != size):
            raise RuntimeError(f"refusing unverified existing destination: {destination}")
        return {"file": str(destination), "size": destination.stat().st_size, "sha256": actual, "reused_verified": True}
    idle()
    if int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) < 16*1024**2:
        raise RuntimeError("x86host has less than 16 GiB available")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+".s2-partial")
    if partial.exists() and not resume:
        raise RuntimeError(f"partial transfer requires inspection before retry: {partial}")
    start = time.monotonic()
    started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    resume_from = partial.stat().st_size if partial.exists() else 0
    if resume_from and (expected is None or size is None or resume_from > size):
        raise RuntimeError("resume requires a known whole-file hash and size, with no oversized partial")
    if resume_from and resume_from == size:
        actual = sha_file(partial)
        if actual != expected:
            raise RuntimeError(f"verification failed for complete checkpoint: {partial}")
        partial.rename(destination)
        return {"file": str(destination), "source": source, "size": size, "sha256": actual,
                "resumed_from_bytes": resume_from, "bytes_transferred": 0, "complete_checkpoint_verified": True}
    digest = hashlib.sha256()
    # The new read starts on a 4 KiB boundary; discard any already-saved overlap.
    aligned_offset = resume_from // 4096 * 4096
    command = "python3 -c "+shlex.quote(REMOTE)+" "+shlex.quote(source)+" "+str(aligned_offset)
    idle()  # Recheck immediately before NIC traffic.
    transfer_start = time.monotonic()
    process = subprocess.Popen(SSH+[command], stdout=subprocess.PIPE)
    stop = threading.Event()
    errors: list[str] = []

    def monitor() -> None:
        while not stop.wait(0.5):
            try:
                idle()
            except Exception as error:
                errors.append(str(error))
                process.terminate()
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    count = resume_from
    try:
        assert process.stdout is not None
        overlap = resume_from-aligned_offset
        if len(process.stdout.read(overlap)) != overlap:
            raise EOFError("short source read while discarding checkpoint overlap")
        with partial.open("ab" if resume_from else "xb") as output:
            while chunk := process.stdout.read(1024*1024):
                # Pace each receive before consuming the next chunk. SSH/pipe buffers
                # bound any read-ahead; the source uses a single 64 MiB dd block.
                delay = (count-resume_from+len(chunk))/RATE - (time.monotonic()-transfer_start)
                if delay > 0:
                    time.sleep(delay)
                output.write(chunk)
                if not resume_from:
                    digest.update(chunk)
                count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        code = process.wait()
        if code or errors:
            message = f"copy aborted: ssh={code}, guards={errors}, bytes={count}"
            if errors:
                raise CopyInterrupted(message)
            raise RuntimeError(message)
        # A resumed prefix is untrusted until the complete on-disk file matches
        # the authoritative manifest. Never infer identity from its size alone.
        actual = sha_file(partial) if resume_from else digest.hexdigest()
        if (expected is not None and actual != expected) or (size is not None and count != size):
            raise RuntimeError(f"verification failed: {source}, size={count}/{size}, sha256={actual}/{expected}")
        # Independently re-read destination, rather than only trusting the stream hash.
        if not resume_from and sha_file(partial) != actual:
            raise RuntimeError(f"destination disk hash mismatch: {partial}")
        partial.rename(destination)
        elapsed = time.monotonic()-start
        receipt = {"file": str(destination), "source": source, "size": count, "sha256": actual,
                   "started_utc": started_utc, "finished_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   "elapsed_seconds_including_verification": elapsed, "bytes_per_second": (count-resume_from)/elapsed,
                   "resumed_from_bytes": resume_from, "bytes_transferred": count-resume_from,
                   "rate_cap_bytes_per_second": RATE, "transport": "10.202.28.2 -> 10.202.28.1 SSH aes128-gcm", "read": "dd iflag=direct bs=64M"}
        print(json.dumps(receipt), flush=True)
        return receipt
    finally:
        stop.set()
        watcher.join()
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


def extract(source: Path, destination: Path, rows: list[TensorRow]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for row in rows:
        length = row["resident_packed_bytes"]
        start, end = row["source_byte_range"]
        if start < 0 or end-start != length or length < 0 or row["name"] in header:
            raise ValueError(f"invalid source range, length or duplicate name: {row['name']}")
        header[str(row["name"])] = {"shape": row["shape"], "dtype": row["dtype"], "data_offsets": [offset, offset+length]}
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = destination.with_suffix(".s2-partial")
    with source.open("rb") as incoming, partial.open("xb") as outgoing:
        outgoing.write(struct.pack("<Q", len(encoded))+encoded)
        for row in rows:
            start, end = row["source_byte_range"]
            incoming.seek(start)
            remaining = end-start
            digest = hashlib.sha256()
            destination_start = outgoing.tell()
            while remaining:
                data = incoming.read(min(8*1024*1024, remaining))
                if not data:
                    raise EOFError(str(source))
                digest.update(data)
                outgoing.write(data)
                remaining -= len(data)
            row["source_byte_range_sha256"] = digest.hexdigest()
            row["destination_file"] = destination.name
            row["destination_byte_range"] = [destination_start, outgoing.tell()]
        outgoing.flush()
        os.fsync(outgoing.fileno())
    # Verify each output range independently after serialization.
    with partial.open("rb") as stream:
        for row in rows:
            start, end = row["destination_byte_range"]
            stream.seek(start)
            digest = hashlib.sha256()
            remaining = end-start
            while remaining:
                chunk = stream.read(min(8*1024*1024, remaining))
                if not chunk:
                    raise EOFError(str(partial))
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != row["source_byte_range_sha256"]:
                raise RuntimeError(f"extraction hash mismatch: {row['name']}")
    partial.rename(destination)


def main() -> None:
    inventory = json.loads(Path(sys.argv[1]).read_text())
    source = json.loads(Path(sys.argv[2]).read_text())
    drafter = ROOT / "dspark"
    receipts: list[dict[str, object]] = []
    files = sorted({row["source_file"] for row in inventory["tensors"]})
    for filename in files:
        entry = source["manifest"][filename]
        if not source["files"][filename]["baseline_same_inode"]:
            raise RuntimeError(f"baseline provenance needs legal-tree hash: {filename}")
        incoming = drafter / "source" / filename
        receipts.append(copy(source["source_root"]+"/"+filename, incoming, entry["sha256"], entry["size"]))
        rows = [row for row in inventory["tensors"] if row["source_file"] == filename]
        extract(incoming, drafter / filename, rows)
    for filename in ("config.json", "exl3_moe_plan.json", "tokenizer.json", "tokenizer_config.json", "CANDIDATE_MANIFEST.json"):
        receipts.append(copy(source["source_root"]+"/"+filename, drafter / filename, source["metadata_sha256"][filename], None))
    cache_a = Path(sys.argv[3])
    cache_b = Path(sys.argv[4])
    receipts.append(copy("/home/rank0user/.cache/dsv41-serve/markov_top_tokens.json", drafter / "markov_top_tokens.json", sha_file(cache_a), cache_a.stat().st_size))
    if cache_a.read_bytes() != cache_b.read_bytes() or (drafter / "markov_top_tokens.json").read_bytes() != cache_a.read_bytes():
        raise RuntimeError("Markov token caches differ")
    (drafter / "TENSORS.json").write_text(json.dumps(inventory, indent=2)+"\n")
    index = {"metadata": {"total_size": inventory["resident_packed_bytes"]},
             "weight_map": {row["name"]: row["destination_file"] for row in inventory["tensors"]}}
    (drafter / "model.safetensors.index.json").write_text(json.dumps(index, indent=2)+"\n")
    done = {"files": receipts, "extracted_files": [{"file": f, "sha256": sha_file(drafter/f), "size": (drafter/f).stat().st_size} for f in files],
            "tensors": len(inventory["tensors"]), "markov_caches_byte_identical": True}
    (drafter / "DSPARK_COPY_DONE").write_text(json.dumps(done, indent=2)+"\n")
    print("DSPARK_COPY_DONE", flush=True)
    engram_receipts = []
    for filename in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        entry = source["manifest"][filename]
        engram_receipts.append(copy(source["source_root"]+"/"+filename, ROOT/"engram"/filename, entry["sha256"], entry["size"]))
    engram_receipts.append(copy(source["source_root"]+"/model.safetensors.index.json", ROOT/"engram"/"model.safetensors.index.json", source["metadata_sha256"]["model.safetensors.index.json"], None))
    (ROOT/"engram"/"ENGRAM_COPY_DONE").write_text(json.dumps({"files": engram_receipts}, indent=2)+"\n")
    print("ENGRAM_COPY_DONE", flush=True)


if __name__ == "__main__":
    main()
