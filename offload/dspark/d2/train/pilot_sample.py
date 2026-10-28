"""Amendment-4 fixed pilot sample. CPU only; never reads model outcomes.

The population is complete five-token rounds on the same fixed K2 trajectory
used by G2. Reading one token vector per sequence avoids materializing taps for
all non-sampled rounds. Keys and their hash stay in the private pilot plan.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import TypedDict
from collections.abc import Iterable

from safetensors import safe_open

from .data import D2Blocks, sha256
from .evaluate import match_length
from .replay import g1_drafts

# Historical analytical fixtures remain readable; real plans require DEV_POPULATION.
SPLITS = ("acap", "heldout_legal")
DEV_POPULATION = ("dev", "heldout_legal")


def population_splits(keys: Iterable[str]) -> tuple[str, ...]:
    if set(keys) == set(DEV_POPULATION):
        return DEV_POPULATION
    if set(keys) == set(SPLITS):
        return SPLITS
    raise ValueError("pilot population requires dev + heldout_legal (or historical analytical acap)")

SAMPLE_SEED = 20261024
SAMPLE_LIMIT = 1000


class RoundKey(TypedDict):
    request: str
    P: int


class Sample(TypedDict):
    schema: str
    seed: int
    limit: int
    manifest_sha256: str
    k2_sha256: dict[str, str]
    population_sha256: dict[str, str]
    population_rounds: dict[str, int]
    keys: dict[str, list[RoundKey]]
    sha256: str


def hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def key_hash(keys: list[RoundKey]) -> str:
    """Matches the scorer's ordered (request, position) population digest."""
    return hash_json([(k["request"], k["P"]) for k in keys])


def validate_sample(sample: Sample, manifest_hash: str) -> None:
    if (sample.get("schema") != "dsv41.d2.pilot-sample.v1" or sample.get("seed") != SAMPLE_SEED
            or sample.get("limit") != SAMPLE_LIMIT or sample.get("manifest_sha256") != manifest_hash):
        raise ValueError("pilot sample schema/seed/limit/manifest mismatch")
    splits = population_splits(sample["keys"])
    for mapping in (sample["keys"], sample["k2_sha256"], sample["population_sha256"], sample["population_rounds"]):
        if set(mapping) != set(splits):
            raise ValueError("pilot sample requires both held-out splits")
    for split in splits:
        keys = sample["keys"][split]
        count = sample["population_rounds"][split]
        if type(count) is not int or count <= 0 or len(keys) != min(SAMPLE_LIMIT, count):
            raise ValueError("pilot sample has the wrong complete-round count")
        if any(set(k) != {"request", "P"} or not isinstance(k["request"], str) or not k["request"]
               or type(k["P"]) is not int or k["P"] < 0 for k in keys):
            raise ValueError("invalid pilot sample key")
        if len({(k["request"], k["P"]) for k in keys}) != len(keys):
            raise ValueError("duplicate pilot sample key")
    if sample.get("sha256") != hash_json(sample["keys"]):
        raise ValueError("pilot sample key-list hash mismatch")


def complete_rounds(data: D2Blocks, k2: Path) -> list[tuple[int, RoundKey]]:
    drafts = g1_drafts(k2, data)
    result: list[tuple[int, RoundKey]] = []
    for seq_index, (path, seq, request) in enumerate(data.sequences):
        token_offset, length, gen = seq[:3]
        start = data.ends[seq_index-1] if seq_index else 0
        with safe_open(path, framework="pt", device="cpu") as f:
            tokens = f.get_slice("tokens")[token_offset:token_offset+length].tolist()
        pos = gen-1
        while pos+6 < length:
            truth = tokens[pos+2:pos+7]
            accepted = match_length(drafts[request][pos], truth)
            if accepted is None:
                raise ValueError("indeterminate complete K2 round")
            result.append((start+pos-(gen-1), {"request": request, "P": pos}))
            pos += accepted+1
    return result


def draw_sample(manifest_hash: str, populations: dict[str, list[RoundKey]], k2_hashes: dict[str, str]) -> Sample:
    splits = population_splits(populations)
    keys: dict[str, list[RoundKey]] = {}
    for split in splits:
        population = populations[split]
        # Independent deterministic RNG per split. Selected keys retain source
        # traversal order, so all passes accumulate in precisely the same order.
        selected = sorted(random.Random(SAMPLE_SEED).sample(range(len(population)), min(SAMPLE_LIMIT, len(population))))
        keys[split] = [population[i] for i in selected]
    sample: Sample = {"schema": "dsv41.d2.pilot-sample.v1", "seed": SAMPLE_SEED, "limit": SAMPLE_LIMIT,
                      "manifest_sha256": manifest_hash, "k2_sha256": k2_hashes,
                      "population_sha256": {s: key_hash(populations[s]) for s in splits},
                      "population_rounds": {s: len(populations[s]) for s in splits},
                      "keys": keys, "sha256": hash_json(keys)}
    validate_sample(sample, manifest_hash)
    return sample


def sample_plan(plan: Path, manifest: Path, k2: dict[str, Path]) -> None:
    from .pilot import manifest_is_synthetic, read_spec
    lock_fd = os.open(str(plan)+".sample.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(lock_fd, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = plan.read_bytes()
        if "sample" in json.loads(before):
            raise FileExistsError("pilot sample already exists; refusing to redraw")
        spec = read_spec(plan)
        manifest_hash = sha256(manifest)
        if manifest_hash != spec["manifest_sha256"]:
            raise ValueError("pilot sample manifest differs from prepared plan")
        if manifest_is_synthetic(manifest) != spec["synthetic"]:
            raise ValueError("pilot sample synthetic label differs from prepared plan")
        splits = population_splits(k2)
        if not spec["synthetic"] and splits != DEV_POPULATION:
            raise ValueError("real pilot requires amendment-7 dev population")
        if set(k2) != set(splits):
            raise ValueError("pilot sampling requires both K2 sources")
        hashes = {s: sha256(k2[s]) for s in splits}
        populations = {s: [key for _, key in complete_rounds(D2Blocks(manifest, s), k2[s])] for s in splits}
        spec["sample"] = draw_sample(manifest_hash, populations, hashes)
        if hashes != {s: sha256(k2[s]) for s in splits} or sha256(manifest) != manifest_hash or plan.read_bytes() != before:
            raise ValueError("pilot sampling inputs changed during the draw")
        # The only permitted plan mutation: atomically add its once-only sample
        # before any arm can start. The companion lock survives inode replacement.
        fd, name = tempfile.mkstemp(prefix=plan.name+".sample-", dir=plan.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(spec, f, indent=2, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, plan)
            directory = os.open(plan.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
