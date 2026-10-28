# SPDX-License-Identifier: Apache-2.0
"""MC1 per-block payload hash dump (diagnostic; off unless VLLM_DSV41_OFFLOAD_HASH_DUMP is set).

The byte-equality gate (F-MATTER 3.3): the GPU KV blocks of a synthetic prefix must be byte-identical after an
in-memory hit and after a restart-and-restore, on both ranks. Each worker writes one JSONL line per GPU block it
moves, keyed by the chunk's offload key (a content hash), so two runs and two ranks join without request ids:

* ``store``: the GPU source blocks of every store job, read on EVERY rank (also ranks that do not write the CPU
  tier), right after the forward step that computed them: the in-memory pool bytes.
* ``load``:  the GPU destination blocks of every load job (or stream window) once its copy completed: the
  restored pool bytes.

Line: ``{ev, rank, job, g, lb, key, sub, sha, seg}``: group position, request-relative block index, offload key
hex, sub-block within the chunk, sha256 of the block row, and digests of fixed-size segments (to localize padding
or inactive regions when ranks differ). Standard library only; the worker supplies the row bytes.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import hashlib
import json
import os
import stat
import threading
from typing import Any

SEGMENT_DIGEST_BYTES = 8


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def block_layout(group_sizes: Sequence[int], block_indices: Sequence[int], keys: Sequence[str] | None,
                 blocks_per_chunk: int) -> Iterator[tuple[int, int, int, str | None, int]]:
    """Yield (position in block_ids, group, request-relative block index, key, sub-block) for every block, with the
    CPU->GPU handler's arithmetic: group g's first block is sub-block block_indices[g] % bpc of its first chunk."""
    pos = 0
    key_pos = 0
    for g, (size, start) in enumerate(zip(group_sizes, block_indices)):
        if size == 0:
            continue
        skip = start % blocks_per_chunk
        n_keys = cdiv(size + skip, blocks_per_chunk)
        if keys is not None and key_pos + n_keys > len(keys):
            raise ValueError('MC1 hash dump: fewer keys than the spec\'s chunks')
        for i in range(size):
            sub = skip + i
            key = keys[key_pos + sub // blocks_per_chunk] if keys is not None else None
            yield pos, g, start + i, key, sub % blocks_per_chunk
            pos += 1
        key_pos += n_keys
    if keys is not None and key_pos != len(keys):
        raise ValueError('MC1 hash dump: more keys than the spec\'s chunks')


def digests(row: bytes, segment: int) -> tuple[str, str]:
    seg = ''
    if segment > 0:
        seg = ''.join(hashlib.blake2b(row[i:i + segment], digest_size=SEGMENT_DIGEST_BYTES).hexdigest()
                      for i in range(0, len(row), segment))
    return hashlib.sha256(row).hexdigest(), seg


class BlockHashDump:
    """Append-only 0600 JSONL at ``<prefix>_r<rank>.jsonl``."""

    def __init__(self, prefix: str, rank: int, blocks_per_chunk: int, segment: int = 4096) -> None:
        if not prefix:
            raise ValueError('MC1 hash dump: empty path prefix')
        self.rank = rank
        self.blocks_per_chunk = blocks_per_chunk
        self.segment = segment
        self.path = f'{prefix}_r{rank}.jsonl'
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
        if os.fstat(self._fd).st_mode & 0o077:
            os.fchmod(self._fd, 0o600)
        self._lock = threading.Lock()

    def dump(self, ev: str, job: int, spec: Any, read_rows: Callable[[int, list[int]], list[bytes]]) -> int:
        """Hash every block of one GPU spec. ``read_rows(group, block_ids)`` returns each block's bytes."""
        group_sizes = list(spec.group_sizes)
        block_indices = list(spec.block_indices)
        keys: list[str] | None = getattr(spec, 'dsv41_keys', None)
        ids = [int(b) for b in spec.block_ids]
        layout = list(block_layout(group_sizes, block_indices, keys, self.blocks_per_chunk))
        lines: list[str] = []
        for g in sorted({entry[1] for entry in layout}):
            entries = [e for e in layout if e[1] == g]
            rows = read_rows(g, [ids[e[0]] for e in entries])
            if len(rows) != len(entries):
                raise ValueError('MC1 hash dump: row reader returned the wrong number of rows')
            for (_, _, lb, key, sub), row in zip(entries, rows):
                sha, seg = digests(row, self.segment)
                record: dict[str, Any] = {'ev': ev, 'rank': self.rank, 'job': job, 'g': g, 'lb': lb, 'key': key,
                                          'sub': sub, 'sha': sha, 'seg': seg, 'n': len(row)}
                lines.append(json.dumps(record, sort_keys=True, separators=(',', ':')))
        if lines:
            data = ('\n'.join(lines) + '\n').encode()
            with self._lock:
                view = memoryview(data)
                while view:
                    view = view[os.write(self._fd, view):]
        return len(lines)

    def close(self) -> None:
        with self._lock:
            os.close(self._fd)


def read_dump(path: str) -> list[dict[str, Any]]:
    if os.stat(path).st_mode & stat.S_IFREG == 0:
        raise ValueError(f'{path} is not a file')
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
