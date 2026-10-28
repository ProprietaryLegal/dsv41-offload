"""Opt-in generation taps. No vLLM imports; CPU-testable, bounded asynchronous D2H.

Mounted beside the image DSpark speculator. The serving thread never opens data
files or waits for a writer/event. Stage errors refuse only their named sequences; unattributable errors refuse
the capture globally, while serving continues. All tensor reads follow a tag
match. CUDA copies use the engine stream so subsequent target buffer reuse is
ordered after D2H; only the writer synchronizes the completion event.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
from typing import Protocol

import torch

TAG = re.compile(r'(?:^|(?:chat)?cmpl-)d2c\.([A-Za-z0-9_-]+)\.(\d+)\.(\d+)(?:\.(\d+))?(?:$|-(?:0-)?[0-9a-f]+$)')
ROW_BYTES = 30720
SYNC_BYTES = 256 * 1024 * 1024
SCHEMA = 'd2.decode-capture.v1'
log = logging.getLogger(__name__)


class Batch(Protocol):
    num_reqs: int
    num_tokens: int
    req_ids: Sequence[str]
    query_start_loc_np: Sequence[int]
    positions: torch.Tensor
    input_ids: torch.Tensor


@dataclass
class Stage:
    seq: str
    seg: int
    r0: int
    round: int
    layers: list[torch.Tensor]
    ids: torch.Tensor
    pos: torch.Tensor
    rejected: torch.Tensor
    event: torch.cuda.Event | None
    prompt_length: int | None = None
    staged_bytes: int = 0


@dataclass(frozen=True)
class Failure:
    message: str
    seq_ids: tuple[str, ...] | None = None


class ByteBudget:
    """Nonblocking serving-side reservation, including the writer's active stage."""
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = self.peak = 0
        self.lock = threading.Lock()

    def acquire(self, size: int, *, blocking: bool = False) -> bool:
        if not self.lock.acquire(blocking=blocking):
            return False
        try:
            if size < 1 or self.used + size > self.limit:
                return False
            self.used += size
            self.peak = max(self.peak, self.used)
            return True
        finally:
            self.lock.release()

    def release(self, size: int) -> None:
        with self.lock:
            if not 0 < size <= self.used:
                raise ValueError('capture staging reservation underflow')
            self.used -= size


def sha(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    while data:
        n = os.pwrite(fd, data, offset)
        if n <= 0:
            raise OSError('short capture pwrite')
        data, offset = data[n:], offset + n


class SequenceFiles:
    """Writer-thread only. Prompt rows stay first-wins; newer segments replace speculative suffix rows."""
    def __init__(self, root: Path, seq: str, r0: int, *, sync_phase: int = 0) -> None:
        self.root, self.seq, self.r0 = root / seq, seq, r0
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.seen: dict[int, int] = {}
        self.since_sync = 0
        self.dup = self.differing = self.zero = 0
        header = self.root / 'CAPTURE.json'
        if header.exists():
            meta = json.loads(header.read_text())
            if meta['schema'] != SCHEMA or meta['seq'] != seq or r0 < meta['from']:
                raise ValueError('capture identity/origin changed on resume')
            self.r0 = meta['from']
            for line in (self.root / 'ledger.jsonl').read_text().splitlines():
                row = json.loads(line)
                replaced = {r['position'] for r in row.get('replaced_rows', [])}
                for p in range(row['first'], row['first'] + row['rows']):
                    if p not in self.seen or p in replaced:
                        self.seen[p] = row['seg']
                self.dup += row['dup_rows']
                self.differing += row['dup_rows_differing']
                self.zero += row['zero_rows']
        else:
            if any(self.root.iterdir()):
                raise ValueError('capture files without identity header')
            header.write_text(json.dumps({'schema': SCHEMA, 'taps_source': 'decode_capture',
                                         'seq': seq, 'from': r0, 'row_bytes': ROW_BYTES,
                                         'capture_sha256': sha(Path(__file__).read_bytes())}))
        self.ft = os.open(self.root / 'taps.bin', os.O_CREAT | os.O_RDWR, 0o600)
        self.fi = os.open(self.root / 'ids.bin', os.O_CREAT | os.O_RDWR, 0o600)
        self.ledger = (self.root / 'ledger.jsonl').open('a')
        owner = root.stat()
        if owner.st_uid != 0:
            # Host-readable while the capture runs (mid-run ledger checks, coordinator polling), not only after a
            # flush: the container writes as root, so new files take the pre-created run dir's owner at creation,
            # the same rule as the replay overlay. A root-owned run dir is reported by Capture and left as is.
            for created in (self.root, header, self.root / 'taps.bin', self.root / 'ids.bin', self.root / 'ledger.jsonl'):
                os.chown(created, owner.st_uid, owner.st_gid)
        self.sync()
        # Three simultaneous streams reach their first sync at 256/171/85 MiB,
        # then each syncs every 256 MiB. This also staggers reopened files.
        self.since_sync = sync_phase % SYNC_BYTES

    def write(self, s: Stage) -> None:
        n = s.pos.numel() - int(s.rejected[0])
        if not 0 <= n <= s.pos.numel() or s.r0 < self.r0:
            raise ValueError('invalid rejected count or resume origin')
        prompt_length = s.prompt_length if s.prompt_length is not None else s.r0 + 128
        if max(0, prompt_length - 128) != s.r0:
            raise ValueError('prompt length and tag origin disagree')
        pos = s.pos[:n].tolist()
        if pos and pos != list(range(pos[0], pos[0] + n)):
            raise ValueError('noncontiguous validated positions')
        keep = [i for i, p in enumerate(pos) if p >= s.r0]
        if not keep:
            return
        lo, hi = keep[0], keep[-1] + 1
        rows = torch.cat(s.layers, dim=-1)[lo:hi]
        tb = rows.view(torch.int16).numpy().astype('<i2', copy=False).tobytes()
        ib = s.ids[lo:hi].numpy().astype('<i4', copy=False).tobytes()
        zeros = (rows.view(-1, 3, 5120) == 0).all(-1).any(-1).tolist()
        dup = differing = zero = 0
        retained_t, retained_i = bytearray(), bytearray()
        replaced_rows: list[dict[str, int]] = []
        hashes_t: list[str] = []
        hashes_i: list[str] = []
        for j, p in enumerate(pos[lo:hi]):
            off = p - self.r0
            t, token = tb[j * ROW_BYTES:(j + 1) * ROW_BYTES], ib[j * 4:(j + 1) * 4]
            if p in self.seen:
                dup += 1
                old_t, old_i = os.pread(self.ft, ROW_BYTES, off * ROW_BYTES), os.pread(self.fi, 4, off * 4)
                differing += int(old_t != t or old_i != token)
                if p >= prompt_length and self.seen[p] < s.seg:
                    replaced_rows.append({'position': p, 'seg': self.seen[p]})
                    pwrite_all(self.ft, t, off * ROW_BYTES)
                    pwrite_all(self.fi, token, off * 4)
                    self.seen[p] = s.seg
                    zero += int(zeros[j])
                else:
                    t, token = old_t, old_i
            else:
                pwrite_all(self.ft, t, off * ROW_BYTES)
                pwrite_all(self.fi, token, off * 4)
                self.seen[p] = s.seg
                zero += int(zeros[j])
            hashes_t.append(sha(t))
            hashes_i.append(sha(token))
            retained_t.extend(t)
            retained_i.extend(token)
        count = hi - lo
        self.dup += dup
        self.differing += differing
        self.zero += zero
        self.ledger.write(json.dumps({'schema': SCHEMA, 'seq': self.seq, 'seg': s.seg, 'tag_r0': s.r0,
            'prompt_length': prompt_length, 'replaced_rows': replaced_rows,
            'row_sha256_taps': hashes_t, 'row_sha256_ids': hashes_i,
            'round': s.round, 'first': pos[lo], 'rows': count, 'dup_rows': dup,
            'dup_rows_differing': differing, 'zero_rows': zero,
            'sha256_taps': sha(retained_t), 'sha256_ids': sha(retained_i)}) + '\n')
        self.ledger.flush()
        self.since_sync += len(retained_t) + len(retained_i)
        if self.since_sync >= SYNC_BYTES:
            self.sync()

    def sync(self) -> None:
        for fd in (self.ft, self.fi):
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self.ledger.flush()
        os.fdatasync(self.ledger.fileno())
        self.since_sync = 0

    def close(self) -> None:
        self.sync()
        os.close(self.ft)
        os.close(self.fi)
        self.ledger.close()


class Capture:
    def __init__(self, root: str, *, staging_bytes: int = 256 * 1024 * 1024,
                 max_stage_bytes: int = 64 * 1024 * 1024,
                 max_open_sequences: int = 32) -> None:
        self.root = Path(root)
        if not self.root.is_dir() or staging_bytes < 1 or max_stage_bytes < 1 or max_open_sequences < 1:
            raise ValueError('create a private host-owned capture directory before starting')
        self.owner = self.root.stat()
        if self.owner.st_uid == 0:
            raise ValueError('capture directory is root-owned; pre-create it as the host user')
        self.faulted = threading.Event()
        self.budget = ByteBudget(staging_bytes)
        self.max_stage_bytes = max_stage_bytes
        self.max_open_sequences = max_open_sequences
        self.jobs: queue.SimpleQueue[Stage | Failure | None] = queue.SimpleQueue()
        self.files: dict[str, SequenceFiles] = {}
        self.active = threading.local()
        self.round = 0
        errors_path = self.root / 'ERRORS.jsonl'
        previous = [json.loads(line) for line in errors_path.read_text().splitlines()] if errors_path.exists() else []
        self.errors = len(previous)
        if any(not row.get('seq_ids') for row in previous):
            self.faulted.set()
        self.sync_phases = {p.name: i for i, p in enumerate(sorted(p for p in self.root.iterdir() if p.is_dir()))}
        self._error_count()
        self.flush_nonce = ''
        self.thread = threading.Thread(target=self._run, daemon=True, name='d2-decode-writer')
        self.thread.start()

    def error(self, message: str, seq_ids: tuple[str, ...] | None = None) -> None:
        if seq_ids is None:
            self.faulted.set()
        log.error('D2 decode capture refused: %s', message)
        self.jobs.put(Failure(message, seq_ids))

    def enqueue(self, ib: Batch, aux: Sequence[torch.Tensor], rejected: torch.Tensor) -> None:
        # Untagged production requests: regex only, no tensor property or transfer.
        tagged = [(b, m) for b, rid in enumerate(ib.req_ids[:ib.num_reqs]) if (m := TAG.fullmatch(rid))]
        if not tagged or self.faulted.is_set():
            return
        self.round += 1
        for b, m in tagged:
            reserved = 0
            event: torch.cuda.Event | None = None
            try:
                start, end = int(ib.query_start_loc_np[b]), min(int(ib.query_start_loc_np[b + 1]), ib.num_tokens)
                if start < 0 or end <= start:
                    raise ValueError('empty or invalid tagged stage')
                # CPU prefill offsets are exact. The decode computed-token mirror
                # is optimistic with async scheduling, so never use it to trim decode.
                prefilling = getattr(ib, 'is_prefilling_np', None)
                if prefilling is not None and prefilling[b]:
                    computed = int(getattr(ib, 'num_computed_prefill_tokens_np')[b])
                    start = min(end, start + max(0, int(m[3]) - computed))
                n = end - start
                if n == 0:
                    continue
                size = n * (ROW_BYTES + 12) + 8
                if size > self.max_stage_bytes:
                    raise ValueError('oversized tagged stage')
                if not self.budget.acquire(size, blocking=False):
                    self.error('bounded staging exhausted; stage dropped', (m[1],))
                    continue
                reserved = size
                if len(aux) != 3 or any(t.dtype != torch.bfloat16 or t.shape[1:] != (5120,) for t in aux):
                    raise ValueError('capture requires three [T,5120] BF16 taps')
                cuda = aux[0].device.type == 'cuda'
                ids = torch.empty(n, dtype=torch.int32, pin_memory=cuda)
                pos = torch.empty(n, dtype=torch.int64, pin_memory=cuda)
                rej = torch.empty(1, dtype=torch.int64, pin_memory=cuda)
                layers: list[torch.Tensor] = []
                for src in aux:
                    stage = torch.empty((n, 5120), dtype=torch.bfloat16, pin_memory=cuda)
                    stage.copy_(src[start:end], non_blocking=True)
                    layers.append(stage)
                ids.copy_(ib.input_ids[start:end], non_blocking=True)
                pos.copy_(ib.positions[start:end], non_blocking=True)
                rej.copy_(rejected[b:b + 1], non_blocking=True)
                if cuda:
                    event = torch.cuda.Event()  # type: ignore[no-untyped-call]  # torch CUDA factory lacks typed constructor
                    event.record(torch.cuda.current_stream(aux[0].device))
                self.jobs.put(Stage(m[1], int(m[2]), int(m[3]), self.round,
                                    layers, ids, pos, rej, event, int(m[4]) if m[4] else int(m[3]) + 128, size))
            except Exception as exc:
                # The pinned allocator tracks in-flight D2H events when references
                # are released. Never synchronize the serving thread on an error.
                if reserved:
                    self.budget.release(reserved)
                self.error(type(exc).__name__ + ': ' + str(exc), (m[1],))

    def _owned(self) -> None:
        for p in self.root.rglob('*'):
            os.chown(p, self.owner.st_uid, self.owner.st_gid)
            os.chmod(p, 0o700 if p.is_dir() else 0o600)

    def _flush_request(self) -> None:
        request = self.root / 'FLUSH.json'
        if not request.exists():
            return
        nonce = json.loads(request.read_text())['nonce']
        if not isinstance(nonce, str) or not nonce or nonce == self.flush_nonce:
            return
        for f in self.files.values():
            f.sync()
        self._owned()
        directories = {p.parent.name: p.parent for p in self.root.glob('*/CAPTURE.json')}
        for directory in directories.values():
            with (directory / 'CAPTURE.json').open('rb') as header:
                os.fdatasync(header.fileno())
            fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        path = self.root / 'FLUSHED.json'
        tmp = path.with_suffix('.tmp')
        with tmp.open('w') as fh:
            json.dump({'schema': SCHEMA, 'nonce': nonce, 'errors': self.errors,
                       'sequences': {seq: {'ledger_sha256': sha((directory / 'ledger.jsonl').read_bytes()),
                                          'taps_bytes': (directory / 'taps.bin').stat().st_size,
                                          'ids_bytes': (directory / 'ids.bin').stat().st_size}
                                     for seq, directory in sorted(directories.items())}}, fh)
            fh.flush()
            os.fdatasync(fh.fileno())
        tmp.replace(path)
        os.chown(path, self.owner.st_uid, self.owner.st_gid)
        fd = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.flush_nonce = nonce

    def _run(self) -> None:
        while True:
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                try:
                    self._flush_request()
                except Exception as exc:
                    self._record_error(str(exc))
                continue
            if job is None:
                break
            if isinstance(job, Failure):
                self._record_error(job.message, job.seq_ids)
                continue
            try:
                if job.event is not None:
                    job.event.synchronize()
                f = self.files.get(job.seq)
                if f is None:
                    if len(self.files) >= self.max_open_sequences:
                        try:
                            self.files.pop(next(iter(self.files))).close()
                        except Exception as exc:
                            self._record_error('eviction flush: ' + str(exc))
                            continue
                    phase = self.sync_phases.setdefault(job.seq, len(self.sync_phases))
                    f = self.files[job.seq] = SequenceFiles(self.root, job.seq, job.r0,
                        sync_phase=(phase % 3) * (SYNC_BYTES // 3))
                f.write(job)
            except Exception as exc:
                self._record_error(type(exc).__name__ + ': ' + str(exc), (job.seq,))
            finally:
                self.budget.release(job.staged_bytes)
        try:
            for f in self.files.values():
                f.close()
            self._owned()
        except Exception as exc:
            self._record_error('close/ownership: ' + str(exc))

    def _error_count(self) -> None:
        path = self.root / 'ERROR_COUNT.json'
        tmp = path.with_suffix('.tmp')
        with tmp.open('w') as fh:
            json.dump({'schema': 'd2.decode-error-count.v1', 'errors': self.errors,
                       'global_refusal': self.faulted.is_set()}, fh)
            fh.flush()
            os.fdatasync(fh.fileno())
        tmp.replace(path)
        os.chown(path, self.owner.st_uid, self.owner.st_gid)
        os.chmod(path, 0o600)

    def _record_error(self, message: str, seq_ids: tuple[str, ...] | None = None) -> None:
        if seq_ids is None:
            self.faulted.set()
        self.errors += 1
        log.error('D2 decode capture incomplete: %s', message)
        try:
            with (self.root / 'ERRORS.jsonl').open('a') as fh:
                fh.write(json.dumps({'schema': SCHEMA, 'error': message, 'seq_ids': seq_ids}) + '\n')
                fh.flush()
                os.fdatasync(fh.fileno())
            self._error_count()
        except OSError:
            log.exception('cannot persist D2 capture error; capture must not be used')

    def close(self) -> None:
        """Offline tests / orderly owner shutdown only; never called by a serving step."""
        self.jobs.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('capture writer did not drain')


def wrapper(capture: Capture, original: Callable[..., object]) -> Callable[..., object]:
    signature = inspect.signature(original)

    @functools.wraps(original)
    def propose(self: object, *args: object, **kwargs: object) -> object:
        ib = args[0] if args else kwargs.get('input_batch')
        # This is the entire opt-in hot-path cost for ordinary requests.
        if ib is None or not any(TAG.fullmatch(rid) for rid in getattr(ib, 'req_ids')):
            return original(self, *args, **kwargs)
        if getattr(capture.active, 'inside', False):
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a.get('dummy_run') or a.get('is_profile'):
            return original(self, *args, **kwargs)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return original(self, *args, **kwargs)
        if getattr(self, 'enable_adaptive_verification', False):
            seq_ids = tuple(m[1] for rid in getattr(ib, 'req_ids') if (m := TAG.fullmatch(rid)))
            capture.error('adaptive verification is enabled; CPU capture offsets are unsafe', seq_ids)
            return original(self, *args, **kwargs)
        capture.active.inside = True
        try:
            capture.enqueue(a['input_batch'], a['aux_hidden_states'], a['num_rejected'])
            return original(self, *args, **kwargs)
        finally:
            capture.active.inside = False
    return propose
