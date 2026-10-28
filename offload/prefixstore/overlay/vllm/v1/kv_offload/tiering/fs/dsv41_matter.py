# SPDX-License-Identifier: Apache-2.0
"""MC1 matter gate for the persistent KV file tier (standard library only).

Cached KV is case data at rest. The persistent tier writes and reads only for requests whose
``kv_transfer_params`` carry ``matter_ns`` (64 lowercase hex, an HMAC of the client folder name made on
Controller) and ``persist: true``. Everything else is refused before any disk access (default deny).

The excluded-namespace file holds hashes only (one 64-hex line each, sorted, unique). It is pinned by
sha256. If it is missing, unreadable, not a regular file, group/other accessible, malformed or
sha-mismatched, the gate is CLOSED: no persistent PUT or GET for anyone (the in-memory tiers are
unaffected). A later change of the file closes the gate until restart (sticky).

The audit log is append-only JSONL with namespaces, counts and bytes; never names, never content.
If it cannot be opened the gate is closed too.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
from typing import Any

NS_RE = re.compile(r'[0-9a-f]{64}')
MAX_EXCLUDED_FILE_BYTES = 1 << 20
NS_DIR = 'ns'
PURGE_DIR = 'purge-requests'
PURGE_ALL = 'ALL'
LOCK_FILE = '.lock'


class Verdict(Enum):
    ALLOW = 'allow'
    UNTAGGED = 'untagged'          # no matter tag at all: never persisted, not audited per request
    NOT_PERSIST = 'not-persist'    # a namespace without persist: true
    BAD_TAG = 'bad-tag'            # malformed matter_ns / persist
    EXCLUDED = 'excluded'          # namespace in the pinned exclusion set
    GATE_CLOSED = 'gate-closed'    # fail closed: exclusion file or audit log unusable
    PURGING = 'purging'            # namespace being purged (tombstone)


# Verdicts that produce one audit 'refuse' line per request (untagged traffic is only counted).
AUDITED_REFUSALS = frozenset({Verdict.BAD_TAG, Verdict.EXCLUDED, Verdict.GATE_CLOSED, Verdict.PURGING})


def parse_tag(params: object) -> tuple[Verdict, str | None]:
    """Classify a request's kv_transfer_params. Only an exact bool True persists."""
    if not isinstance(params, dict) or ('matter_ns' not in params and 'persist' not in params):
        return Verdict.UNTAGGED, None
    ns = params.get('matter_ns')
    persist = params.get('persist')
    if not isinstance(ns, str) or NS_RE.fullmatch(ns) is None:
        return Verdict.BAD_TAG, None
    if persist is not True:
        return (Verdict.NOT_PERSIST, ns) if persist in (None, False) else (Verdict.BAD_TAG, ns)
    return Verdict.ALLOW, ns


def parse_excluded(data: bytes) -> frozenset[str]:
    """Strict format: >= 1 line, each exactly 64 lowercase hex, sorted, unique, LF-terminated, nothing else."""
    if not data or not data.endswith(b'\n'):
        raise ValueError('excluded-ns file empty or not LF-terminated')
    lines = data[:-1].split(b'\n')
    text = [line.decode('ascii', 'strict') for line in lines]
    if any(NS_RE.fullmatch(line) is None for line in text):
        raise ValueError('excluded-ns file has a line that is not 64 lowercase hex')
    if text != sorted(set(text)):
        raise ValueError('excluded-ns file is not sorted and unique')
    return frozenset(text)


class PrivateFileError(ValueError):
    """A private input is unusable; ``code`` is a stable reason code for the audit (no paths, no content)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def read_private(path: str, max_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read a regular, owner-only file without following a final symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError as exc:
        raise PrivateFileError('missing') from exc
    except OSError as exc:
        raise PrivateFileError('symlink' if exc.errno == errno.ELOOP else 'unreadable') from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PrivateFileError('not-regular')
        if st.st_mode & 0o077:
            raise PrivateFileError('mode')
        if st.st_size > max_bytes:
            raise PrivateFileError('too-large')
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b''.join(chunks), st
    finally:
        os.close(fd)


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode


class ExclusionSet:
    """The pinned excluded-namespace set, or a CLOSED state with a reason code."""

    def __init__(self, path: str, pinned_sha256: str, recheck_s: float = 30.0) -> None:
        self.path = path
        self.pinned = pinned_sha256
        self.recheck_s = recheck_s
        self.excluded: frozenset[str] = frozenset()
        self.reason: str | None = 'not-loaded'
        self._identity: tuple[int, int, int, int, int] | None = None
        self._last_check = 0.0
        self._load()

    @property
    def closed(self) -> bool:
        return self.reason is not None

    def _load(self) -> None:
        if NS_RE.fullmatch(self.pinned) is None:
            self.reason = 'pin-malformed'
            return
        try:
            data, st = read_private(self.path, MAX_EXCLUDED_FILE_BYTES)
        except PrivateFileError as exc:
            self.reason = exc.code
            return
        except OSError:
            self.reason = 'unreadable'
            return
        if hashlib.sha256(data).hexdigest() != self.pinned:
            self.reason = 'sha-mismatch'
            return
        try:
            self.excluded = parse_excluded(data)
        except (ValueError, UnicodeDecodeError):
            self.reason = 'malformed'
            return
        self._identity = _identity(st)
        self.reason = None
        self._last_check = time.monotonic()

    def revalidate(self, now: float | None = None) -> bool:
        """Re-stat at most every recheck_s; any change of the file closes the gate (sticky). True if it closed now."""
        if self.closed:
            return False
        now = time.monotonic() if now is None else now
        if now - self._last_check < self.recheck_s:
            return False
        self._last_check = now
        try:
            st = os.stat(self.path, follow_symlinks=False)
        except OSError:
            self.reason = 'missing'
            return True
        if _identity(st) != self._identity:
            self.reason = 'changed'
            return True
        return False

    def is_excluded(self, ns: str) -> bool:
        return ns in self.excluded


class AuditLog:
    """Append-only JSONL, mode 0600. One line per event; namespaces, counts and bytes only."""

    def __init__(self, path: str, rank: int) -> None:
        self.path = path
        self.rank = rank
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            os.close(fd)
            raise ValueError('audit log must be a regular 0600 file')
        self._fd = fd
        self._lock = threading.Lock()

    def write(self, op: str, ns: str | None, **fields: Any) -> None:
        record: dict[str, Any] = {'ts': round(time.time(), 3), 'rank': self.rank, 'op': op, 'ns': ns}
        record.update(fields)
        line = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
        with self._lock:
            written = os.write(self._fd, line)  # O_APPEND: one write per line
        if written != len(line):
            raise OSError(errno.EIO, 'short audit write')

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(OSError):
                os.close(self._fd)


@dataclass(frozen=True)
class RequestTag:
    verdict: Verdict
    ns: str | None


class MatterGate:
    """Decides every persistent-tier PUT/GET. Thread-safe for reads; tombstones are lock-protected."""

    def __init__(self, exclusions: ExclusionSet, audit: AuditLog | None, audit_error: str | None = None) -> None:
        self.exclusions = exclusions
        self.audit = audit
        self.audit_error = audit_error
        self._tombstones: dict[str, float] = {}
        self._lock = threading.Lock()
        self.untagged_requests = 0

    @property
    def closed_reason(self) -> str | None:
        if self.audit is None:
            return self.audit_error or 'audit-unavailable'
        return self.exclusions.reason

    def classify(self, params: object) -> RequestTag:
        verdict, ns = parse_tag(params)
        return RequestTag(verdict, ns)

    def decide(self, tag: RequestTag) -> Verdict:
        """Live decision for a parsed tag (gate state and tombstones may change during a request)."""
        if tag.verdict is not Verdict.ALLOW:
            return tag.verdict
        assert tag.ns is not None
        if self.closed_reason is not None:
            return Verdict.GATE_CLOSED
        if self.exclusions.is_excluded(tag.ns):
            return Verdict.EXCLUDED
        if self.is_tombstoned(tag.ns):
            return Verdict.PURGING
        return Verdict.ALLOW

    def tombstone(self, ns: str, until: float) -> None:
        with self._lock:
            self._tombstones[ns] = max(until, self._tombstones.get(ns, 0.0))

    def is_tombstoned(self, ns: str) -> bool:
        with self._lock:
            return ns in self._tombstones or PURGE_ALL in self._tombstones

    def expire_tombstones(self, now: float) -> list[str]:
        with self._lock:
            done = [ns for ns, until in self._tombstones.items() if until <= now]
            for ns in done:
                del self._tombstones[ns]
        return done


def ns_root(root: str, ns: str) -> str:
    if NS_RE.fullmatch(ns) is None:
        raise ValueError('namespace must be 64 lowercase hex')
    return os.path.join(root, NS_DIR, ns)


def namespaces(root: str) -> list[str]:
    try:
        names = os.listdir(os.path.join(root, NS_DIR))
    except FileNotFoundError:
        return []
    return sorted(n for n in names if NS_RE.fullmatch(n))


def tree_bytes(path: str) -> tuple[int, int]:
    """(files, bytes) under path; missing path is (0, 0)."""
    files = size = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            with contextlib.suppress(FileNotFoundError):
                size += os.stat(os.path.join(dirpath, name), follow_symlinks=False).st_size
                files += 1
    return files, size


def purge_namespace(root: str, ns: str, attempts: int = 5) -> tuple[int, int]:
    """Delete one namespace's tree. Returns (files, bytes) found before deletion.

    Never follows symlinks out of root (rmtree refuses a symlinked top and does not descend into links).
    A store that lands during the deletion can re-create directories; retry, then fail loudly."""
    path = ns_root(root, ns)
    if os.path.islink(path):
        raise ValueError('namespace directory is a symlink')
    files, size = tree_bytes(path)
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            if attempt == attempts - 1:
                raise
        if not os.path.lexists(path):
            return files, size
        time.sleep(0.05)
    raise OSError(errno.ENOTEMPTY, 'namespace directory reappeared during purge')


def ensure_private_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError(f'{path} is not a directory')
    if st.st_mode & 0o077:
        os.chmod(path, 0o700)
        if os.stat(path).st_mode & 0o077:
            raise ValueError(f'{path} must be 0700')


def take_purge_requests(root: str) -> list[str]:
    """Consume online purge requests: files named <ns>.req or ALL.req in <root>/purge-requests."""
    directory = os.path.join(root, PURGE_DIR)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return []
    out: list[str] = []
    for name in names:
        if not name.endswith('.req'):
            continue
        target = name[:-4]
        if target != PURGE_ALL and NS_RE.fullmatch(target) is None:
            continue
        with contextlib.suppress(FileNotFoundError):
            os.unlink(os.path.join(directory, name))
            out.append(target)
    return out


# --------------------------------------------------------------------------- MC1 item 6: bounds, TTL, eviction
@dataclass(frozen=True)
class BlockFile:
    """One stored chunk: its data file (+ CRC sidecar), size of both, last use (data file mtime)."""

    path: str
    nbytes: int
    mtime: float


def scan_namespace(root: str, ns: str) -> tuple[list[BlockFile], int]:
    """Every chunk file of a namespace (oldest first) and the bytes of everything else under it (config files,
    stale temp files). Missing namespace: ([], 0)."""
    base = ns_root(root, ns)
    blocks: list[BlockFile] = []
    other = 0
    sizes: dict[str, os.stat_result] = {}
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            path = os.path.join(dirpath, name)
            with contextlib.suppress(FileNotFoundError):
                sizes[path] = os.stat(path, follow_symlinks=False)
    for path, st in sizes.items():
        if path.endswith('.bin'):
            crc = sizes.get(path + '.crc')
            blocks.append(BlockFile(path, st.st_size + (crc.st_size if crc else 0), st.st_mtime))
        elif not (path.endswith('.bin.crc') and path[:-4] in sizes):
            other += st.st_size
    blocks.sort(key=lambda b: (b.mtime, b.path))
    return blocks, other


def remove_block(block: BlockFile) -> int:
    """Delete a chunk file and its CRC sidecar (sidecar first, so a half-removed block is a CRC miss)."""
    freed = 0
    for path in (block.path + '.crc', block.path):
        with contextlib.suppress(FileNotFoundError):
            size = os.stat(path, follow_symlinks=False).st_size
            os.unlink(path)
            freed += size
    return freed


def prune_empty_dirs(base: str) -> None:
    for dirpath, _, _ in sorted(os.walk(base, topdown=False), key=lambda t: -len(t[0])):
        with contextlib.suppress(OSError):
            os.rmdir(dirpath)


def remove_stale_temps(root: str, ns: str, older_than: float) -> int:
    """Temp files a crashed writer left behind (never valid blocks)."""
    freed = 0
    for dirpath, _, filenames in os.walk(ns_root(root, ns)):
        for name in filenames:
            if name.endswith('.tmp'):
                path = os.path.join(dirpath, name)
                with contextlib.suppress(FileNotFoundError):
                    st = os.stat(path, follow_symlinks=False)
                    if st.st_mtime < older_than:
                        os.unlink(path)
                        freed += st.st_size
    return freed


class StoreUsage:
    """Byte accounting of the persistent tier, per namespace, for the LRU bounds (thread-safe).

    Bytes are exact after a scan and an upper bound in between (a store of an already present chunk is counted
    again until the next scan), so the bounds are never exceeded by under-counting. Namespace recency (last put or
    hit) orders whole-namespace LRU for the tier bound; file mtimes order eviction inside a namespace."""

    def __init__(self, max_bytes: int, max_ns_bytes: int, high: float = 0.9, low: float = 0.8) -> None:
        if max_bytes <= 0 or max_ns_bytes <= 0:
            raise ValueError('MC1: the matter tier needs positive byte bounds (tier and namespace)')
        if not 0 < low < high <= 1:
            raise ValueError('MC1: bounds need 0 < low < high <= 1')
        self.max_bytes = max_bytes
        self.max_ns_bytes = min(max_ns_bytes, max_bytes)
        self.high = high
        self.low = low
        self._bytes: dict[str, int] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.ready = False

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._bytes.values())

    def ns_bytes(self, ns: str) -> int:
        with self._lock:
            return self._bytes.get(ns, 0)

    def namespaces(self) -> list[str]:
        with self._lock:
            return list(self._bytes)

    def set_ns(self, ns: str, nbytes: int, last_use: float) -> None:
        with self._lock:
            if nbytes <= 0:
                self._bytes.pop(ns, None)
                self._last.pop(ns, None)
            else:
                self._bytes[ns] = nbytes
                self._last[ns] = max(last_use, self._last.get(ns, 0.0))

    def add(self, ns: str, nbytes: int, now: float) -> None:
        with self._lock:
            self._bytes[ns] = self._bytes.get(ns, 0) + nbytes
            self._last[ns] = now

    def touch(self, ns: str, now: float) -> None:
        with self._lock:
            if ns in self._bytes:
                self._last[ns] = now

    def admit(self, ns: str, nbytes: int) -> str | None:
        """None if a store of nbytes fits under both hard bounds now; else the refusal reason."""
        with self._lock:
            if not self.ready:
                return 'scan-pending'
            if sum(self._bytes.values()) + nbytes > self.max_bytes:
                return 'tier-cap'
            if self._bytes.get(ns, 0) + nbytes > self.max_ns_bytes:
                return 'ns-cap'
            return None

    def over_high(self) -> bool:
        with self._lock:
            return (sum(self._bytes.values()) > self.high * self.max_bytes
                    or any(b > self.high * self.max_ns_bytes for b in self._bytes.values()))

    def lru_namespaces(self) -> list[str]:
        with self._lock:
            return sorted(self._bytes, key=lambda ns: (self._last.get(ns, 0.0), ns))
