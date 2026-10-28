#!/usr/bin/env python3
"""DSV41 matter-cache client helper: tag an OpenAI-compatible request for the persistent matter cache.

One file, standard library only, harness-agnostic: local-law, the drafting harness or a script vendors this file as is and
uses it for every request it sends to the DeepSeek server (the cache lives inside that server, below any harness)::

    tagger = MatterTagger.from_env()                  # or MatterTagger(key_file, exclusion_list=..., ...)
    body = tagger.tag(client_folder).apply(body)      # JSON body of /v1/chat/completions (or /v1/completions)
    # OpenAI SDK: client.chat.completions.create(..., extra_body=tagger.tag(client_folder).body_fields())

What it does, and nothing else:

* **Namespace.** ``matter_ns = HMAC-SHA256(K, b"dsv41-matter-ns-v1\\0" + canonical(folder name))``, hex. The folder
  name is the last component of the client-folder string (``"Lastname Firstname"``, ``".../client folders/<name>"``
  and ``"...\\client folders\\!Closed\\<name>"`` give the same namespace, so a matter keeps it when closed);
  ``canonical`` = NFKC, casefold, whitespace runs collapsed, stripped. This file is the single definition:
  ``ns.py`` and ``derive_excluded_ns.py`` import it, so the client, the exclusion hashes and purge agree.
* **Exclusions before sending.** An excluded matter is never tagged. Sources (at least one): the private name list
  (compared by canonical name AND by namespace, independently of the server's hash file) and/or the hash-only
  excluded-namespace file the server also loads; each optionally pinned by sha256. A source that changes is
  reloaded; one that cannot be read, fails its pin or is malformed makes EVERY request untagged until it reads
  cleanly again (fail closed: nothing persists, in-memory reuse continues).
* **Request fields.** A persisting tag adds ``kv_transfer_params = {"matter_ns": ns, "persist": true}`` and
  ``cache_salt = ns`` (the prompt-layout rule; it also keeps matters apart in the GPU prefix cache). Any other tag
  (excluded, disabled, not requested, invalid folder, exclusions unavailable) adds nothing and strips a stale
  ``matter_ns``/``persist``: the request is untagged, is never persisted, and keeps normal in-memory reuse.

Never logged, raised or returned: a name, the key, or a namespace of an excluded matter. Reasons are fixed codes.
The key is read once; a new key means new namespaces (restart the harness and re-derive the hash file).

CLI (counts only)::

    python3 dsv41_matter_client.py selfcheck --key K --exclusion-list L [--excluded-ns F] [--client-files DIR ...]
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import logging
import os
import re
import stat
import sys
import threading
from typing import Any
import unicodedata

DOMAIN = b'dsv41-matter-ns-v1\x00'
KEY_BYTES = 32
NS_RE = re.compile(r'[0-9a-f]{64}')
MAX_PRIVATE_BYTES = 1 << 20
_WS = re.compile(r'\s+')
_SEPARATORS = '/\\'

PERSIST = 'persist'
EXCLUDED = 'excluded'
DISABLED = 'disabled'
NOT_REQUESTED = 'not-requested'
INVALID_FOLDER = 'invalid-folder'
EXCLUSIONS_UNAVAILABLE = 'exclusions-unavailable'

log = logging.getLogger('dsv41_matter_client')


class MatterInputError(ValueError):
    """An input is unusable. The message never contains a matter name, a namespace or key material."""


# ------------------------------------------------------------------ namespace
def canonical(name: str) -> str:
    return _WS.sub(' ', unicodedata.normalize('NFKC', name).casefold()).strip()


def folder_name(folder: str) -> str:
    """Last path component of a client-folder string ('/' or '\\'); refuses empty, '.', '..' and '!' folders."""
    name = folder.rstrip(_SEPARATORS + ' \t\r\n')
    for sep in _SEPARATORS:
        name = name.rsplit(sep, 1)[-1]
    name = name.strip()
    if name in ('', '.', '..') or name.startswith('!'):
        raise MatterInputError('not a client folder name (empty, dot or a "!" system folder)')
    return name


def matter_key(folder: str) -> str:
    text = canonical(folder_name(folder))
    if not text:
        raise MatterInputError('empty matter name after canonicalization')
    return text


def derive_ns(key: bytes, folder: str) -> str:
    if len(key) != KEY_BYTES:
        raise MatterInputError('HMAC key must be exactly 32 bytes')
    return hmac.new(key, DOMAIN + matter_key(folder).encode('utf-8'), hashlib.sha256).hexdigest()


# -------------------------------------------------------------- private files
def read_private(path: str, what: str, max_bytes: int = MAX_PRIVATE_BYTES) -> bytes:
    """Owner-only regular file, final symlink refused; errors name the input kind, never its content."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError as exc:
        raise MatterInputError(f'{what}: missing') from exc
    except OSError as exc:
        raise MatterInputError(f'{what}: cannot open (symlink or permission)') from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise MatterInputError(f'{what}: not a regular file')
        if st.st_mode & 0o077:
            raise MatterInputError(f'{what}: mode {stat.S_IMODE(st.st_mode):o}, 0600 required')
        if st.st_size > max_bytes:
            raise MatterInputError(f'{what}: too large')
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        return b''.join(chunks)
    finally:
        os.close(fd)


def load_key(path: str) -> bytes:
    key = read_private(path, 'HMAC key')
    if len(key) != KEY_BYTES:
        raise MatterInputError('HMAC key: must be exactly 32 bytes')
    return key


def parse_names(data: bytes) -> list[str]:
    """Entries of the private exclusion list ('#' comments and blank lines skipped); errors cite no content."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise MatterInputError('exclusion list: not UTF-8') from exc
    names = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not names:
        raise MatterInputError('exclusion list: no entries')
    return names


def parse_excluded_ns(data: bytes) -> frozenset[str]:
    """The server's strict format: >= 1 line, each 64 lowercase hex, sorted, unique, LF-terminated."""
    if not data or not data.endswith(b'\n'):
        raise MatterInputError('excluded-ns file: empty or not LF-terminated')
    try:
        lines = data[:-1].decode('ascii').split('\n')
    except UnicodeDecodeError as exc:
        raise MatterInputError('excluded-ns file: not ASCII') from exc
    if any(NS_RE.fullmatch(line) is None for line in lines) or lines != sorted(set(lines)):
        raise MatterInputError('excluded-ns file: lines must be 64 lowercase hex, sorted and unique')
    return frozenset(lines)


def _pin(data: bytes, expected: str | None, what: str) -> None:
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise MatterInputError(f'{what}: sha256 differs from the pinned value')


def _identity(path: str) -> tuple[int, ...] | None:
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode


# ----------------------------------------------------------------- the tagger
@dataclass(frozen=True)
class MatterTag:
    """The outcome for one request. ``matter_ns`` is set only when ``persist`` is True."""

    persist: bool
    reason: str
    matter_ns: str | None = None

    def body_fields(self) -> dict[str, Any]:
        """Fields to add to the request body (``extra_body`` in the OpenAI SDK); empty unless persisting."""
        if not self.persist:
            return {}
        return {'kv_transfer_params': {'matter_ns': self.matter_ns, 'persist': True}, 'cache_salt': self.matter_ns}

    def apply(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """A copy of ``body`` carrying this tag. A stale ``matter_ns``/``persist`` never survives a non-persisting
        tag; another ``cache_salt`` on a persisting request is an error (the layout rule requires the namespace)."""
        out = dict(body)
        params = out.get('kv_transfer_params')
        if params is not None and not isinstance(params, Mapping):
            raise MatterInputError('kv_transfer_params must be a JSON object')
        merged = {k: v for k, v in (params or {}).items() if k not in ('matter_ns', 'persist')}
        if self.persist:
            if out.get('cache_salt') not in (None, self.matter_ns):
                raise MatterInputError('cache_salt is already set to a different value')
            merged.update(matter_ns=self.matter_ns, persist=True)
            out['cache_salt'] = self.matter_ns
        if merged:
            out['kv_transfer_params'] = merged
        else:
            out.pop('kv_transfer_params', None)
        return out


class _Source:
    """One exclusion source, reloaded when its file changes; ``error`` set while it cannot be used."""

    def __init__(self, path: str, pinned: str | None, what: str) -> None:
        if pinned is not None and NS_RE.fullmatch(pinned) is None:
            raise MatterInputError(f'{what}: pinned sha256 must be 64 lowercase hex')
        self.path, self.pinned, self.what = path, pinned, what
        self.identity: tuple[int, ...] | None = None
        self.names: frozenset[str] = frozenset()
        self.namespaces: frozenset[str] = frozenset()
        self.error: str | None = None

    def refresh(self, key: bytes) -> None:
        identity = _identity(self.path)
        if identity is not None and identity == self.identity and self.error is None:
            return
        try:
            data = read_private(self.path, self.what)
            _pin(data, self.pinned, self.what)
            if self.what == 'exclusion list':
                names = parse_names(data)
                keys: set[str] = set()
                for number, entry in enumerate(names, 1):
                    try:
                        keys.add(matter_key(entry))
                    except MatterInputError as exc:
                        raise MatterInputError(f'exclusion list entry {number}: {exc}') from None
                self.names = frozenset(keys)
                self.namespaces = frozenset(derive_ns(key, k) for k in keys)
            else:
                self.namespaces = parse_excluded_ns(data)
        except MatterInputError as exc:
            if self.error != str(exc):
                log.warning('matter cache client: %s; every request is sent untagged until it reads cleanly', exc)
            self.error = str(exc)
            self.identity = None
            return
        if self.error is not None:
            log.warning('matter cache client: %s readable again; tagging resumes', self.what)
        self.error = None
        self.identity = identity


class MatterTagger:
    """Thread-safe. Construct once per harness process; ``tag`` per request."""

    def __init__(self, key_file: str | None = None, *, exclusion_list: str | None = None,
                 exclusion_list_sha256: str | None = None, excluded_ns_file: str | None = None,
                 excluded_ns_sha256: str | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._sources: list[_Source] = []
        self._key = b''
        if not enabled:
            return
        if key_file is None:
            raise MatterInputError('matter cache enabled without an HMAC key file')
        if exclusion_list is None and excluded_ns_file is None:
            raise MatterInputError('matter cache enabled without an exclusion source (list or excluded-ns file)')
        self._key = load_key(key_file)
        if exclusion_list is not None:
            self._sources.append(_Source(exclusion_list, exclusion_list_sha256, 'exclusion list'))
        if excluded_ns_file is not None:
            self._sources.append(_Source(excluded_ns_file, excluded_ns_sha256, 'excluded-ns file'))
        for source in self._sources:
            source.refresh(self._key)
            if source.error is not None:
                raise MatterInputError(source.error)   # misconfiguration fails loudly at start

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MatterTagger:
        """``DSV41_MATTER_CACHE`` = 1/on (default 0/off) plus ``DSV41_MATTER_KEY_FILE``,
        ``DSV41_MATTER_EXCLUSION_LIST[_SHA256]`` and/or ``DSV41_MATTER_EXCLUDED_NS_FILE[_SHA256]``."""
        env = os.environ if environ is None else environ
        switch = env.get('DSV41_MATTER_CACHE', '0').strip().lower()
        if switch not in ('0', '1', 'off', 'on', ''):
            raise MatterInputError('DSV41_MATTER_CACHE must be 0/1/off/on')
        if switch not in ('1', 'on'):
            return cls(enabled=False)
        return cls(env.get('DSV41_MATTER_KEY_FILE') or None,
                   exclusion_list=env.get('DSV41_MATTER_EXCLUSION_LIST') or None,
                   exclusion_list_sha256=env.get('DSV41_MATTER_EXCLUSION_LIST_SHA256') or None,
                   excluded_ns_file=env.get('DSV41_MATTER_EXCLUDED_NS_FILE') or None,
                   excluded_ns_sha256=env.get('DSV41_MATTER_EXCLUDED_NS_SHA256') or None)

    def namespace(self, folder: str) -> str:
        """The namespace alone (e.g. purge by matter on Controller). Raises on an invalid folder or a disabled tagger."""
        if not self.enabled:
            raise MatterInputError('matter cache disabled')
        return derive_ns(self._key, folder)

    def tag(self, folder: str, persist: bool = True) -> MatterTag:
        """The tag for one request. Never raises for a bad folder or an unreadable source: it returns an untagged
        result with a reason code (log or count it; the request still runs, with in-memory reuse only)."""
        if not self.enabled:
            return MatterTag(False, DISABLED)
        if not persist:
            return MatterTag(False, NOT_REQUESTED)
        try:
            key_name = matter_key(folder)
        except MatterInputError:
            return MatterTag(False, INVALID_FOLDER)
        ns = derive_ns(self._key, folder)
        with self._lock:
            for source in self._sources:
                source.refresh(self._key)
            if any(source.error is not None for source in self._sources):
                return MatterTag(False, EXCLUSIONS_UNAVAILABLE)
            excluded = any(key_name in s.names or ns in s.namespaces for s in self._sources)
        return MatterTag(False, EXCLUDED) if excluded else MatterTag(True, PERSIST, ns)


# ------------------------------------------------------------------------ CLI
def _selfcheck(args: argparse.Namespace) -> int:
    key = load_key(args.key)
    names = _Source(args.exclusion_list, args.exclusion_list_sha256, 'exclusion list')
    names.refresh(key)
    if names.error is not None:
        raise MatterInputError(names.error)
    print(f'exclusion list: {len(names.names)} entries')
    status = 0
    if args.excluded_ns is not None:
        hashes = _Source(args.excluded_ns, args.excluded_ns_sha256, 'excluded-ns file')
        hashes.refresh(key)
        if hashes.error is not None:
            raise MatterInputError(hashes.error)
        agree = names.namespaces <= hashes.namespaces
        print(f'excluded-ns file: {len(hashes.namespaces)} entries; every listed matter is in it: '
              f'{"yes" if agree else "NO"}')
        status |= 0 if agree else 1
    if args.client_files:
        folders: set[str] = set()
        for root in args.client_files:
            for entry in os.scandir(root):
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('!'):
                    folders.add(canonical(entry.name))
        matched = sum(1 for n in names.names if n in folders)
        print(f'client folders: {matched} of {len(names.names)} exclusion entries match a folder name')
        status |= 0 if matched == len(names.names) else 1
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('selfcheck', help='exclusion sources agree / match client folders (prints counts only)')
    p.add_argument('--key', required=True)
    p.add_argument('--exclusion-list', required=True)
    p.add_argument('--exclusion-list-sha256')
    p.add_argument('--excluded-ns')
    p.add_argument('--excluded-ns-sha256')
    p.add_argument('--client-files', nargs='*', default=[],
                   help='client-folder roots (e.g. client folders and its !Closed) to check the names against')
    args = parser.parse_args(argv)
    try:
        return _selfcheck(args)
    except MatterInputError as exc:
        print(f'dsv41_matter_client: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
