"""Inline, cgroup-v2 ownership for window mutations, including nested SSH hops.

Persistent fences are control-plane transactions, deliberately outside the scopes
being killed. Business commands execute only after the in-scope guard checks the
fence under its registration lock. Never remove a run's control directory.
Containers created by dockerd require separate named-container/compute checks.
"""
from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

CONTROL = 'dsv41-offload-run/scoped-control'
FENCED = 'SCOPED REMOTE FENCED'
ABSENT = 'SCOPED REMOTE MUTATORS ABSENT'
CLOSED = 'SCOPED OWNED HOLD CLOSED'
RETAINED = 'SCOPED ORIGINAL HOLD RETAINED'
SSH_BINARY = '/usr/bin/ssh'
TMUX_BINARY = '/usr/bin/tmux'
HOSTS = frozenset(('sparkhead', 'sparkhead', 'spark', 'sparkworker', 'sparkworker', 'sparkworker-wired', 'x86host'))
NODES = ('sparkhead', 'sparkworker-wired', 'x86host')
CGROUP = Path('/sys/fs/cgroup')
Invoke = Callable[[str, str], tuple[int, str]]


def source() -> str:
    return str(globals()['SOURCE']) if 'SOURCE' in globals() else Path(__file__).read_text()


def root_for(tag: str) -> Path:
    if re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,95}', tag) is None:
        raise ValueError('invalid scoped run tag (1..96 safe characters required)')
    return Path.home() / CONTROL / tag


def expiry() -> float:
    deadline = float(os.environ['SCOPED_REMOTE_DEADLINE'])
    if not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError('scoped mutation deadline expired; HOLD stays open')
    return deadline


def command(mode: str, tag: str, argv: list[str] | None = None, *, deadline: float | None = None,
            close_owner: str | None = None) -> str:
    root_for(tag)
    code = 'SOURCE = ' + repr(source()) + '\nexec(compile(SOURCE,"<scoped-remote>","exec"))'
    if mode == 'launch':
        if deadline is None:
            deadline = float(os.environ['SCOPED_REMOTE_DEADLINE'])
        if not math.isfinite(deadline):
            raise ValueError('finite immutable scoped deadline required')
        lease = ['env', f'SCOPED_REMOTE_DEADLINE={deadline:.6f}']
        op = [uuid.uuid4().hex]
    else:
        lease = [] if deadline is None else ['env', f'SCOPED_REMOTE_DEADLINE={deadline:.6f}']
        op = []
    # Explicit owner survives nested SSH and the in-scope rm shim. Legacy tags
    # keep their full owner name, even if it happens to end in -close-txn-N.
    close_owner = close_owner if close_owner is not None else os.environ.get('SCOPED_CLOSE_OWNER')
    if close_owner is not None:
        root_for(close_owner)
        if re.fullmatch(re.escape(close_owner)+r'-close-txn-[1-9][0-9]*-restore-close', tag) is None:
            raise ValueError('numbered close transaction does not match its explicit owner')
        lease = ['env', *(lease[1:] if lease else []), f'SCOPED_CLOSE_OWNER={close_owner}']
    return shlex.join([*lease, 'python3', '-c', code, mode, tag, *op, *(argv or [])])


def checked(argv: list[str]) -> str:
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    if p.returncode:
        raise ValueError(f'scoped control failed rc{p.returncode}: {shlex.join(argv)}: {p.stderr.strip()}')
    return p.stdout


def units(tag: str) -> list[str]:
    prefix = 'dsv41w-' + tag + '-'
    output = checked(['systemctl', '--user', 'list-units', '--all', '--plain', '--no-legend',
                      '--no-pager', prefix + '*.scope'])
    result = []
    for line in output.splitlines():
        unit = line.split()[0]
        # systemctl's glob also matches other phase tags with this prefix.
        # Only an exact 32-hex operation suffix belongs to this fence.
        if re.fullmatch(re.escape(prefix) + '[0-9a-f]{32}\\.scope', unit):
            result.append(unit)
    return result


def paths(root: Path, tag: str) -> list[Path]:
    result = []
    for record in root.glob('*.cgroup'):
        value = record.read_text().strip()
        # Only a direct scope child of this user's user manager is accepted.
        expected = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/'
        unit = 'dsv41w-' + tag + '-' + record.stem + '.scope'
        if not value.startswith(expected) or Path(value).name != unit or '..' in Path(value).parts:
            raise ValueError('invalid owned cgroup record: ' + str(record))
        result.append(CGROUP / value.lstrip('/'))
    return result


def populated(path: Path) -> bool:
    try:
        values = dict(line.split() for line in (path/'cgroup.events').read_text().splitlines())
    except FileNotFoundError:
        # A removed cgroup cannot contain tasks; other read failures remain fatal.
        if path.exists():
            raise
        return False
    if values.get('populated') not in ('0', '1'):
        raise ValueError('unreadable cgroup populated proof: ' + str(path))
    return values['populated'] == '1'


def prove(root: Path, tag: str) -> None:
    if not (root/'ABORT').exists():
        raise ValueError('absence requires persistent fence')
    if any(p.stat().st_size for p in root.glob('*.watch.log')):
        raise ValueError('remote deadline guardian failed; inspect phase logs; HOLD stays open')
    if units(tag) or any(populated(p) for p in paths(root, tag)):
        raise ValueError('owned scopes/cgroups remain; HOLD stays open')


def reap(root: Path, tag: str) -> None:
    if not (root/'ABORT').exists():
        raise ValueError('kill requires persistent fence')
    # Every possible launcher has either passed guard before ABORT (listed), or
    # will see ABORT and exit 75. Do not clear failed units to manufacture proof.
    for unit in units(tag):
        try:
            checked(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', unit])
        except ValueError:
            # Another killed scope can finish this one while systemctl is in
            # flight. Accept only independently proved disappearance, never
            # swallow a transport/permissions failure for a live cgroup.
            if unit in units(tag) or any(populated(p) for p in paths(root, tag) if p.name == unit):
                raise
    until = time.monotonic() + 10
    while True:
        if not units(tag) and not any(populated(p) for p in paths(root, tag)):
            prove(root, tag)
            print(ABSENT, flush=True)
            return
        if time.monotonic() >= until:
            raise ValueError('owned scope/cgroup absence unproved; HOLD stays open')
        time.sleep(.02)


def fence_all(tag: str, invoke: Invoke, hosts: Sequence[str] = NODES) -> None:
    """Write on ALL destinations before killing ANY, then prove ALL absent.

    invoke must fail on transport errors; a missing receipt is also fatal. A
    caller may add localhost for locally scoped W8. Retain HOLD on any error.
    """
    for mode, marker in (('fence', FENCED), ('reap', ABSENT), ('absent', ABSENT)):
        errors = []
        for host in hosts:
            try:
                rc, text = invoke(host, command(mode, tag))
                if rc != 0 or marker not in text.splitlines():
                    errors.append(f'{host}/{mode}: rc{rc}, required {marker!r}: {text[-500:]}')
            except Exception as exc:
                errors.append(f'{host}/{mode}: {type(exc).__name__}: {exc}')
        if errors:
            raise ValueError('; '.join(errors) + '; HOLD stays open')


def install(root: Path, tag: str) -> Path:
    """Called only after the in-scope fence check while holding its lock."""
    script = root/'owner.py'
    code = source()
    if script.exists() and script.read_text() != code:
        raise ValueError('scoped owner source drift')
    if not script.exists():
        script.write_text(code)
    bindir = root/'bin'
    bindir.mkdir(mode=0o700, exist_ok=True)
    wrappers = {
        'ssh': ('#!' + sys.executable + '\nimport os,sys\n'
                f'os.execv(sys.executable,[sys.executable,{str(script)!r},"ssh",{tag!r},*sys.argv[1:]])\n'),
        'tmux': ('#!' + sys.executable + '\nimport os,sys\n'
                 f'os.execv(sys.executable,[sys.executable,{str(script)!r},"tmux",{tag!r},*sys.argv[1:]])\n'),
        'scp': ('#!' + sys.executable + '\nimport os,sys\n'
                f'os.execv("/usr/bin/scp",["/usr/bin/scp","-O","-S",{str(bindir/"ssh")!r},*sys.argv[1:]])\n'),
    }
    if tag.endswith('-restore-close'):
        wrappers['rm'] = ('#!' + sys.executable + '\nimport os,sys\n'
                          f'os.execv(sys.executable,[sys.executable,{str(script)!r},"rm",{tag!r},*sys.argv[1:]])\n')
    for name, text in wrappers.items():
        path = bindir/name
        if path.exists() and path.read_text() != text:
            raise ValueError('scoped shim drift: ' + name)
        if not path.exists():
            path.write_text(text)
            path.chmod(0o700)
    return bindir


def ssh(tag: str, args: list[str]) -> None:
    options: list[str] = []
    while args and args[0].startswith('-'):
        opt = args.pop(0)
        if opt == '--':
            break
        if opt in ('-o', '-p', '-l', '-i', '-F', '-J') and args:
            options.extend((opt, args.pop(0)))
        elif opt in ('-q', '-n', '-T', '-4', '-6', '-x', '-a', '-v') or opt.startswith('-o') and len(opt) > 2:
            options.append(opt)
        else:
            raise ValueError('unsupported scoped nested SSH option: ' + opt)
    if len(args) < 2 or args[0] not in HOSTS:
        raise ValueError('scoped SSH requires reviewed destination and command')
    host, *parts = args
    remote = command('launch', tag, ['bash', '-c', ' '.join(parts)], deadline=expiry())
    os.execv(SSH_BINARY, [SSH_BINARY, *options, host, remote])


def launch(tag: str, op: str, child: list[str]) -> None:
    if re.fullmatch('[0-9a-f]{32}', op) is None or not child:
        raise ValueError('invalid scoped operation/command')
    deadline = expiry()
    # --collect lets systemd unload failed/expired scopes AFTER termination;
    # recorded cgroup.events still independently proves transitive emptiness.
    # --scope inherits environment and cgroup descendants. KillSignal=KILL
    # gives a server-side ceiling even if SSH/local supervision disappears.
    argv = ['systemd-run', '--user', '--scope', '--collect', '--quiet', '--unit=dsv41w-' + tag + '-' + op,
            '--property=KillMode=control-group', '--property=KillSignal=SIGKILL',
            f'--property=RuntimeMaxSec={max(.001, deadline-time.time()):.6f}',
            'python3', '-c', 'SOURCE = ' + repr(source()) + '\nexec(compile(SOURCE,"<scoped-guard>","exec"))',
            'guard', tag, op, *child]
    os.execvpe(argv[0], argv, os.environ)



def watch(tag: str, op: str) -> None:
    """Kill this exact cgroup at the immutable wall deadline, independent of PGIDs."""
    if re.fullmatch('[0-9a-f]{32}', op) is None:
        raise ValueError('invalid scoped guardian identity')
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    root = root_for(tag)
    unit = f'dsv41w-{tag}-{op}.scope'
    value = Path('/proc/self/cgroup').read_text().strip().removeprefix('0::')
    saved = root/(op+'.cgroup')
    if saved.read_text().strip() != value or Path(value).name != unit:
        raise ValueError('guardian is outside its exact owned cgroup')
    group = CGROUP / value.lstrip('/')
    deadline = float(os.environ['SCOPED_REMOTE_DEADLINE'])
    if not math.isfinite(deadline) or float((root/'deadline').read_text()) != deadline:
        raise ValueError('guardian immutable deadline differs')
    while time.time() < deadline:
        # Unlike a PGID registry, this includes setsid and timeout descendants.
        files = [group/'cgroup.procs', *group.glob('**/cgroup.procs')]
        members: set[int] = set()
        for path in files:
            try:
                members.update(int(pid) for pid in path.read_text().split())
            except FileNotFoundError:
                continue  # subgroup disappeared; the parent scope still owns us
        if not members - {os.getpid()}:
            return
        time.sleep(min(.02, max(0, deadline-time.time())))
    checked(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', unit])
    raise ValueError('guardian unexpectedly survived cgroup KILL')

def main(argv: list[str] | None = None) -> int:
    mode, tag, *args = sys.argv[1:] if argv is None else argv
    root = root_for(tag)
    if mode == 'watch':
        watch(tag, args[0])
        return 0
    if mode == 'tmux':
        if any(value in ('dsv41-hold', '=dsv41-hold') for value in args):
            if any(value in ('-L', '-S') for value in args):
                raise ValueError('placeholder tmux socket override refused')
            expiry()
            # Never delegate to an existing shared daemon outside our cgroup.
            # Its private server is born here and dies with this phase scope.
            socket = 'dsv41w-' + hashlib.sha256(tag.encode()).hexdigest()[:32]
            # tmux 3.6 moves panes into its own systemd scopes by default.
            # Deny its private server a bus connection so its ordinary fork
            # fallback retains our cgroup. Only sleep placeholders inherit
            # this environment; the owner/guardian retain the real user bus.
            os.execve(TMUX_BINARY, [TMUX_BINARY, '-f', '/dev/null', '-L', socket, *args],
                      {**os.environ, 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/dev/null'})
        os.execv(TMUX_BINARY, [TMUX_BINARY, *args])
    if mode == 'ssh':
        ssh(tag, args)
        return 0
    if mode == 'launch':
        launch(tag, args[0], args[1:])
        return 0
    # In-scope first action checks the fence, BEFORE mkdir, shim install or any
    # business command. Recheck under the registration lock against fencing.
    if mode == 'guard' and (root/'ABORT').exists():
        return 75
    if mode == 'rm' and (args != ['-f', str(Path.home()/'dsv41-deploy/HOLD')]
                         or not tag.endswith('-restore-close')):
        os.execv('/usr/bin/rm', ['/usr/bin/rm', *args])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if mode == 'fence':
            with (root/'ABORT').open('a') as handle:
                handle.write('permanent window fence\n')
                handle.flush()
                os.fsync(handle.fileno())
            print(FENCED, flush=True)
            return 0
        if mode == 'guard':
            if (root/'ABORT').exists():
                return 75
            deadline = expiry()
            lease = root/'deadline'
            if lease.exists() and float(lease.read_text()) != deadline:
                raise ValueError('immutable scoped deadline changed')
            if not lease.exists():
                lease.write_text(str(deadline))
            op, *child = args
            if re.fullmatch('[0-9a-f]{32}', op) is None or not child:
                raise ValueError('invalid scoped guard command')
            lines = Path('/proc/self/cgroup').read_text().splitlines()
            if len(lines) != 1 or not lines[0].startswith('0::'):
                raise ValueError('cgroup v2 required')
            value = lines[0][3:]
            if Path(value).name != f'dsv41w-{tag}-{op}.scope':
                raise ValueError('guard did not enter its own scope')
            (root/(op+'.cgroup')).write_text(value)
            paths(root, tag)  # validate this user's manager path before mutation
            bindir = install(root, tag)
            # A scope's RuntimeMaxSec cannot be changed after creation (even
            # systemd 259 rejects it). An in-cgroup guardian enforces the
            # absolute deadline; the creation-time cap is a second backstop.
            with (root/(op+'.watch.log')).open('x') as log:
                subprocess.Popen([sys.executable, str(root/'owner.py'), 'watch', tag, op],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log)
            expiry()
            os.execvpe(child[0], child, {**os.environ, 'PATH': str(bindir)+os.pathsep+os.environ['PATH']})
        if mode == 'rm':
            expiry()
            if (root/'ABORT').exists():
                raise ValueError('close fenced; HOLD stays open')
            hold = Path.home()/'dsv41-deploy/HOLD'
            text = hold.read_text()
            # Numbered close transactions share the original HOLD owner, but
            # never share a fence, deferred unlink, or commit/cancel record.
            close_owner = os.environ.get('SCOPED_CLOSE_OWNER', tag.removesuffix('-restore-close'))
            if re.search(r'(?:^|\s)' + re.escape(close_owner) + ':', text) is None:
                raise ValueError('HOLD belongs to another window')
            (root/'CLOSE_HOLD').write_text(text)
            return 0
        if mode == 'cancel-close':
            (root/'COMMIT_ABORT').touch()
            print('SCOPED CLOSE COMMIT FENCED', flush=True)
            return 0
        if mode == 'retain-close':
            (root/'COMMIT_ABORT').touch()
            hold = Path.home()/'dsv41-deploy/HOLD'
            original = (root/'CLOSE_HOLD').read_text()
            if hold.exists():
                if hold.read_text() != original:
                    raise ValueError('another HOLD exists; refusing overwrite')
            else:
                with hold.open('x') as handle:
                    handle.write(original)
            print(RETAINED, flush=True)
            return 0
        if mode == 'commit-close':
            expiry()
            prove(root, tag)
            if (root/'COMMITTED').exists() or (root/'COMMIT_ABORT').exists():
                raise ValueError('close already committed/cancelled')
            hold = Path.home()/'dsv41-deploy/HOLD'
            if hold.read_text() != (root/'CLOSE_HOLD').read_text():
                raise ValueError('HOLD changed; refusing removal')
            expiry()
            (root/'COMMITTED').touch()
            hold.unlink()
            print(CLOSED, flush=True)
            return 0
    if mode == 'reap':
        reap(root, tag)
    elif mode == 'absent':
        prove(root, tag)
        print(ABSENT, flush=True)
    else:
        raise ValueError('unknown scoped operation: ' + mode)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
