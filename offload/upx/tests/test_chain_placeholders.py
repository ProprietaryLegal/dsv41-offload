"""WIN2-R2 hold placeholders under the scoped owner, with REAL user scopes on Controller (no Spark, no network).

Runs inside test_scoped_remote's sandbox: a fake ssh transport maps sparkhead/spark -> one private HOME and
sparkworker-wired/sparkworker -> another, each hop a separate real user scope; TMPDIR and TMUX_TMPDIR must be private and outside
/tmp (the sandbox fixture refuses otherwise), and TMUX must be unset.

What it pins (finding R-UPX 2026-10-28, reported to the coordinator): a runner that opens its HOLD through
scoped_remote (every NBW/PF1-lineage runner) gets its dsv41-hold placeholders from scoped_remote's tmux shim, which
puts them on a PRIVATE socket inside the hold phase's scopes. So (a) the default-socket probe that chain.preflight and
chain.gate use does not see them, (b) chain.CHAINED_CLOSE run inside the window's -restore-close scope, as WIN2.md
prescribes, checks a different private socket and refuses with the HOLD untouched, and (c) fencing the hold phase kills
them. The UPX runner therefore checks both views before a chained close and restores production when either fails.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from offload.launch.windows import chain, hold_admission
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests.test_scoped_remote import Sandbox, sandbox, wait_for
from offload.upx import battery_window as bw

__all__ = ['sandbox']       # the shared fixture, re-exported for pytest

NAME = 'upx-test placeholder'


def placeholder_loop(tag: str) -> str:
    """open_script's placeholder loop exactly, after the admission step, with a harmless process name and lifetime."""
    token = '0' * 32
    admit = hold_admission.command('admit', tag, token, ['purpose', '30', 'allow-unreachable'])
    script = hold_admission.open_script(tag, token, 'purpose', 30, allow_unreachable=True)
    head, sep, loop = script.partition(admit + '; ')
    assert sep and head == 'set -euo pipefail; cd ~/dsv41-deploy; '
    renamed = loop.replace('vllm serve [DSV41-OFFLOAD hold]', NAME).replace('sleep 86400', 'sleep 60')
    assert renamed != loop and NAME in renamed
    return head + renamed


def running() -> int:
    found = subprocess.run(['pgrep', '-f', '-c', f'^{NAME} 60$'], capture_output=True, text=True)   # the sleep itself
    return int(found.stdout.strip() or 0)


def test_scoped_placeholders_are_invisible_to_the_chain_probe_and_to_chained_close(sandbox: Sandbox) -> None:
    s = sandbox
    window = s.tag('upxph')
    hold_tag, close_tag = window + '-hold', window + '-restore-close'
    for tag in (hold_tag, close_tag):
        s.tags.add(tag)
        s.deadlines[tag] = time.time() + 25
    deploy = s.homes['sparkhead'] / 'dsv41-deploy'
    deploy.mkdir()
    (deploy / 'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\n')
    hold = deploy / 'HOLD'
    hold.write_text(f'2026-10-28T00:00:00Z DSV41-OFFLOAD (session coordinator): UPX-0085B {window}: test\n')
    ready = s.root / 'placeholders-ready'
    s.launch('new', hold_tag, placeholder_loop(hold_tag) + '; touch ' + shlex.quote(str(ready)))
    wait_for(ready, 20)
    # Created through the scoped ssh and tmux shims. The sandbox's two fake hosts share one TMUX_TMPDIR, so the hold
    # tag's private socket is one path and the second Spark's has-session finds the first one's session.
    assert running() == 1

    # (a) the default-socket view (chain.PLACEHOLDER_PROBE's tmux test, as chain.preflight / chain.gate run it)
    for host in ('sparkhead', 'sparkworker-wired'):
        probe = s.run('tmux has-session -t dsv41-hold', host)
        assert probe.returncode != 0, 'the scoped placeholder is on a private socket'

    # (b) chain.CHAINED_CLOSE inside the window's own -restore-close scope, as WIN2.md prescribes
    closed = s.run(owner.command('launch', close_tag, ['bash', '-c', chain.CHAINED_CLOSE], deadline=s.deadlines[close_tag]))
    assert closed.returncode != 0 and 'hold placeholder missing on spark' in closed.stdout, closed.stdout + closed.stderr
    assert hold.read_text().endswith(': test\n'), 'the HOLD is untouched'
    assert not (s.homes['sparkhead'] / owner.CONTROL / close_tag / 'CLOSE_HOLD').exists()
    # the runner's own pre-check, built from CHAINED_CLOSE minus its unlink, reaches the same answer
    probe_tag = window + '-restore-placeholders'
    s.tags.add(probe_tag)
    s.deadlines[probe_tag] = time.time() + 25
    pre = s.run(owner.command('launch', probe_tag, ['bash', '-c', bw.PLACEHOLDER_CHECK], deadline=s.deadlines[probe_tag]))
    assert pre.returncode != 0 and bw.PLACEHOLDERS_OK not in pre.stdout.splitlines()

    # Control: an UNSCOPED default-socket placeholder (window.sh holdproc style) is what the probe sees; even then the
    # scoped CHAINED_CLOSE and the runner's scoped pre-check still look at their own private sockets and refuse.
    unscoped = s.run('tmux new-session -d -s dsv41-hold ' + shlex.quote(f'exec -a "{NAME}" sleep 60'))
    try:
        assert unscoped.returncode == 0, unscoped.stderr
        assert s.run('tmux has-session -t dsv41-hold').returncode == 0, 'the probe method does see a default-socket session'
        probe_tag2 = window + '-restore-placeholders2'
        s.tags.add(probe_tag2)
        s.deadlines[probe_tag2] = time.time() + 25
        pre = s.run(owner.command('launch', probe_tag2, ['bash', '-c', bw.PLACEHOLDER_CHECK],
                                  deadline=s.deadlines[probe_tag2]))
        assert pre.returncode != 0 and 'hold placeholder missing on spark' in pre.stdout
    finally:
        s.run('tmux kill-session -t dsv41-hold')

    # (c) fencing the hold phase (every runner does before its commit-close) takes the placeholders down
    s.close('new', hold_tag)
    until = time.monotonic() + 5
    while running() and time.monotonic() < until:
        time.sleep(.05)
    assert running() == 0
