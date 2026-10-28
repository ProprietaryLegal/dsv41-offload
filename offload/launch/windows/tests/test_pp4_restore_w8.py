"""PP4 production W8: parses, checks the NEW references, and speaks post_close_watch.restore_verified's contract."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from offload.launch.windows import post_close_watch as watch
from offload.launch.windows import production_health
from offload.launch.windows import pp4_promote_switch as switch

SCRIPT = Path(__file__).resolve().parents[1] / 'restore-pp4.sh'
TEXT = SCRIPT.read_text()


def test_parses_and_pins_this_pp4_production() -> None:
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0
    assert f'ID={switch.PP4_ID}\n' in TEXT and f'DSPARK_SOURCE_COMMIT={switch.PP4_COMMIT}' in TEXT
    assert f"{switch.DROPIN.split('/', 3)[-1]}" in TEXT


def test_compares_against_the_pp4_references_only() -> None:
    assert 'REF=$C/K1-production/greedy-post-pp4.jsonl' in TEXT
    assert '--compare "$REF"' in TEXT and 'greedy-post.jsonl' not in TEXT.replace('greedy-post-pp4.jsonl', '')


def test_restore_verified_accepts_exactly_the_lines_it_prints() -> None:
    start = re.search(r"q '(START: [^']*)'", TEXT)
    close = re.search(r"q '(CLOSE: [^']*)'", TEXT)
    assert start is not None and close is not None
    watch.restore_verified('start', 0, start.group(1))
    watch.restore_verified('close', 0, close.group(1))
    prefix = re.search(r"sed 's/\^/(greedy K1-post vs restored: )/'", TEXT)
    assert prefix is not None
    check = '\n'.join(f'{prefix.group(1)}{i} IDENTICAL 100' for i in range(6)) + '\nPRODUCTION HOLD OK\nCHECK: DONE\n'
    watch.restore_verified('check', 0, check)
    with pytest.raises(production_health.WatchError):
        watch.restore_verified('check', 0, check.replace('5 IDENTICAL', '5 DIVERGES'))


def body(name: str) -> str:
    return TEXT.split(f'\n{name}() {{', 1)[1].split('\n}\n', 1)[0]


def test_start_and_check_require_both_verifications() -> None:
    start, check = body('start'), body('check')
    for marker in ("grep -q 'K1 START VERIFIED'", "grep -qx 'PP4 RUNNING VERIFIED'", 'pp4_worker',
                   "grep -qx 'PRODUCTION HOLD OK'"):
        assert marker in start, marker
    # S91 #5: check verifies the CURRENT configuration, never the latest historical guard log.
    for marker in ('pp4_promote_switch.py active', 'manage.py verify', 'bash offload/promote/k1/verify.sh',
                   'grep -qx "PP4 ACTIVE $ID"', "grep -qx 'PP4 RUNNING VERIFIED'", "grep -q '^K1 runtime evidence: '",
                   'containers >', 'pp4_worker', "grep -qx 'PRODUCTION HOLD OK'"):
        assert marker in check, marker
    assert 'k1-guard' not in TEXT
