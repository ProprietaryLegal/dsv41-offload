"""PF2-LOWKV runner, item 3: pure checks, the telemetry/sampler scripts on this CPU host, the cell driver with fakes and
the two PF1 hooks. No ssh, HTTP or service."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pf2_window as p
from offload.launch.windows.tests import test_pf1_window as base
from offload.pf2_lowkv import arith
from offload.pf2_lowkv import cells as pc
from offload.prefill_maxlen.evidence import Json
from offload.prefill_trace.window_requests import HEADER

PLATFORM = p.Platform({'sparkhead': '7.0.0-1004-nvidia', 'sparkworker-wired': '6.17.0-1026-nvidia'},
                      {'sparkhead': '580.178.02', 'sparkworker-wired': '580.159.03'})


def parity(kernel: str, driver: str, available: int = 9_000_000) -> str:
    return (f'{kernel}\n{driver}\nalways [madvise] never\nalways defer defer+madvise [madvise] never\n'
            'Node 0, zone   Normal  1 2 3\ncompact_stall 10\ncompact_fail 2\ncompact_success 8\n'
            f'MemAvailable:   {available} kB\n')


def test_arms_alternate_h_l_with_the_checked_low() -> None:
    plan = p.arms(arith.LOW_KV)
    assert [a.name for a in plan] == ['s1-H', 's2-L', 's3-H', 's4-L']
    assert [a.kv for a in plan] == [arith.H_KV, arith.LOW_KV] * 2
    assert all(a.cap == 524288 and not a.trace and not a.eager and a.config == p.CONFIG for a in plan)
    with pytest.raises(ValueError):
        p.arms(2 * arith.GIB)


def test_strip_kv_normalises_exactly_one_reservation() -> None:
    argv = "vllm serve /m --port 8001 \\\n --max-model-len 524288 --kv-cache-memory-bytes 3221225472 --x '{\"a\":1}'"
    got = shlex.split(p.strip_kv(argv, arith.LOW_KV))
    assert got[got.index('--kv-cache-memory-bytes') + 1] == str(arith.H_KV) and got[-1] == '{"a":1}'
    with pytest.raises(ValueError, match="arm's"):
        p.strip_kv(argv, arith.H_KV)
    with pytest.raises(ValueError, match='exactly one'):
        p.strip_kv('vllm serve /m', arith.H_KV)
    with pytest.raises(ValueError, match='exactly one'):
        p.strip_kv(argv + ' --kv-cache-memory-bytes 1', arith.LOW_KV)


def test_host_identity_binds_the_parameters_not_constants() -> None:
    row = p.host_identity(parity('7.0.0-1004-nvidia', '580.178.02'), 'sparkhead', PLATFORM)
    assert row['kernel'] == '7.0.0-1004-nvidia' and row['thp_enabled'] == 'always [madvise] never'
    assert row['mem_available_kb'] == 9_000_000 and row['compact'] == {'compact_stall': 10, 'compact_fail': 2, 'compact_success': 8}
    with pytest.raises(ValueError, match='kernel'):
        p.host_identity(parity('6.17.0-1026-nvidia', '580.178.02'), 'sparkhead', PLATFORM)
    with pytest.raises(ValueError, match='driver'):
        p.host_identity(parity('7.0.0-1004-nvidia', '580.159.03'), 'sparkhead', PLATFORM)
    with pytest.raises(ValueError, match='MemAvailable'):
        p.host_identity('7.0.0-1004-nvidia\n580.178.02\na\nb\n', 'sparkhead', PLATFORM)


def test_platform_parameters_are_validated() -> None:
    PLATFORM.validate()
    for bad in (p.Platform({'sparkhead': 'x', 'sparkworker-wired': '6.17.0-1026-nvidia'}, PLATFORM.driver),
                p.Platform(PLATFORM.kernel, {'sparkhead': '580', 'sparkworker-wired': '580.159.03'}),
                p.Platform({'sparkhead': '7.0.0-1004-nvidia'}, {'sparkhead': '580.178.02'})):
        with pytest.raises(ValueError):
            bad.validate()


def test_extended_telemetry_script_runs_here_and_keeps_the_guard_row_shape() -> None:
    command = p.telemetry_command(1)
    assert command.startswith('exec python3 -u -c ')
    argv = shlex.split(command)[1:]
    out = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, timeout=30, check=True).stdout
    rows = [json.loads(x) for x in out.splitlines()]
    assert rows[0]['kind'] == 'memory' and rows[1]['kind'] in ('gpu', 'gpu_error')
    assert pf.memory_row(rows[0])[0] > 0
    assert set(p.VMSTAT) - {'thp_collapse_alloc'} <= set(rows[0]['vmstat'])
    if Path('/proc/pressure/memory').exists():
        assert {'some', 'full'} <= set(rows[0]['psi'])
    probe = [rows[0], {'kind': 'gpu', 'csv': 'GPU-x,2100,4000,50.1,100,60,90,0x0'}]
    rows[0]['vmstat'].setdefault('thp_collapse_alloc', 0)
    rows[0]['vmstat'].setdefault('allocstall_normal', 0)
    rows[0]['psi'] = rows[0]['psi'] or {'some': {}, 'full': {}}
    assert 'vmstat' in p.telemetry_probe(probe)
    with pytest.raises(ValueError, match='PSI'):
        p.telemetry_probe([{**rows[0], 'psi': None}, probe[1]])
    with pytest.raises(ValueError, match='vmstat lacks'):
        p.telemetry_probe([{**rows[0], 'vmstat': {}}, probe[1]])


def test_procmem_sampler_roles_and_one_live_sample() -> None:
    ns: dict[str, Any] = {}
    exec(p.PROCMEM.split('while True:')[0], ns)
    role = ns['role']
    assert role('VLLM::EngineCore') == 'engine_core' and role('VLLM::Worker_TP1') == 'worker'
    assert role('/usr/bin/python3 /usr/local/bin/vllm serve /srv/dsv41/legal-v2a --port 8001') == 'api_server'
    assert role('python3 -m offload.something') is None
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        subprocess.run([sys.executable, '-u', '-c', p.PROCMEM], capture_output=True, text=True, timeout=1.5)
    out = caught.value.stdout
    text = out.decode() if isinstance(out, bytes) else (out or '')
    rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
    assert rows and rows[0]['kind'] == 'procmem' and isinstance(rows[0]['procs'], list)
    assert 'cmdline' not in text and '/srv/' not in text            # numbers only


def test_reap_and_hold_text_match_each_other() -> None:
    live = object.__new__(p.PF2Live)
    live.tag, live.low = 'PF2-W-20261027T000000Z', arith.LOW_KV
    reap = live.reap_script()
    assert 'PF1' not in reap and 'SP1' not in reap and reap.count('PF2-lowkv[.]env') == 2
    opener = re.search(r"reap '(window\[\.\]sh open PF2-LOWKV [^']+)'", reap)
    assert opener is not None and re.search(opener[1], f'offload/launch/window.sh open {live.hold_purpose()}')
    assert re.search(r'start_exp[.]sh offload/launch/configs/PF2-lowkv[.]env', f'bash offload/launch/start_exp.sh {p.CONFIG}')


def test_hooks_pass_the_kv_value_and_expected_pool() -> None:
    live = object.__new__(p.PF2Live)
    low = p.arms(arith.LOW_KV)[1]
    assert live.arm_env(low) == f' PF2_KV_BYTES={arith.LOW_KV}' and live.expected_kv(low) == 1_410_348
    plain = object.__new__(pf.Live)
    off = pf.Arm('off-524', 524288, 0)
    assert plain.arm_env(off) == '' and plain.expected_kv(off) == pf.KV[524288] == 2_820_756


def test_logs_kv_override_and_default() -> None:
    summary = base.log_summary(524288, kv_tokens=[1_410_348])
    assert pf.logs(summary, 0, 524288, kv=1_410_348) is summary
    with pytest.raises(ValueError, match='KV capacity is not 2820756'):
        pf.logs(summary, 0, 524288)
    pf.logs(base.log_summary(524288), 0, 524288)
    pf.logs(base.log_summary(327680), 0, 327680, production=True)


def test_dry_run_and_refusals(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert p.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith('DRY RUN') and 'PF2_KV_BYTES=3221225472' in out and 'needle-480k' in out and 'restore.sh start' in out
    with pytest.raises(SystemExit):
        p.main(['--low-kv-bytes', str(2 * arith.GIB)])
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    assert p.main(['--execute', '--out', str(pf.CAMPAIGN / 'PF2-W-new'), '--expected-experiment-commit', 'a' * 40]) == 1
    assert 'STOP before any remote action' in capsys.readouterr().err       # no platform parameters
    flags = ['--sparkhead-kernel', '7.0.0-1004-nvidia', '--sparkhead-driver', '580.178.02',
             '--sparkworker-kernel', '6.17.0-1026-nvidia', '--sparkworker-driver', '580.159.03']
    monkeypatch.delenv('WINDOW_COORDINATOR')
    assert p.main(['--execute', *flags]) == 1
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    assert p.main(['--execute', *flags, '--out', '/mnt/data/elsewhere/x', '--expected-experiment-commit', 'a' * 40]) == 1
    assert not (pf.CAMPAIGN / 'PF2-W-new').exists()


# ---------------------------------------------------------------- cell driver (fake API)

class Api:
    def __init__(self) -> None:
        self.calls = 0

    def tokens(self, text: str) -> int:
        self.calls += 1
        return int(len(text) / 3.3 * .99)

    def metrics(self) -> str:
        return 'vllm:num_requests_running{m="x"} 0\nvllm:num_requests_waiting{m="x"} 0\n'


def test_build_and_rebuild_only_the_two_prompts() -> None:
    materials = HEADER + 'abcdefghij' * 1_000_000
    cuts = pc.build(cast(Any, Api()), materials, {'c': 'h'})
    assert set(cast(dict[str, Json], cuts['prompts'])) == {'warm-32k', '150k'}
    texts = pc.texts_from(cuts, materials, {'c': 'h'})
    assert texts['150k'][1] <= 150000 and 'abc' in texts['150k'][0]
    with pytest.raises(ValueError, match='corpus changed'):
        pc.texts_from(cuts, materials, {'c': 'other'})


def test_cells_are_sp1_cells_at_524288_with_one_mid_prompt_capture() -> None:
    assert pc.ORDER == ('warmup', 'c150-a', 'c150-b', 'p150')
    assert [c.captures for c in pc.CELLS.values()] == [(), (), (), ('prefill-late',)]
    assert all(c.prompts in (('warm-32k',), ('150k',)) for c in pc.CELLS.values())
    assert pc.CAP == arith.CAP and 20 <= pc.MID_DELAY_S <= 50


class NeedleApi:
    def __init__(self, answer: str = '', counts: int = 0) -> None:
        self.answer, self.counts, self.asked = answer, counts, 0

    def count(self, text: str) -> int:
        self.counts += 1
        return int(len(text) / 3.3)

    def ask(self, prompt: str, salt: str) -> dict[str, Json]:
        self.asked += 1
        codes = re.findall(r'The vault code for (\w+) is (\d+)\.', prompt)
        text = self.answer or '\n'.join(f'{n}: {c}' for n, c in codes)
        return {'text': text, 'elapsed_s': 300.0, 'usage': {'prompt_tokens': 479_000, 'completion_tokens': 40,
                                                              'prompt_tokens_details': {'cached_tokens': 0}}}

    def metrics(self) -> str:
        return Api().metrics()


DOCS = [{'title': f't{i}', 'court': 'c', 'date': 'd', 'text': 'word ' * 50_000} for i in range(40)]


def test_needle_records_counts_and_hashes_only(tmp_path: Path) -> None:
    api = NeedleApi()
    row = pc.run_needle(api, DOCS, 1e12, 'tag', ledger=tmp_path / 'ledger.jsonl')
    assert row['status'] == 'RAN' and row['recall'] == row['of'] == 6 and row['recalled_all'] is True
    assert 'answer' not in row and 'text' not in row and re.fullmatch('[0-9a-f]{64}', str(row['answer_sha256']))
    assert cast(int, row['count_calls']) <= pc.MAX_NEEDLE_COUNTS and api.asked == 1
    events = [json.loads(x)['event'] for x in (tmp_path / 'ledger.jsonl').read_text().splitlines()]
    assert events == ['send', 'completed']
    wrong = pc.run_needle(NeedleApi(answer='Beaufort: 123456'), DOCS, 1e12, 'tag')
    assert wrong['recall'] == 0 and wrong['recalled_all'] is False


def test_needle_skips_on_time_and_bounds_tokenize() -> None:
    api = NeedleApi()
    assert pc.run_needle(api, DOCS, 0.0, 'tag')['status'] == 'SKIPPED_TIME' and api.counts == 0
    bounded = pc.Bounded(DOCS, lambda text: 10**9)          # never fits: the builder keeps trimming
    with pytest.raises(RuntimeError, match='/tokenize calls'):
        bounded.build(pc.NEEDLE_CTX, pc.NEEDLES, pc.NEEDLE_SEED)
    assert bounded.calls == pc.MAX_NEEDLE_COUNTS + 1


def test_cell_driver_plan_without_execute(capsys: pytest.CaptureFixture[str]) -> None:
    assert pc.main(['build', '--output', '/nonexistent/x']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['executed'] is False and plan['cap'] == 524288 and plan['needle']['ctx'] == 480000
