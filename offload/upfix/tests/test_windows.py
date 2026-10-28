from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import decision_v2 as dv
from offload.prefill_maxlen.evidence import Json
from offload.upfix import configs
from offload.upfix import windows as w
from offload.upfix.topk import bitcompare


@pytest.mark.parametrize('name', sorted(w.WINDOWS))
def test_every_design_meets_win2(name: str) -> None:
    w.validate(w.WINDOWS[name])


def test_estimated_holds_are_stable() -> None:
    assert {n: (w.hold_minutes(x), w.hold_minutes(x, last=True)) for n, x in w.WINDOWS.items()} == {
        'UPF-W1': (143, 165), 'UPF-0091': (96, 118), 'UPF-TOPK': (94, 116)}
    assert len(bitcompare.cases()) == 149


def test_validate_refuses_short_repeats_missing_gates_and_unbracketed_controls() -> None:
    window = w.WINDOWS['UPF-0091']
    thin = dataclasses.replace(window.starts[1], steps=(w.WARMUP, w.Step('greedy'), w.Step('needle-150k'),
                                                        *w.cells(2), w.Step('soak')))
    with pytest.raises(ValueError, match='< 3 repeats'):
        w.validate(dataclasses.replace(window, starts=(window.starts[0], thin, window.starts[2])))
    no_soak = dataclasses.replace(window.starts[1], steps=window.starts[1].steps[:-1])
    with pytest.raises(ValueError, match=r"\['soak'\] missing"):
        w.validate(dataclasses.replace(window, starts=(window.starts[0], no_soak, window.starts[2])))
    with pytest.raises(ValueError, match='bracket'):
        w.validate(dataclasses.replace(window, starts=(window.starts[1], window.starts[0], window.starts[2])))
    topk_without_pre = dataclasses.replace(w.WINDOWS['UPF-TOPK'], pre=())
    with pytest.raises(ValueError, match='bitcompare'):
        w.validate(topk_without_pre)
    no_stop = dataclasses.replace(w.WINDOWS['UPF-TOPK'], pre=w.TOPK_PRE[1:])
    with pytest.raises(ValueError, match='stop_idle'):
        w.validate(no_stop)
    assert [s.kind for s in w.TOPK_PRE] == ['stop_idle', 'jit_build', 'bitcompare']


def decision(control_ms: float, candidate_ms: list[float]) -> dict[str, Json]:
    series = []
    for cell in w.DECISION_CELLS:
        for metric, _ in w.METRICS[cell]:
            control = [control_ms, control_ms * 1.01, control_ms * 0.99, control_ms * 1.005] if metric == 'ms_per_round' \
                else [3.0, 3.02, 2.98, 3.01] if metric == 'accepted_per_round' else [69.0, 69.5, 68.9, 69.2]
            candidate = candidate_ms if metric == 'ms_per_round' else control[:3]
            series.append(w.series(cell, metric, control, candidate))
    return dict(dv.decide_window(series, w.DECISION_CELLS, groups=w.GROUPS))


def test_no_loss_semantics() -> None:
    gates = {g: True for g in w.REQUIRED_GATES['upf-0091']}
    same = decision(35.0, [35.1, 34.9, 35.05])
    assert same['status'] == 'INCONCLUSIVE'
    assert w.arm_verdict('upf-0091', same, gates) == 'PROMOTABLE (no loss)'
    worse = decision(35.0, [37.0, 37.2, 36.9])
    assert worse['status'] == 'NOT A CANDIDATE'
    assert w.arm_verdict('upf-0091', worse, gates) == 'NOT A CANDIDATE'
    assert w.arm_verdict('upf-0091', same, {**gates, 'soak': False}) == 'STOP: soak'
    with pytest.raises(ValueError, match='required'):
        w.arm_verdict('topk-fix', same, gates)
    with pytest.raises(ValueError, match='win2-v1'):
        w.arm_verdict('upf-0091', {**same, 'rule': 'v1'}, gates)


def test_soak_gate() -> None:
    off = {'head:EngineCore': 4.0, 'head:worker0': 30.0, 'worker:worker1': 12.0}
    assert w.soak_ok(off, {'head:EngineCore': 15.0, 'head:worker0': 59.0, 'worker:worker1': 20.0})
    assert not w.soak_ok(off, {'head:EngineCore': 17.0, 'head:worker0': 10.0, 'worker:worker1': 1.0})
    assert not w.soak_ok(off, {'head:EngineCore': 1.0, 'head:worker0': 61.0, 'worker:worker1': 1.0})
    with pytest.raises(ValueError, match='roles differ'):
        w.soak_ok(off, {'head:EngineCore': 1.0})


def test_greedy_threshold_is_ndm_0_9() -> None:
    assert w.GREEDY_NEAR_TIE_NATS == pytest.approx(0.901)


def test_t2_command_is_bounded_and_cleans_up() -> None:
    text = (configs.HERE / 'WINDOW.md').read_text()
    block = text.split('# T2, one GB10 GPU', 1)[1].split('```', 1)[0]
    runs = re.findall(r'^(.*)docker run (.*)$', block, re.M)
    assert len(runs) == 1
    prefix, args = runs[0]
    assert prefix == f'timeout -k 30 {w.T2_TIMEOUT_S} ' and args.startswith('--rm --name upfix-t2 ')
    assert 'case $t2 in 124|137) docker rm -f upfix-t2' in block
    assert w.GATE_S['bitcompare'] < w.T2_TIMEOUT_S
    assert 'at the kernel level at both caps' in text and 'at the 327,680 cap only' in text


PROD = 'a' * 40


def fake_window(tmp_path: Path, caps: tuple[int, int] = (2150, 2150), name: str = 'UPF-W1-W-20261028T010203Z') -> Path:
    out = tmp_path / name
    (out / 'cmd').mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps({'production_commit': PROD, 'experiment_commit': 'b' * 40}))
    for i, start in enumerate(('C1', 'C2')):
        (out / 'cmd' / f'000{i}-{start}-start.log').write_text(
            f'spark clock cap {caps[0]} MHz\nsparkworker clock cap {caps[1]} MHz\n')
    return out


def test_archive_rows_are_v3_and_append_with_the_shared_api(tmp_path: Path) -> None:
    out = fake_window(tmp_path)
    values = {('ctl-150k-b1', 'ms_per_round'): [35.0, 35.4], ('ctl-32k-b4', 'accepted_per_round'): [3.0, 3.1]}
    rows = [*w.archive_rows(out, 'C1', values), *w.archive_rows(out, 'C2', values)]
    assert len(rows) == 8 and all(set(dv.ARCHIVE_REQUIRED) <= set(r) for r in rows)
    first = rows[0]
    assert first['commit'] == PROD and first['clock_cap_mhz'] == 2150 and first['clock_source'] == 'start-log'
    assert first['date'] == '20261028' and first['max_tokens'] == 512 and first['start'] == 'C1' and first['repeat'] == 0
    archive = tmp_path / 'control-archive.jsonl'
    dv.archive_append(archive, rows)
    assert dv.archive_head(archive)['rows'] == 8
    where = w.spread_where(out, 'C1')
    assert dv.archive_spread(archive, 'ctl-150k-b1', 'ms_per_round', where=where) is None   # one window so far
    later = fake_window(tmp_path, name='UPF-W1-W-20261029T010203Z')
    dv.archive_append(archive, w.archive_rows(later, 'C1', {('ctl-150k-b1', 'ms_per_round'): [36.0, 36.2]}))
    assert dv.archive_spread(archive, 'ctl-150k-b1', 'ms_per_round', where=where) == pytest.approx(
        (36.1 - 35.2) / ((36.1 + 35.2) / 2))


def test_archive_rows_refuse_unproven_clocks_candidates_and_foreign_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='one clock cap'):
        w.archive_rows(fake_window(tmp_path / 'a', (2150, 2300)), 'C1', {('ctl-32k-b4', 'ms_per_round'): [72.0]})
    out = fake_window(tmp_path / 'b')
    with pytest.raises(ValueError, match='only control starts'):
        w.archive_rows(out, 'X-0091', {('ctl-32k-b4', 'ms_per_round'): [72.0]})
    with pytest.raises(ValueError, match='not a decision-cell metric'):
        w.archive_rows(out, 'C1', {('ctl-32k-b4', 'ttft_s'): [15.0]})
    assert w.HISTORY_NAMES == {'ctl-150k-b1': '150k-b1', 'ctl-32k-b4': '32k-b4'}


def test_rule_revision_2_guard_noise_alone_is_not_a_loss() -> None:
    """WIN2-R2: a guard metric LOSES only when its loss-repeat test fires AND its interval is below zero."""
    gates = {g: True for g in w.REQUIRED_GATES['upf-0091']}
    series = []
    for cell in w.DECISION_CELLS:
        for metric, _ in w.METRICS[cell]:
            if metric == 'accepted_per_round':      # 2 of 3 repeats worse than the spread, one far better: noisy
                control, candidate = [3.0, 3.02, 2.98, 3.01], [2.9, 2.91, 3.3]
            elif metric == 'ms_per_round':
                control, candidate = [35.0, 35.35, 34.65, 35.17], [35.1, 34.9, 35.05]
            else:
                control, candidate = [69.0, 69.5, 68.9, 69.2], [69.1, 69.3, 69.0]
            series.append(w.series(cell, metric, control, candidate))
    decision = dict(dv.decide_window(series, w.DECISION_CELLS, groups=w.GROUPS))
    assert decision['rule_revision'] == w.RULE_REVISION and decision['status'] == 'INCONCLUSIVE'
    assert w.arm_verdict('upf-0091', decision, gates) == 'PROMOTABLE (no loss)'
    with pytest.raises(ValueError, match='revision 2'):
        w.arm_verdict('upf-0091', {**decision, 'rule_revision': 1}, gates)


LAUNCH = configs.HERE / 'launch' / 'UPF-RUN.sh'


def run_launch(tmp_path: Path, files: dict[str, str], gate_exit: int | None) -> subprocess.CompletedProcess[str]:
    """Run the template against a throwaway clean checkout with every host tool replaced by a recorder."""
    repo = tmp_path / 'run'
    for rel, text in {'offload/__init__.py': '', 'offload/upfix/__init__.py': '', **files}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    if gate_exit is not None:
        gate = repo / 'offload/launch/windows/chain_prev_ok.sh'
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(f'#!/bin/bash\necho "gate $*" >> "{tmp_path}/calls"\nexit {gate_exit}\n')
    for args in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x']):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for tool in ('ssh', 'scp', 'docker', 'curl', 'nvidia-smi', 'tmux'):
        (bin_dir / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{tmp_path}/calls"\nexit 97\n')
        (bin_dir / tool).chmod(0o755)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'UPF_RUN_CHECKOUT': str(repo)}
    return subprocess.run(['bash', str(LAUNCH), 'UPF-W1', head, 'prev-tmux', '-'], env=env, capture_output=True,
                          text=True)


def test_launch_refuses_without_a_runner_before_any_gate_or_host_call(tmp_path: Path) -> None:
    result = run_launch(tmp_path, {}, gate_exit=0)
    assert result.returncode == 2 and 'offload.upfix.window not built' in result.stdout
    assert not (tmp_path / 'calls').exists()


@pytest.mark.parametrize('gate_exit,expected', [(3, 'CHAIN STOPPED (chain_prev_ok exit 3)'), (2, 'gate refused')])
def test_launch_gates_on_chain_prev_ok_and_passes_exit_3_through(tmp_path: Path, gate_exit: int, expected: str) -> None:
    result = run_launch(tmp_path, {'offload/upfix/window.py': ''}, gate_exit=gate_exit)
    assert result.returncode == gate_exit and expected in result.stdout
    assert (tmp_path / 'calls').read_text() == 'gate prev-tmux -\n'


def test_launch_template_always_uses_the_chain_gate() -> None:
    text = LAUNCH.read_text()
    assert 'bash offload/launch/windows/chain_prev_ok.sh "$PREV_TMUX" "$PREV_FILE" || rc=$?' in text
    assert 'prev-ok.sh' not in text.replace('chain_prev_ok.sh', '')
