from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
from safetensors.torch import save_file
import torch

from d2.after_replay import run
from d2.post_replay import aggregate, binding, context_ready, g1_gate, k2_complete, sha
from d2.replay_window import DryIO


def test_post_window_dry_run_has_only_x86host_commands_and_current_pilot_contract() -> None:
    class Trace(DryIO):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[str] = []
        def call(self, name: str, host: str, command: str, timeout: int = 60, supervised: bool = False) -> str:
            assert host == 'x86host'
            self.commands.append(command)
            return super().call(name, host, command, timeout, supervised)
    io = Trace()
    with patch('subprocess.Popen', side_effect=AssertionError('process')), patch(
            'urllib.request.urlopen', side_effect=AssertionError('network')):
        run(io, 'synthetic', '/opt/dsv41/x86host-data/d2/provenance/sequence-context.json', 'all')
    assert io.calls.index('context') < io.calls.index('dataset-build')
    assert io.calls.index('g1-binding-final') < io.calls.index('fresh-pilot-source')
    assert io.calls.index('pilot-prepare') < io.calls.index('pilot-sample')
    commands = '\n'.join(io.commands)
    assert 'amendment-6ab-source-prompt-bootstrap-v2' in commands
    assert 'G2-reference-top20-tail-fixed-K2-amend6ab-v2' in commands
    assert 'flock -n 9' in commands and 'timeout 840' in commands
    for forbidden in ('GPU-c4e7a371', 'GPU-7744b44b', 'prod_window.sh', 'systemctl', 'http://', 'pilot arm'):
        assert forbidden not in commands


@pytest.mark.parametrize('phase', ['build', 'g1', 'pilot'])
def test_resume_phases_do_not_rebuild_or_redraw_unrelated_work(phase: str) -> None:
    io = DryIO()
    run(io, 'synthetic', '/opt/dsv41/x86host-data/d2/context.json', phase)
    assert ('dataset-build' in io.calls) == (phase == 'build')
    assert ('pilot-sample' in io.calls) == (phase == 'pilot')
    assert ('g1-k2-acap' in io.calls) == (phase == 'g1')


@pytest.mark.parametrize('step', ['context', 'dataset-build', 'dataset-verify', 'plumbing', 's25-aggregate',
                                  'g1-complete-acap', 'g1-binding', 'pilot-rule'])
def test_post_failure_never_prepares_or_samples(step: str) -> None:
    io = DryIO(step)
    with pytest.raises(ValueError, match='synthetic'):
        run(io, 'synthetic', '/opt/dsv41/x86host-data/d2/context.json', 'all')
    assert 'pilot-prepare' not in io.calls and 'pilot-sample' not in io.calls


def teacher_fixture(root: Path, corrupt_mass: bool = False) -> None:
    # Small synthetic teacher fixture with analytically known masses, not served data.
    path = root/'synthetic.safetensors'
    lp = torch.tensor([[math.log(.5), math.log(.25)], [math.log(.75), math.log(.25)]], dtype=torch.float32)
    resid = torch.tensor([.25, .1 if corrupt_mass else 0])
    save_file({'top_lp': lp, 'resid': resid}, str(path))
    (root/'MANIFEST.json').write_text(json.dumps({'synthetic': True, 'splits': {'dev': {'files': [
        {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}}}))


def test_s25_aggregate_counts_zero_residual_and_measures_mass_error(tmp_path: Path) -> None:
    teacher_fixture(tmp_path)
    result = aggregate(tmp_path)
    assert result['span_positions'] == 2 and result['resid_eq_zero'] == 1
    assert result['min_top_lp'] == pytest.approx(math.log(.25))
    mass = result['abs_sum_p_plus_resid_minus_1']
    assert isinstance(mass, dict) and mass['count_gt_1e-4'] == 0 and mass['max'] < 1e-6


def test_s25_refuses_mass_errors_and_modified_shards(tmp_path: Path) -> None:
    teacher_fixture(tmp_path, corrupt_mass=True)
    with pytest.raises(ValueError, match='mass error'):
        aggregate(tmp_path)
    (tmp_path/'synthetic.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        aggregate(tmp_path)


def test_k2_deadline_success_is_not_complete_coverage(tmp_path: Path) -> None:
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'splits': {'acap': {'files': [{'sidecar': 'side.json'}]}}}))
    (tmp_path/'side.json').write_text(json.dumps({'sequences': [{'seq_id': 'synthetic', 'gen_start': 8, 'L': 12}]}))
    drafts = tmp_path/'acap.k2_drafts.jsonl'
    drafts.write_text('')
    with pytest.raises(ValueError, match='deadline'):
        k2_complete(tmp_path, tmp_path, 'acap')
    # Synthetic IDs here test shape only; there is no served content in this fixture.
    drafts.write_text(json.dumps({'seq_id': 'synthetic', 'drafts': {str(p): [0]*5 for p in range(7, 10)}})+'\n')
    k2_complete(tmp_path, tmp_path, 'acap')
    drafts.write_text(json.dumps({'seq_id': 'synthetic', 'drafts': {'7': [0]*5}})+'\n')
    with pytest.raises(ValueError, match='block coverage'):
        k2_complete(tmp_path, tmp_path, 'acap')


def test_context_scaffold_and_nonbinding_g1_refused(tmp_path: Path) -> None:
    context, manifest, plan = [tmp_path/name for name in ('context.json', 'manifest.json', 'plan.jsonl')]
    context.write_text(json.dumps({'status': 'INCOMPLETE_DO_NOT_BUILD'}))
    manifest.write_text('{}')
    plan.write_text(json.dumps({'seq_id': 'synthetic', 'kind': 'gen'})+'\n')
    with pytest.raises(ValueError, match='scaffold'):
        context_ready(context, manifest, plan)
    context.write_text(json.dumps({'split': 'heldout_legal', 'g1': {'pass': True, 'binding': False}}))
    with pytest.raises(ValueError, match='binding'):
        g1_gate(context)


def test_resume_binding_rejects_foreign_context_plan_and_overrides(tmp_path: Path) -> None:
    # Text-free synthetic metadata exercises the real gate-binding validator.
    replay = tmp_path/'replay'
    (replay/'taps/acap0').mkdir(parents=True)
    (replay/'plan.jsonl').write_text(json.dumps({'seq_id': 'acap0', 'kind': 'acap'})+'\n')
    hashes = {'sha256_taps': 'a'*64, 'sha256_ids': 'b'*64}
    (replay/'taps/acap0/DONE.json').write_text(json.dumps(hashes))
    gate, context, prompts = [tmp_path/name for name in ('gate.json', 'context.json', 'prompts.json')]
    context.write_text('synthetic context fixture')
    prompts.write_text('synthetic prompt identity fixture')
    gate.write_text(json.dumps({'schema': 'd2.equiv.gate.v1', 'blocks_inputs_differ': 1,
        'gate': {'pass': True, 'verdict': 'pass', 'population': 'synthetic', 'threshold': .99, 'rule2_bf16_noise': {}},
        'replay_binding': {'acap0': hashes}}))
    manifest = {'replay': {'plan_sha256': sha(replay/'plan.jsonl')},
                'generation': {'sequence_context_sha256': sha(context), 'synthetic': False},
                'prompt_manifest_sha256': sha(prompts), 'taps_source': 'prefill_replay',
                'checks': {'equivalence_gate': {'sha256': sha(gate), 'override': None},
                           'skipped_sequences': 0, 'g1_plumbing': {'override': None}}}
    target = tmp_path/'MANIFEST.json'
    target.write_text(json.dumps(manifest))
    binding(tmp_path, replay, gate, context, prompts)
    for file in (context, replay/'plan.jsonl'):
        before = file.read_text()
        # Preserve valid JSONL syntax while changing its byte identity.
        file.write_text(' '+before)
        with pytest.raises(ValueError, match='bound'):
            binding(tmp_path, replay, gate, context, prompts)
        file.write_text(before)
    changed = json.loads(target.read_text())
    changed['checks']['skipped_sequences'] = 1
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='skips'):
        binding(tmp_path, replay, gate, context, prompts)
