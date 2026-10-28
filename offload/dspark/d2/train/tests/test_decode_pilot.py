"""CPU decode-writer/assembly -> sample -> optimizer -> reference scorer rehearsal.

Tiny synthetic weights and a dimension-only input view bound CPU cost. Targets,
positions, request identity, masking and top20/tail values remain shard-derived.
FAST exercises its explicit bucket objective/backward, not CUDA/Triton kernels.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open

from d2 import assemble as A, generation_export as E
from d2.tests.test_assemble_rehearsal import make_copy
from offload.dspark.d2.train import loop, pilot, pilot_eval, pilot_sample
from offload.dspark.d2.train.data import Block, D2Blocks, sha256
from offload.dspark.d2.train.fast_loss import bucket_loss
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


class TinyView(D2Blocks):
    def __getitem__(self, index: int) -> Block:
        block = super().__getitem__(index)
        inp = block['inputs']
        for name in ('positions', 'mask', 'has_taps', 'taps', 'engine_mx'):
            inp[name] = inp[name][:, -8:]
        inp['taps'] = inp['taps'][..., :192]
        inp['engine_mx'] = inp['engine_mx'][..., :64]
        return block


def test_decode_assembly_sample_train_and_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    # Any attempt at CUDA is a test failure, including accidental memory probes.
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU rehearsal touched CUDA'))
    root, smoke, prompts, context = make_copy(tmp_path, monkeypatch, include_synthetic_acap=False)
    with patch.object(E, 'smoke'):  # only the external image smoke, as in assembly's own rehearsal
        receipt = A.run(root, smoke, E.digest(smoke), prompts, context, {})
    assert receipt['synthetic'] is True and receipt['pilot_cleared'] is False
    manifest = root/'shards/MANIFEST.json'
    metadata = json.loads(manifest.read_text())
    assert metadata['taps_source'] == 'decode_capture'
    assert metadata['generation']['synthetic'] is True
    assert 'synthetic' not in metadata  # the writer's nested marker must work
    assert pilot.manifest_is_synthetic(manifest)
    data = TinyView(manifest, 'train')
    block = data[0]
    # Censored rank 21 is outside top20, not a missing logprob target.
    with safe_open(data.sequences[0][0], framework='pt', device='cpu') as shard:
        assert 21 in shard.get_tensor('tok_rank').tolist()
    outside = (block['targets']['labels'][..., None] != block['targets']['top_ids']).all(-1)
    assert outside.any() and block['targets']['present'][outside].all()
    assert torch.equal(block['targets']['present'], block['targets']['position_mask'])
    assert not data[len(data)-1]['targets']['position_mask'].all()
    c = replace(tiny_config(), vocab=1024, noise=1023)
    def model(reference: bool = False) -> Drafter:
        torch.manual_seed(23)
        return Drafter(SyntheticSource(c), c, reference_math=reference)
    points = [1, 2, 3, 4]
    schedules = {arm: pilot.schedule(len(data), points, arm) for arm in pilot.ARMS}
    spec: pilot.Spec = {
        'schema': 'dsv41.d2.pilot-plan.v1', 'synthetic': True,
        'manifest_sha256': sha256(manifest), 'parity_sha256': 'synthetic', 'g1_sha256': 'synthetic',
        'source_sha256': pilot.sources(), 'frozen_identity': model().frozen_identity,
        'initialization_sha256': pilot.initialization(model()), 'run': dict(asdict(pilot.pilot_run())),
        'checkpoints': points, 'order_seed': 23, 'order_seed_b': 24,
        'population_sha256': pilot.digest(schedules['REF'][1]),
        'arm_orders': {arm: schedules[arm][2] for arm in pilot.ARMS},
        'matched_units': pilot.matched_units(data, points)}
    plan = root/'synthetic-plan.json'
    pilot.write_json(plan, spec)
    k2 = {}
    for split in pilot_sample.DEV_POPULATION:
        held = D2Blocks(manifest, split)
        path = root/(split+'.k2.jsonl')
        path.write_text(''.join(json.dumps({'seq_id': req, 'drafts': {
            str(p): [0]*5 for p in range(row[2]-1, row[1]-2)}})+'\n' for _, row, req in held.sequences))
        k2[split] = path
    # Freeze the real fixed-seed draw BEFORE either optimizer receives a step.
    pilot_sample.sample_plan(plan, manifest, k2)
    sampled = pilot.read_spec(plan, sampled=True)
    before = plan.read_bytes()
    pilot_eval.verify_sample_draw(sampled, manifest, k2)
    assert before == plan.read_bytes()
    assert sampled['sample']['seed'] == 20261024
    with pytest.raises(FileExistsError, match='refusing to redraw'):
        pilot_sample.sample_plan(plan, manifest, k2)
    run = replace(pilot.pilot_run(), chunk=256)
    ref = model(True)
    result = loop.train(ref, data, run, root/'ref.pt', plan=schedules['REF'][0])
    assert result['step'] == 4 and result['complete']
    fast = model()
    def cpu_fast(m: Drafter, h: torch.Tensor, *, chunk: int,
                 divergence: Literal['kl', 'tvd'] = 'kl',
                 **targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        value, _ = bucket_loss(h, m.t['head.weight'], m.last('markov_head.embed.weight'),
            m.last('markov_head.head.weight'), targets['previous'], targets['top_ids'],
            targets['top_logp'], targets['tail_mass'], targets['labels'], targets['present'],
            targets['position_mask'], chunk, divergence)
        return value, {}
    with patch.object(loop, 'distillation_loss', cpu_fast):
        result = loop.train(fast, data, run, root/'fast-objective.pt', plan=schedules['FAST'][0])
    assert result['step'] == 4 and result['complete']
    for split in pilot_sample.DEV_POPULATION:
        metrics = pilot_eval.score(ref.eval(), TinyView(manifest, split), k2[split],
            sampled['sample']['keys'][split], chunk=256)
        pilot_eval.validate_sample_metrics(sampled['sample'], split, metrics)
        assert metrics['rounds'] == 2
        assert [key['P'] for key in sampled['sample']['keys'][split]] == [199, 200]
        assert torch.isfinite(torch.tensor(metrics['kl_by_position'])).all()
    # Real FAST GPU admission stays intact on CPU.
    with pytest.raises(ValueError, match='all-GPU'):
        fast.enable_fast_training()
