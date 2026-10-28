"""Feature boundaries and the target-trajectory survival estimator, CPU only."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from offload.dspark.d2.train.data import D2Blocks, sha256
from offload.dspark.d2.train.evaluate import match_length
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, inputs, tiny_config
from offload.dspark.d2.train.tests.test_data import write_shard


def test_future_taps_and_teacher_rows_do_not_change_backbone_inputs(tmp_path: Path) -> None:
    manifest = write_shard(tmp_path)
    before = D2Blocks(manifest, "train")[0]
    assert int(before["inputs"]["P"][0]) == 3
    tensors = load_file(tmp_path / "train.safetensors")
    tensors["taps"][4:] = 900  # all rows after P, including the anchor's target taps
    tensors["top_lp"][1:] = torch.tensor(.02).log()  # different future targets
    tensors["resid"][1:] = .6
    save_file(tensors, tmp_path / "train.safetensors", metadata={"schema": "d2.shard.v1", "split": "train"})
    side = tmp_path / "train.json"
    meta = json.loads(side.read_text())
    meta["sha256"] = sha256(tmp_path / "train.safetensors")
    side.write_text(json.dumps(meta))
    record = json.loads(manifest.read_text())
    entry = record["splits"]["train"]["files"][0]
    entry["sha256"], entry["sidecar_sha256"] = meta["sha256"], sha256(side)
    manifest.write_text(json.dumps(record))
    after = D2Blocks(manifest, "train")[0]
    for name in before["inputs"]:
        assert torch.equal(before["inputs"][name], after["inputs"][name]), name
    assert not torch.equal(before["targets"]["top_logp"], after["targets"]["top_logp"])
    assert int(after["inputs"]["positions"].max()) == 3


def test_masked_context_has_no_effect_on_queries() -> None:
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c, checkpoint_stages=False).eval()
    original = inputs(c, batch=1)
    original["mask"][:, :2] = False
    changed = {k: v.clone() for k, v in original.items()}
    changed["taps"][:, :2] = 900  # finite but unrelated padding features
    with torch.no_grad():
        a = m.hidden_from(original)
        b = m.hidden_from(changed)
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_target_sample_prefix_match_is_unbiased_survival_for_fixed_deterministic_chain() -> None:
    first = torch.tensor([.2, .5, .3], dtype=torch.float64)
    transition = torch.tensor([[.7, .2, .1], [.1, .3, .6], [.4, .5, .1]], dtype=torch.float64)
    draft = [1, 2, 0]
    total = 0.
    for chain in itertools.product(range(3), repeat=3):
        probability = first[chain[0]] * transition[chain[0], chain[1]] * transition[chain[1], chain[2]]
        truth: list[int | None] = list(chain)
        accepted = match_length(draft, truth)
        assert accepted is not None
        total += float(probability) * accepted
    expected = .5 + .5 * .6 + .5 * .6 * .4
    assert abs(total - expected) < 1e-12
