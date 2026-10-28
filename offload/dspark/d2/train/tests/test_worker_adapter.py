from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from offload.dspark.d2.train.model import Config, Stage as TrainingStage
from offload.dspark.d2.train.synthetic import tiny_config
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.d2.train.worker_adapter import DeltaLinear, GroupedDelta, Worker, install, adapted_identity
from offload.dspark.native.arms import ARMS


class Base:
    def __init__(self, weight: Tensor):
        self.w = weight.clone()
        self.out_features, self.in_features = weight.shape

    def __call__(self, x: Tensor, out_f32: bool = False, out: Tensor | None = None) -> Tensor:
        value = F.linear(x.float(), self.w.float())
        value = value if out_f32 else value.to(x.dtype)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def nbytes(self) -> int:
        return self.w.numel()*self.w.element_size()


def fixture() -> tuple[Worker, dict[str, Tensor], Config]:
    torch.manual_seed(32)
    m = model()
    c = m.c
    values = {n: p.detach().clone() for n, p in m.named_parameters()}
    for n, value in values.items():
        if n.endswith(".b"):
            value.normal_(std=.1)
    ranks = []
    for r in range(2):
        stages = []
        for module_stage in m.stages:
            stage = cast(TrainingStage, module_stage)
            g0, g1, i0, i1, h0, h1 = r, r+1, r*c.inter//2, (r+1)*c.inter//2, r, r+1
            def base(op: str, rows: slice = slice(None), cols: slice = slice(None)) -> Base:
                return Base(cast(Tensor, stage.dense[op.replace(".", "__")].weight)[rows, cols])
            stages.append(SimpleNamespace(wq_a=base("attn.wq_a"), wkv=base("attn.wkv"),
                wq_b=base("attn.wq_b", slice(h0*c.hd, h1*c.hd)),
                wo_b=base("attn.wo_b", cols=slice(g0*c.outr, g1*c.outr)),
                sh_w1=base("ffn.shared_experts.w1", slice(i0, i1)),
                sh_w3=base("ffn.shared_experts.w3", slice(i0, i1)),
                sh_w2=base("ffn.shared_experts.w2", cols=slice(i0, i1)),
                wo_a_n=Base(stage.wo_a_base[g0:g1].reshape(c.outr, -1)),
                shared_pair=object(), adapter_wo_a=None))
        ranks.append(SimpleNamespace(g0=r, g1=r+1, i0=r*c.inter//2, i1=(r+1)*c.inter//2,
            h0=r, h1=r+1, v0=r*c.vocab//2, v1=(r+1)*c.vocab//2, dev=torch.device("cpu"), stages=stages,
            markov=SimpleNamespace(cache=torch.ones(2), slot_of_token=torch.ones(2, dtype=torch.int64)),
            ring_kv=torch.ones(2), ring_pos=torch.ones(2, dtype=torch.int64)))
    worker = cast(Worker, SimpleNamespace(cfg=ARMS["native-hybrid-head-hot"], ALL=(0, 1), ranks=ranks,
                                        graphs={}, margin_graphs={}, insert_graphs={}))
    return worker, values, c


def test_tp2_installs_all_trainables_shards_factors_and_rebuilds_caches() -> None:
    worker, values, c = fixture()
    originals = [[cast(Base, stage.wq_b).w.clone() for stage in rank.stages] for rank in worker.ranks]
    calls: list[tuple[Tensor, Tensor]] = []
    def cache(w1: Tensor, w2: Tensor, tokens: list[int], rows: int) -> object:
        calls.append((w1, w2))
        return "rebuilt-adapted-hot-rows"
    install(worker, values, c, 32, cache, [3, 4])
    assert len(calls) == 2
    for r, rk in enumerate(worker.ranks):
        assert rk.mrows == "rebuilt-adapted-hot-rows"
        assert not bool(rk.ring_kv.any()) and bool((rk.ring_pos == -1).all())
        assert not bool(rk.markov.cache.any()) and bool((rk.markov.slot_of_token == -1).all())
        for i, stage in enumerate(rk.stages):
            layer = cast(DeltaLinear, stage.wq_b)
            assert torch.equal(cast(Base, layer.base).w, originals[r][i])
            assert torch.equal(layer.b, values[f"stages.{i}.dense.attn__wq_b.b"][r*c.hd:(r+1)*c.hd])
            row = cast(DeltaLinear, stage.wo_b)
            assert torch.equal(row.a, values[f"stages.{i}.dense.attn__wo_b.a"][:, r*c.outr:(r+1)*c.outr])
            assert isinstance(stage.adapter_wo_a, GroupedDelta) and stage.shared_pair is None
    worker.adapter_guard.check()
    worker.graphs["captured"] = object()
    cast(DeltaLinear, worker.ranks[1].stages[0].wkv).b.add_(1)
    with pytest.raises(RuntimeError, match="dependency changed"):
        worker.adapter_guard.check()
    assert not worker.graphs and all(r.mrows is None for r in worker.ranks)


@pytest.mark.parametrize("sharding", ["row", "column", "replicated"])
def test_additive_fp32_delta_sharding_matches_full_matrix_before_rounding(sharding: str) -> None:
    torch.manual_seed(23)
    x = torch.randn(5, 64).bfloat16()
    w = torch.randn(96, 64).bfloat16()
    a, b = torch.randn(32, 64)*.1, torch.randn(96, 32)*.1
    expected = F.linear(x.float(), w.float())+F.linear(F.linear(x.float(), a), b)
    if sharding == "row":
        outputs = [DeltaLinear(Base(w[:, lo:lo+32]), a[:, lo:lo+32], b)(x[:, lo:lo+32], out_f32=True) for lo in (0, 32)]
        actual = sum(outputs)
    elif sharding == "column":
        actual = torch.cat([DeltaLinear(Base(w[lo:lo+48]), a, b[lo:lo+48])(x, out_f32=True) for lo in (0, 48)], -1)
    else:
        actual = DeltaLinear(Base(w), a, b)(x, out_f32=True)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)
    if sharding == "replicated":
        layer = DeltaLinear(Base(w), a, b)
        assert torch.equal(layer(x), expected.bfloat16())
        assert torch.equal(cast(Base, layer.base).w, w)


def test_grouped_updates_never_mix_group_inputs() -> None:
    torch.manual_seed(43)
    w = torch.randn(2, 32, 32).bfloat16()
    factors = [(torch.randn(32, 32)*.1, torch.randn(32, 32)*.1) for _ in range(2)]
    x = torch.randn(5, 64).bfloat16()
    actual = GroupedDelta(w, factors)(x)
    expected = torch.cat([F.linear(x[:, g*32:(g+1)*32].float(), w[g].float())+
        F.linear(F.linear(x[:, g*32:(g+1)*32].float(), a), b) for g, (a, b) in enumerate(factors)], -1).bfloat16()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("defect", ["missing", "extra", "nonfinite", "shape", "captured", "second_install", "lowbit"])
def test_unsupported_or_partial_adapter_refuses_before_mutation(defect: str) -> None:
    worker, values, config = fixture()
    if defect == "missing":
        values.pop(next(iter(values)))
    elif defect == "extra":
        values["unknown"] = torch.ones(1)
    elif defect == "nonfinite":
        values[next(iter(values))].fill_(torch.nan)
    elif defect == "shape":
        values[next(iter(values))] = torch.ones(1)
    elif defect == "captured":
        worker.graphs["captured"] = object()
    elif defect == "second_install":
        install(worker, values, config, 32, lambda *args: object(), [])
    else:
        worker.cfg = ARMS["native-packed"]
    before = worker.ranks[0].stages[0].wq_a
    with pytest.raises(ValueError):
        install(worker, values, config, 32, lambda *args: object(), [])
    assert worker.ranks[0].stages[0].wq_a is before


def test_adapted_identity_separates_adapter_base_and_profile() -> None:
    base: dict[str, object] = {"drafter_identity": "base", "markov_identity": "original", "recipe": {"precision": "native"}}
    identity = adapted_identity(base, "a"*64, tiny_config())
    assert identity["drafter_identity"] != base["drafter_identity"]
    assert identity["markov_identity"] != base["markov_identity"]
    assert identity != adapted_identity(base, "b"*64, tiny_config())
    assert identity["profile_label"] == "adapted; not unchanged source-faithful"


def test_dependency_rebinding_also_invalidates_cached_state() -> None:
    worker, values, c = fixture()
    install(worker, values, c, 32, lambda *args: object(), [])
    worker.ranks[0].norm = worker.ranks[0].norm.clone()
    with pytest.raises(RuntimeError, match="dependency changed"):
        worker.adapter_guard.check()


def test_delta_is_added_before_native_output_rounding() -> None:
    x = torch.ones(1, 2).bfloat16()
    base = Base(torch.tensor([[1., .00390625]]).bfloat16())
    a, b = torch.zeros(32, 2), torch.zeros(1, 32)
    a[0, 0], b[0, 0] = 1., .0001
    layer = DeltaLinear(base, a, b)
    correct = (base(x, out_f32=True)+.0001).bfloat16()
    wrong = (base(x).float()+.0001).bfloat16()
    assert not torch.equal(correct, wrong)
    assert torch.equal(layer(x), correct)
