"""S24 item 4: TP2 adapter sharding tied back to the TRAINING math, with counterexamples.

The lane's tests compare DeltaLinear against slices the test itself constructs. Here `install()` performs the slicing
on a two-rank worker whose base shards and rank ranges follow native/tp2_native.py (h0 = r*heads/tp, g0 = r*groups/tp,
i0 = r*inter/tp, uneven head-tile vocabulary split), and the TP2 composition (concat column shards, sum row partials in
FP32, replicated, grouped wo_a) is compared with the training model's own Dense / grouped wo_a forward with the same
trained factors. Deliberately mis-sharded variants must be caught by the same comparison.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from offload.dspark.d2.train.model import Drafter, Stage as TrainingStage
from offload.dspark.d2.train.ops import Dense, grouped_linear
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config
from offload.dspark.d2.train.worker_adapter import DeltaLinear, GroupedDelta, Worker, install
from offload.dspark.native.arms import ARMS
from offload.dspark.native.formats import act_qdq

TP = 2
OPS = {"attn.wq_a": ("wq_a", "replicated"), "attn.wkv": ("wkv", "replicated"), "attn.wq_b": ("wq_b", "column"),
       "attn.wo_b": ("wo_b", "row"), "ffn.shared_experts.w1": ("sh_w1", "column"),
       "ffn.shared_experts.w3": ("sh_w3", "column"), "ffn.shared_experts.w2": ("sh_w2", "row")}


class Base:
    """Native-dense stand-in: exact FP32 product of a BF16 shard, optional FP32 output (as NativeDense)."""
    def __init__(self, weight: Tensor):
        self.w = weight.detach().clone().contiguous()
        self.out_features, self.in_features = self.w.shape

    def __call__(self, x: Tensor, out_f32: bool = False, out: Tensor | None = None) -> Tensor:
        value = F.linear(x.double(), self.w.double()).float()
        return value if out_f32 else value.to(x.dtype)

    def nbytes(self) -> int:
        return self.w.numel()*self.w.element_size()


def drafter() -> Drafter:
    torch.manual_seed(57)
    # 4 heads / 4 groups so each rank owns TWO heads and TWO wo_a groups (ownership ordering matters).
    c = replace(tiny_config(), heads=4, groups=4)
    m = Drafter(SyntheticSource(c), c)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(".b"):
                p.normal_(std=.2)
            elif "norm" in n:
                p.add_(torch.randn_like(p)*1e-3)        # FP32 values not representable in BF16
    return m


def worker_for(m: Drafter, vocab_cut: int) -> Worker:
    c = m.c
    ranks = []
    for r in range(TP):
        h0, h1 = r*c.heads//TP, (r+1)*c.heads//TP
        g0, g1 = r*c.groups//TP, (r+1)*c.groups//TP
        i0, i1 = r*c.inter//TP, (r+1)*c.inter//TP
        v0, v1 = (0, vocab_cut) if r == 0 else (vocab_cut, c.vocab)
        stages = []
        for module in m.stages:
            st = cast(TrainingStage, module)
            def w(op: str) -> Tensor:
                return cast(Tensor, st.dense[op.replace(".", "__")].weight)
            r0, r1 = g0*c.outr, g1*c.outr
            stages.append(SimpleNamespace(
                wq_a=Base(w("attn.wq_a")), wkv=Base(w("attn.wkv")),
                wq_b=Base(w("attn.wq_b")[h0*c.hd:h1*c.hd]), wo_b=Base(w("attn.wo_b")[:, r0:r1]),
                sh_w1=Base(w("ffn.shared_experts.w1")[i0:i1]), sh_w3=Base(w("ffn.shared_experts.w3")[i0:i1]),
                sh_w2=Base(w("ffn.shared_experts.w2")[:, i0:i1]),
                wo_a_n=SimpleNamespace(w=st.wo_a_base.reshape(c.groups*c.outr, -1)[r0:r1].bfloat16().contiguous()),
                shared_pair=object(), adapter_wo_a=None,
                attn_norm=torch.ones(c.dim).bfloat16(), ffn_norm=torch.ones(c.dim).bfloat16(),
                q_norm=torch.ones(c.qr).bfloat16(), kv_norm=torch.ones(c.hd).bfloat16()))
        ranks.append(SimpleNamespace(r=r, g0=g0, g1=g1, h0=h0, h1=h1, i0=i0, i1=i1, v0=v0, v1=v1,
                                     dev=torch.device("cpu"), stages=stages,
                                     main_norm=torch.ones(c.dim).bfloat16(), norm=torch.ones(c.dim).bfloat16(),
                                     conf_w=torch.zeros(c.dim+c.markov),
                                     markov=SimpleNamespace(w1=None, w2=None, cache=torch.ones(3), slot_of_token=torch.ones(3, dtype=torch.long)),
                                     mrows="original-bf16-rows", ring_kv=torch.ones(3), ring_pos=torch.ones(3, dtype=torch.long)))
    return cast(Worker, SimpleNamespace(cfg=ARMS["native-hybrid-head-hot"], ALL=(0, 1), ranks=ranks,
                                        graphs={}, margin_graphs={}, insert_graphs={}))


def tp_apply(layers: list[DeltaLinear], kind: str, x: Tensor, c_in: list[tuple[int, int]] | None) -> Tensor:
    """TP2 composition of rank outputs in FP32 (column: concat; row: sum of partials; replicated: rank 0)."""
    if kind == "column":
        return torch.cat([layer(x, out_f32=True) for layer in layers], -1)
    if kind == "row":
        assert c_in is not None
        return cast(Tensor, sum(layer(x[:, lo:hi], out_f32=True) for layer, (lo, hi) in zip(layers, c_in)))
    outs = [layer(x, out_f32=True) for layer in layers]
    assert torch.equal(outs[0], outs[1])
    return outs[0]


def training_fp32(st: TrainingStage, op: str, x16: Tensor) -> Tensor:
    """The training Dense forward in FP32 (no output rounding): FP32 input => Dense returns FP32."""
    layer = cast(Dense, st.dense[op.replace(".", "__")])
    with torch.no_grad():
        return cast(Tensor, layer(x16.float()))


@pytest.mark.parametrize("vocab_cut", [48, 56])
def test_s24_installed_tp2_deltas_reproduce_training_dense_and_wo_a(vocab_cut: int) -> None:
    m = drafter()
    c = m.c
    values = {n: p.detach().clone() for n, p in m.named_parameters()}
    worker = worker_for(m, vocab_cut)
    base_bytes = [[{a: getattr(stage, a).w.clone() for a, _ in OPS.values()} for stage in rk.stages] for rk in worker.ranks]
    install(worker, values, c, 32, lambda w1, w2, tokens, rows: ("rebuilt", w1, w2), [1, 2])
    torch.manual_seed(5)
    for i, module in enumerate(m.stages):
        st = cast(TrainingStage, module)
        for op, (attribute, kind) in OPS.items():
            layer = cast(Dense, st.dense[op.replace(".", "__")])
            width = layer.weight.shape[1]
            x = act_qdq(torch.randn(3, width).bfloat16())            # the worker receives QDQ'd BF16 input
            rank_layers = [cast(DeltaLinear, getattr(rk.stages[i], attribute)) for rk in worker.ranks]
            lim = [(rk.g0*c.outr, rk.g1*c.outr) if op == "attn.wo_b" else (rk.i0, rk.i1) for rk in worker.ranks]
            got = tp_apply(rank_layers, kind, x, lim if kind == "row" else None)
            want = training_fp32(st, op, x)
            torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6, msg=f"stage {i} {op}")
            delta = got-tp_apply([DeltaLinear(lay.base, torch.zeros_like(lay.a), torch.zeros_like(lay.b))
                                  for lay in rank_layers], kind, x, lim if kind == "row" else None)
            assert float(delta.abs().max()) > 1e-3, f"adapter has no effect on {op}"
            for rk, before in zip(worker.ranks, base_bytes):
                assert torch.equal(cast(Base, cast(DeltaLinear, getattr(rk.stages[i], attribute)).base).w, before[i][attribute])
        # Grouped wo_a: concatenated rank outputs == training grouped_linear + per-group LoRA (FP32).
        o = act_qdq(torch.randn(3, c.heads*c.hd).bfloat16())
        heads_r = c.heads//TP
        got_wo_a = torch.cat([cast(GroupedDelta, rk.stages[i].adapter_wo_a)(
            o[:, rk.h0*c.hd:rk.h1*c.hd].float()) for rk in worker.ranks], -1)
        assert heads_r*c.hd == (c.groups//TP)*(c.heads//c.groups)*c.hd
        og = o.float().reshape(1, 3, c.groups, -1)
        projected = grouped_linear(og, st.wo_a_base)
        updates = torch.stack([F.linear(F.linear(og[:, :, g], cast(Dense, lay).a), cast(Dense, lay).b)
                               for g, lay in enumerate(st.wo_a)], -2)
        want_wo_a = (projected+updates).reshape(3, -1).detach()
        torch.testing.assert_close(got_wo_a, want_wo_a, rtol=2e-6, atol=2e-6)
        assert all(rk.stages[i].shared_pair is None for rk in worker.ranks)
    # Norms / confidence are exact FP32 trained values (no BF16 rounding); Markov head vocab-sliced by rank.
    last = f"t.mtp__{c.stages-1}__"
    for rk in worker.ranks:
        assert rk.main_norm.dtype == torch.float32 and torch.equal(rk.main_norm, values["t.mtp__0__main_norm__weight"])
        assert torch.equal(rk.norm, values[last+"norm__weight"])
        assert torch.equal(rk.conf_w, values[last+"confidence_head__proj__weight"].flatten())
        assert torch.equal(rk.stages[1].q_norm, values["stages.1.t.mtp__1__attn__q_norm__weight"])
        rebuilt = cast(tuple[str, Tensor, Tensor], rk.mrows)
        assert rebuilt[0] == "rebuilt" and torch.equal(rebuilt[2], rk.markov.w2)
    head = values[last+"markov_head__head__weight"]
    assert torch.equal(torch.cat([rk.markov.w2 for rk in worker.ranks]), head)
    assert worker.ranks[0].markov.w2.shape[0] == vocab_cut


def test_s24_counterexamples_are_caught() -> None:
    """Mis-sharded factors (wrong side sliced, wrong rank's slice, or rounding before adding) fail the comparison."""
    m = drafter()
    c = m.c
    st = cast(TrainingStage, m.stages[0])
    torch.manual_seed(9)
    def compare(make: Callable[[int, Tensor, Tensor, Tensor], DeltaLinear], op: str, kind: str) -> bool:
        layer = cast(Dense, st.dense[op.replace(".", "__")])
        x = act_qdq(torch.randn(3, layer.weight.shape[1]).bfloat16())
        a, b, w = layer.a.detach(), layer.b.detach(), layer.weight
        if kind == "row":
            half = w.shape[1]//TP
            got = sum(make(r, w[:, r*half:(r+1)*half], a, b)(x[:, r*half:(r+1)*half], out_f32=True) for r in range(TP))
        else:
            half = w.shape[0]//TP
            got = torch.cat([make(r, w[r*half:(r+1)*half], a, b)(x, out_f32=True) for r in range(TP)], -1)
        return bool(torch.allclose(cast(Tensor, got), training_fp32(st, op, x), rtol=2e-6, atol=2e-6))
    # Correct conventions pass.
    assert compare(lambda r, w, a, b: DeltaLinear(Base(w), a[:, r*(a.shape[1]//2):(r+1)*(a.shape[1]//2)], b), "attn.wo_b", "row")
    assert compare(lambda r, w, a, b: DeltaLinear(Base(w), a, b[r*(b.shape[0]//2):(r+1)*(b.shape[0]//2)]), "attn.wq_b", "column")
    # Row-parallel with the FULL A on each rank's input shard is impossible (shape refusal) ...
    with pytest.raises(ValueError, match="shape"):
        compare(lambda r, w, a, b: DeltaLinear(Base(w), a, b), "attn.wo_b", "row")
    # ... row-parallel with the WRONG rank's A columns is caught,
    assert not compare(lambda r, w, a, b: DeltaLinear(Base(w), a[:, (1-r)*(a.shape[1]//2):(2-r)*(a.shape[1]//2)], b), "attn.wo_b", "row")
    # ... column-parallel with B rows of the other rank is caught,
    assert not compare(lambda r, w, a, b: DeltaLinear(Base(w), a, b[(1-r)*(b.shape[0]//2):(2-r)*(b.shape[0]//2)]), "ffn.shared_experts.w1", "column")
    # ... and a delta added on BOTH ranks for a row-parallel op (double counted after the reduce) is caught.
    class Twice(DeltaLinear):
        def __call__(self, x: Tensor, out_f32: bool = False, out: Tensor | None = None) -> Tensor:
            return self.base(x, out_f32=True)+F.linear(F.linear(x.float(), self.a), self.b)*2
    assert not compare(lambda r, w, a, b: Twice(Base(w), a[:, r*(a.shape[1]//2):(r+1)*(a.shape[1]//2)], b), "attn.wo_b", "row")


def test_s24_dependency_guard_scope() -> None:
    """OBSERVATION probe: the guard tracks autograd version counters and bindings. A write through `.data`
    bypasses the version counter and is NOT detected; nothing in the evaluation harness writes adapters
    after install, so this is defence-in-depth scope, not a reachable defect."""
    m = drafter()
    worker = worker_for(m, 48)
    install(worker, {n: p.detach().clone() for n, p in m.named_parameters()}, m.c, 32, lambda *a: object(), [])
    worker.adapter_guard.check()
    layer = cast(DeltaLinear, worker.ranks[0].stages[2].sh_w2)
    layer.b.data.add_(1.)                                  # bypasses _version
    worker.adapter_guard.check()                           # not detected
    with torch.no_grad():
        layer.b.add_(1.)                                   # regular in-place write is detected
    with pytest.raises(RuntimeError, match="dependency changed"):
        worker.adapter_guard.check()
    assert not worker.graphs and all(rk.mrows is None for rk in worker.ranks)
    with pytest.raises(RuntimeError, match="dependency changed"):
        worker.adapter_guard.check()                       # stays invalid until a new worker


# ---------------------------------------------------------------- reachability / identity / leases ------------------
def test_s24_service_cli_cannot_reach_the_evaluation_factory() -> None:
    import ast
    import inspect
    from offload.dspark.worker import arm_selection, server
    tree = ast.parse(inspect.getsource(server.main))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"build_adapted_evaluation_core", "build", "install", "worker_eval_build", "worker_adapter"}
    flags = [a for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             for a in n.args if isinstance(a, ast.Constant)]
    assert flags and not [f for f in flags if "adapt" in str(f).lower() or "lora" in str(f).lower()]
    assert not [a for a in arm_selection.SERVING_ARMS if "adapt" in a or "d2" in a]
    for bad in ("d2-adapted-native", "native-hybrid-head-hot"):
        with pytest.raises(ValueError, match="no fallback"):
            arm_selection.selected_arm(bad)
    # The adapted label/identities never equal the unchanged base receipt.
    from offload.dspark.d2.train.worker_adapter import adapted_identity
    base = {"drafter_identity": "d"*64, "markov_identity": "m"*64}
    ident = adapted_identity(base, "a"*64, tiny_config())
    assert ident["arm"] == "d2-adapted-native" and "not unchanged source-faithful" in str(ident["profile_label"])
    assert {ident["drafter_identity"], ident["markov_identity"]}.isdisjoint(base.values())
    assert ident["drafter_identity"] != ident["markov_identity"]


PROD = ("GPU-c4e7a371", "GPU-7744b44b")


@pytest.mark.parametrize("visible", [",".join(PROD), PROD[0], "", "0,1",
                                     "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7,GPU-240876ad-ea42-47de-1b8d-f6153810b7e1",
                                     "GPU-240876ad-ea42-47de-1b8d-f6153810b7e1," + PROD[0]])
def test_s24_two_solo_lease_refuses_everything_but_the_ordered_solo_pair(visible: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.d2.train.worker_eval_lease import require_evaluation_lease
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("D2_ADAPTER_EVAL", "1")
    with pytest.raises(RuntimeError, match="only both named solos"):
        require_evaluation_lease(False)


@pytest.mark.parametrize("visible", [",".join(PROD), PROD[1], "GPU-240876ad-ea42-47de-1b8d-f6153810b7e1,GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"])
def test_s24_emulated_lease_refuses_production_and_multi_card(visible: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.d2.train.worker_eval_lease import require_evaluation_lease
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("D2_PILOT_LEASE", "1")
    with pytest.raises(RuntimeError, match="permitted solo UUID"):
        require_evaluation_lease(True)


@pytest.mark.parametrize("gpu", ["GPU-c4e7a371-0000-0000-0000-000000000000", "GPU-7744b44b-0000-0000-0000-000000000000", "0"])
def test_s24_launcher_refuses_production_pair_before_any_side_effect(gpu: str, tmp_path: Path) -> None:
    import subprocess
    script = Path(__file__).resolve().parents[2]/"train/run_pilot_locked.sh"
    root = tmp_path/"root"
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "D2_TRAIN_ROOT": str(root), "D2_PILOT_GPU": gpu}
    done = subprocess.run(["bash", str(script), "worker-eval", "-c", "pass"], env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "only the two named solo cards" in done.stderr
    assert not root.exists()                               # no logs dir, no lock, no nvidia-smi, no python
