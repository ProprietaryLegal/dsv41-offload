"""S25 merge check: the independent FP64 oracle is right (finite differences), and the code at this
tree's HEAD matches it on generic and boundary inputs for REF, FAST and the evaluator."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import importlib.util

import pytest
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s25_oracle", HERE / "s25_oracle.py")
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)
ROOT = HERE.parents[4]


def _case(seed: int, divergence: str, kind: str) -> tuple[dict[str, torch.Tensor], str]:
    g = torch.Generator().manual_seed(seed)
    V, k = 9, (9 if kind == "full_support" else 3)
    c = {"h": torch.randn(2, 5, 4, generator=g, dtype=torch.float64),
         "head": torch.randn(V, 4, generator=g, dtype=torch.float64),
         "embed": torch.randn(V, 3, generator=g, dtype=torch.float64),
         "markov": torch.randn(V, 3, generator=g, dtype=torch.float64),
         "previous": torch.randint(0, 2, (2, 5), generator=g),
         "ids": torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(10)]).reshape(2, 5, k)}
    teacher = torch.randn(2, 5, V, generator=g, dtype=torch.float64).softmax(-1)
    p = teacher.gather(-1, c["ids"])
    tail = 1 - p.sum(-1)
    if kind in ("zero_tail", "full_support"):
        p, tail = p / p.sum(-1, keepdim=True), torch.zeros_like(tail)
    if kind == "rounded_mass":
        tail = tail + 8e-5
    present = torch.ones(2, 5, dtype=torch.bool)
    mask = present.clone()
    labels = torch.randint(0, V, (2, 5), generator=g)
    if kind == "missing":
        mask[0, 4] = present[0, 4] = present[1, 1] = False
        c["ids"][1, 1] = -1
        p[1, 1], tail[1, 1] = 0, 1
    c.update(p=p, logp=p.log(), tail=tail, labels=labels, present=present, mask=mask)
    return c, divergence


@pytest.mark.parametrize("divergence", ("kl", "tvd"))
@pytest.mark.parametrize("kind", ("generic", "zero_tail", "full_support", "rounded_mass", "missing"))
def test_oracle_gradient_matches_central_differences(divergence: str, kind: str) -> None:
    c, _ = _case(7, divergence, kind)
    args = [c[n] for n in ("h", "head", "embed", "markov", "previous", "ids", "p", "logp", "tail", "labels",
                           "present", "mask")]
    _, _, grad = oracle.objective(*args, divergence, 1.)
    eps = 1e-6
    for name, index in (("hidden", 0), ("embed", 2), ("markov", 3)):
        base = args[index]
        numeric = torch.zeros_like(base)
        for i in range(base.numel()):
            up, down = base.clone(), base.clone()
            up.view(-1)[i] += eps
            down.view(-1)[i] -= eps
            a, b = list(args), list(args)
            a[index], b[index] = up, down
            numeric.view(-1)[i] = (oracle.objective(*a, divergence)[0] - oracle.objective(*b, divergence)[0]) / (2 * eps)
        torch.testing.assert_close(grad[name], numeric, rtol=1e-6, atol=1e-8)


def test_oracle_keeps_true_support_mismatch_infinite() -> None:
    z = torch.tensor([[0., -math.inf, 1.], [0., 1., -math.inf]], dtype=torch.float64)
    ids = torch.tensor([[1], [1]])
    p = torch.tensor([[.2], [1.]], dtype=torch.float64)
    tail = torch.tensor([.8, 0.], dtype=torch.float64)
    T = torch.ones(2, dtype=torch.bool)
    per, _ = oracle.per_position(z, ids, p, p.log(), tail, torch.zeros(2, dtype=torch.long), T, T, "kl")
    assert math.isinf(float(per[0])) and float(per[0]) > 0  # positive teacher on a hard-masked student token
    assert math.isfinite(float(per[1]))                    # zero teacher tail against an empty student tail


def test_head_matches_independent_oracle(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    subprocess.run([sys.executable, "-I", str(HERE / "s25_probe.py"), str(ROOT), str(out)], check=True, env=env)
    results = json.loads(out.read_text())["results"]
    for r in results:
        if r["case"] == "hard_mask_student":
            assert "error" not in r
            for got, want in zip(r["got"], r["oracle"]):
                assert (math.isinf(got) and math.isinf(want) and got > 0) or abs(got - want) <= 1e-12
            continue
        if r["backend"] == "EVAL" and r["case"] == "missing":
            # bucket_kl requires real ids; callers pass only scored rows, where present == position_mask.
            assert r.get("error", "").startswith("RuntimeError")
            continue
        assert "error" not in r, r
        if r["backend"] == "EVAL":
            assert r["finite"] and r["abs_err"] <= 2e-6, r
            continue
        assert r["loss_abs_err"] <= 1e-6 and r["per_abs_err"] <= 3e-6, r
        for name in ("hidden", "embed", "markov"):
            assert r[f"grad_{name}_finite"] and r[f"grad_{name}_rel_err"] <= 5e-6, (name, r)


def test_ref_and_fast_agree_with_oracle_in_logit_space() -> None:
    spec = importlib.util.spec_from_file_location("s25_agree", HERE / "s25_agree.py")
    assert spec is not None and spec.loader is not None
    agree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agree)
    _, worst = agree.sweep(range(120), identity=True)
    assert worst["loss_rel"] <= 1e-6 and worst["per_abs"] <= 1e-5, worst
    # FP32 accumulation on |dlogit| up to ~1.3 (2.3x incoming gradient); FP64 oracle is exact here.
    assert worst["dlogit_ref_oracle"] <= 5e-6 and worst["dlogit_fast_oracle"] <= 5e-6, worst
    assert worst["dlogit_ref_fast"] <= 5e-6, worst
