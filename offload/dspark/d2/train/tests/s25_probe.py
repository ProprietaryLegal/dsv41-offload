"""S25 probe: run REF (distillation_loss), FAST (bucket_loss) and EVAL (bucket_kl) from ROOT against the
independent FP64 oracle in s25_oracle.py, on generic and boundary inputs.

  python -I s25_probe.py ROOT [OUT.json]

ROOT is a tree containing offload/dspark/d2/train (e.g. a `git archive` export of cfe04a3 or 1fdda56).
CPU only; synthetic tensors only.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("s25_oracle", Path(__file__).with_name("s25_oracle.py"))
assert _spec and _spec.loader
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from offload.dspark.d2.train.evaluate import bucket_kl  # noqa: E402
from offload.dspark.d2.train.fast_loss import bucket_loss  # noqa: E402
from offload.dspark.d2.train.loss import distillation_loss  # noqa: E402

V, H, E = 13, 4, 3
SCALE = 3.7


@dataclass
class Vocabulary:
    vocab: int


class Head:
    def __init__(self, head: Tensor, embed: Tensor, markov: Tensor) -> None:
        self.head, self.embed, self.markov = head, embed, markov

    @property
    def c(self) -> Vocabulary:
        return Vocabulary(self.head.shape[0])

    def logit_chunk(self, hidden: Tensor, previous: Tensor, start: int, end: int) -> Tensor:
        return F.linear(hidden, self.head[start:end]) + F.linear(F.embedding(previous, self.embed), self.markov[start:end])


def make_case(name: str, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = V if name == "full_support" else 4
    h = torch.randn(2, 5, H, generator=g)
    head = torch.randn(V, H, generator=g)
    embed = torch.randn(V, E, generator=g)
    markov = torch.randn(V, E, generator=g)
    previous = torch.randint(0, 3, (2, 5), generator=g)  # repeated previous ids
    ids = torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(10)]).reshape(2, 5, k)
    if name == "leading_covered_chunk":
        ids = torch.tensor([0, 1, 2, 9]).expand(2, 5, 4).clone()
    teacher = (2 * torch.randn(2, 5, V, generator=g, dtype=torch.float64)).softmax(-1)
    p = teacher.gather(-1, ids)
    tail = 1 - p.sum(-1)
    if name == "zero_top_mass":
        tail = tail + p[..., 1]
        p[..., 1] = 0
    if name == "zero_tail":
        p = p / p.sum(-1, keepdim=True)
        tail = torch.zeros_like(tail)
    if name == "full_support":
        p = p / p.sum(-1, keepdim=True)
        tail = torch.zeros_like(tail)
    if name == "rounded_mass":
        tail = tail + 8e-5  # parser/loss admit |sum p + p_T - 1| <= 1e-4
    if name == "rounded_mass_realistic":
        tail = tail + 1e-6
    logp = p.log()
    if name == "underflow_top":
        logp[..., 1] = -120.  # finite logprob whose FP32 exp underflows to 0
        tail = tail + p[..., 1]
        p = logp.float().exp().double()
    labels = torch.randint(0, V, (2, 5), generator=g)
    present = torch.ones(2, 5, dtype=torch.bool)
    mask = present.clone()
    if name in ("missing", "padding"):
        # reader padding convention (data.py): ids -1, lp -inf, resid 1, present=mask=False, labels -1
        mask[0, 3:] = False
        present[0, 3:] = False
        ids[0, 3:] = -1
        logp[0, 3:] = -math.inf
        p[0, 3:] = 0
        tail[0, 3:] = 1
        labels[0, 3:] = -1
    if name == "missing":
        present[1, 2] = False
        ids[1, 2] = -1
        logp[1, 2] = -math.inf
        p[1, 2] = 0
        tail[1, 2] = 1
        labels[1, 2] = int(torch.tensor([t for t in range(V) if t not in ids[1, 1].tolist()][0]))
    return {"h": h, "head": head, "embed": embed, "markov": markov, "previous": previous, "ids": ids,
            "p": p.float(), "logp": logp.float(), "tail": tail.float(), "labels": labels, "present": present, "mask": mask}


def rel(a: Tensor, b: Tensor) -> float:
    a, b = a.double(), b.double()
    if not bool(torch.isfinite(a).all()):
        return math.nan
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def run_backend(backend: str, c: dict[str, Tensor], divergence: str, chunk: int) -> dict[str, object]:
    h = c["h"].clone().requires_grad_(True)
    embed = c["embed"].clone().requires_grad_(True)
    markov = c["markov"].clone().requires_grad_(True)
    args = (c["previous"], c["ids"], c["logp"], c["tail"], c["labels"], c["present"], c["mask"])
    want_loss, want_per, want_grad = oracle.objective(c["h"], c["head"], c["embed"], c["markov"], c["previous"],
                                                      c["ids"], c["p"], c["logp"], c["tail"], c["labels"],
                                                      c["present"], c["mask"], divergence, SCALE)
    out: dict[str, object] = {"oracle_loss": float(want_loss)}
    try:
        if backend == "REF":
            loss, per = distillation_loss(Head(c["head"], embed, markov), h, *args, chunk=chunk,
                                          divergence=divergence)  # type: ignore[arg-type]
        else:
            loss, per = bucket_loss(h, c["head"], embed, markov, *args, chunk, divergence)  # type: ignore[arg-type]
    except Exception as exc:  # recorded, never hidden
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["loss"] = float(loss)
    out["loss_abs_err"] = abs(float(loss) - float(want_loss))
    m = c["mask"]
    out["per_abs_err"] = float((per.double()[m] - want_per[m]).abs().max())
    grads = torch.autograd.grad(SCALE * loss, (h, embed, markov), allow_unused=True)
    for name, got in zip(("hidden", "embed", "markov"), grads):
        if got is None:
            out[f"grad_{name}"] = "none"
            continue
        out[f"grad_{name}_finite"] = bool(torch.isfinite(got).all())
        out[f"grad_{name}_rel_err"] = rel(got, want_grad[name])
    return out


def run_eval(c: dict[str, Tensor]) -> dict[str, object]:
    z = oracle.logits(c["h"], c["head"], c["embed"], c["markov"], c["previous"])
    m = c["mask"]
    want, _ = oracle.per_position(z, c["ids"], c["p"], c["logp"], c["tail"], c["labels"], c["present"], m, "kl")
    try:
        got = bucket_kl(z.float()[m], c["ids"][m], c["logp"][m], c["tail"][m])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"finite": bool(torch.isfinite(got).all()), "abs_err": float((got.double() - want[m]).abs().max())}


def hard_mask_eval() -> dict[str, object]:
    """Evaluator/audit-only boundary: student hard -inf logits (training logits are always finite)."""
    z = torch.tensor([[1.0, -math.inf, 0.5, -math.inf, 0.2],     # zero teacher on masked support token
                      [1.0, -math.inf, 0.5, -math.inf, 0.2],     # POSITIVE teacher on masked support token
                      [1.0, 0.3, -math.inf, -math.inf, -math.inf],  # empty student tail, zero teacher tail
                      [1.0, 0.3, -math.inf, -math.inf, -math.inf]], dtype=torch.float64)  # positive teacher tail
    ids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 4], [0, 1, 4]])
    p = torch.tensor([[.6, 0, .3], [.5, .1, .3], [.7, .3, 0], [.6, .2, 0]], dtype=torch.float64)
    tail = torch.tensor([.1, .1, 0, .2], dtype=torch.float64)
    T = torch.ones(4, dtype=torch.bool)
    want, _ = oracle.per_position(z, ids, p, p.log(), tail, torch.zeros(4, dtype=torch.long), T, T, "kl")
    try:
        got = bucket_kl(z, ids, p.log(), tail)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"got": [float(x) for x in got], "oracle": [float(x) for x in want]}


CASES = ("generic", "padding", "missing", "zero_top_mass", "underflow_top", "zero_tail", "full_support",
         "leading_covered_chunk", "rounded_mass", "rounded_mass_realistic")


def main() -> None:
    torch.set_num_threads(2)
    results = []
    for name in CASES:
        for seed in (1, 2):
            c = make_case(name, seed)
            ev = run_eval(c)
            results.append({"case": name, "seed": seed, "backend": "EVAL", "divergence": "kl", **ev})
            for divergence in ("kl", "tvd"):
                for chunk in (1, 2, 3, 5, 13, 4096):
                    for backend in ("REF", "FAST"):
                        r = run_backend(backend, c, divergence, chunk)
                        results.append({"case": name, "seed": seed, "backend": backend, "divergence": divergence,
                                        "chunk": chunk, **r})
    results.append({"case": "hard_mask_student", "backend": "EVAL", **hard_mask_eval()})
    text = json.dumps({"root": str(ROOT), "torch": torch.__version__, "results": results}, indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
