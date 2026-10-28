"""S25 item 2: randomized REF (distillation_loss reference path) vs FAST (bucket_loss) agreement against the
independent FP64 oracle, in absolute dloss/dlogit space (identity head: z = hidden exactly).

  python -I s25_agree.py ROOT OUT.pt [--identity]     (CPU, synthetic tensors only)

Relative-to-max gradient error is NOT used as the criterion: with sharp distributions the whole gradient is
~1e-6 and FP32 cancellation (e.g. TV's a_T - sum a q with q_T ~ 1) dominates any relative metric in BOTH paths.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from offload.dspark.d2.train.fast_loss import bucket_loss  # noqa: E402
from offload.dspark.d2.train.loss import distillation_loss  # noqa: E402

_spec = importlib.util.spec_from_file_location("s25_oracle", Path(__file__).with_name("s25_oracle.py"))
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


class Head:
    def __init__(self, head: Tensor, embed: Tensor, markov: Tensor) -> None:
        self.head, self.embed, self.markov = head, embed, markov

    @property
    def c(self) -> Any:
        return type("C", (), {"vocab": self.head.shape[0]})()

    def logit_chunk(self, hidden: Tensor, previous: Tensor, start: int, end: int) -> Tensor:
        return F.linear(hidden, self.head[start:end]) + F.linear(F.embedding(previous, self.embed), self.markov[start:end])


def case(seed: int, identity: bool) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    V = int(torch.randint(21, 300, (1,), generator=g))
    k = int(torch.randint(1, 21, (1,), generator=g))
    B, H, E = int(torch.randint(1, 4, (1,), generator=g)), 8, 4
    temp = float(torch.empty(1).uniform_(.3, 6, generator=g))
    c: dict[str, Any] = {"h": torch.randn(B, 5, H, generator=g), "head": torch.randn(V, H, generator=g) * temp,
                         "embed": torch.randn(V, E, generator=g), "markov": torch.randn(V, E, generator=g),
                         "previous": torch.randint(0, 4, (B, 5), generator=g)}
    if identity:
        c["h"] = torch.randn(B, 5, V, generator=g) * temp
        c["head"], c["markov"] = torch.eye(V), torch.zeros(V, E)
    ids = torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(B * 5)]).reshape(B, 5, k)
    teacher = (torch.randn(B, 5, V, generator=g, dtype=torch.float64) * temp).softmax(-1)
    lp = teacher.gather(-1, ids).log().float()
    tail = (1 - lp.double().exp().sum(-1)).clamp(0, 1).float()   # replay.parse_logprobs convention
    present = torch.ones(B, 5, dtype=torch.bool)
    n = torch.randint(1, 6, (B,), generator=g)
    labels = torch.randint(0, V, (B, 5), generator=g)
    for b in range(B):  # data.py padding convention
        present[b, n[b]:] = False
        ids[b, n[b]:] = -1
        lp[b, n[b]:] = -torch.inf
        tail[b, n[b]:] = 1
        labels[b, n[b]:] = -1
    mask = present.clone()
    if seed % 5 == 0 and bool(mask.sum() > 1):  # missing-top20 fallback on one scored position
        b, j = torch.nonzero(mask)[0].tolist()
        present[b, j] = False
        ids[b, j] = -1
        lp[b, j] = -torch.inf
        tail[b, j] = 1
    c.update(ids=ids, lp=lp, tail=tail, labels=labels, present=present, mask=mask,
             chunk=int(torch.randint(1, V + 5, (1,), generator=g)), divergence="kl" if seed % 3 else "tvd")
    return c


def run(c: dict[str, Any], backend: str) -> tuple[Tensor, Tensor, list[Tensor]]:
    h = c["h"].clone().requires_grad_(True)
    e = c["embed"].clone().requires_grad_(True)
    m = c["markov"].clone().requires_grad_(True)
    a = (c["previous"], c["ids"], c["lp"], c["tail"], c["labels"], c["present"], c["mask"])
    if backend == "REF":
        loss, per = distillation_loss(Head(c["head"], e, m), h, *a, chunk=c["chunk"], divergence=c["divergence"])
    else:
        loss, per = bucket_loss(h, c["head"], e, m, *a, c["chunk"], c["divergence"])
    g = torch.autograd.grad(2.3 * loss, (h, e, m))
    return loss.detach(), per.detach(), [x.detach() for x in g]


def sweep(seeds: range, identity: bool) -> tuple[list[dict[str, Any]], dict[str, float]]:
    out, worst = [], {"loss_rel": 0., "per_abs": 0., "dlogit_ref_fast": 0., "dlogit_ref_oracle": 0.,
                      "dlogit_fast_oracle": 0.}
    for seed in seeds:
        c = case(seed, identity)
        r, f = run(c, "REF"), run(c, "FAST")
        m = c["mask"]
        worst["loss_rel"] = max(worst["loss_rel"], float((r[0] - f[0]).abs() / r[0].abs().clamp_min(1e-12)))
        worst["per_abs"] = max(worst["per_abs"], float((r[1][m] - f[1][m]).abs().max()))
        if identity:
            p = c["lp"].double().exp().masked_fill(c["ids"] < 0, 0)
            _, _, g = oracle.objective(c["h"], c["head"], c["embed"], c["markov"], c["previous"], c["ids"], p,
                                       c["lp"].double(), c["tail"].double(), c["labels"], c["present"], c["mask"],
                                       c["divergence"], 2.3)
            worst["dlogit_ref_fast"] = max(worst["dlogit_ref_fast"], float((r[2][0] - f[2][0]).abs().max()))
            worst["dlogit_ref_oracle"] = max(worst["dlogit_ref_oracle"], float((r[2][0].double() - g["hidden"]).abs().max()))
            worst["dlogit_fast_oracle"] = max(worst["dlogit_fast_oracle"], float((f[2][0].double() - g["hidden"]).abs().max()))
        out.append({"seed": seed, "ref": r, "fast": f})
    return out, worst


if __name__ == "__main__":
    torch.set_num_threads(4)
    results, worst = sweep(range(300), "--identity" in sys.argv)
    torch.save(results, sys.argv[2])
    print({"cases": len(results), "worst": worst})
