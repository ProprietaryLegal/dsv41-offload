"""S25 item 4: independent checks of vocab_audit.audit_logits.
  python -I s25_vocab_check.py ROOT dense|memory|effects
dense   : random hard masks on both sides, -1 padding, non-dividing chunks, vs my own dense FP64 formulas
memory  : peak RSS at vocab 129280 with an on-the-fly provider vs the dense rows x vocab FP64 size
effects : audit hook records open/socket/subprocess events during import and during the audit call
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
EVENTS: list[str] = []
PHASE = ["import"]
if sys.argv[2] == "effects":
    def hook(event: str, args: tuple) -> None:
        if event.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn")) or (
                event == "open" and PHASE[0] == "call"):
            EVENTS.append(f"{PHASE[0]}:{event}:{str(args)[:80]}")
    sys.addaudithook(hook)
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import math  # noqa: E402

import torch  # noqa: E402

from offload.dspark.d2.train.vocab_audit import audit_logits  # noqa: E402


def dense(t, s, top, lab, prop):
    lp, lq = t.log_softmax(-1), s.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    out = {"teacher_logz": t.logsumexp(-1), "student_logz": s.logsumexp(-1)}
    kl = []
    for r in range(t.shape[0]):
        acc = 0.
        for j in range(t.shape[1]):
            if p[r, j] > 0:
                acc += math.inf if math.isinf(float(lq[r, j])) else float(p[r, j] * (lp[r, j] - lq[r, j]))
        kl.append(acc)
    out["full_kl"] = torch.tensor(kl, dtype=torch.float64)
    out["full_tv"] = .5 * (p - q).abs().sum(-1)
    S = torch.zeros_like(p, dtype=torch.bool)
    for r in range(top.shape[0]):
        for i in top[r].tolist():
            if i >= 0:
                S[r, i] = True
    pt, qt = p.masked_fill(S, 0).sum(-1), q.masked_fill(S, 0).sum(-1)
    lpt, lqt = lp.masked_fill(S, -math.inf).logsumexp(-1), lq.masked_fill(S, -math.inf).logsumexp(-1)
    out["teacher_tail"], out["student_tail"] = pt, qt
    bkl, btv = [], []
    for r in range(top.shape[0]):
        acc, tv = 0., 0.
        for i in top[r].tolist():
            if i < 0:
                continue
            pi, qi = float(p[r, i]), float(q[r, i])
            tv += abs(pi - qi)
            if pi > 0:
                acc += math.inf if math.isinf(float(lq[r, i])) else pi * (float(lp[r, i]) - float(lq[r, i]))
        a, b = float(pt[r]), float(qt[r])
        tv += abs(a - b)
        if a > 0:
            acc += math.inf if math.isinf(float(lqt[r])) else a * (float(lpt[r]) - float(lqt[r]))
        bkl.append(acc)
        btv.append(.5 * tv)
    out["bucket_kl"], out["bucket_tv"] = torch.tensor(bkl, dtype=torch.float64), torch.tensor(btv, dtype=torch.float64)
    out["teacher_sample_ce"] = -lp.gather(-1, lab[:, None]).squeeze(-1)
    out["student_sample_ce"] = -lq.gather(-1, lab[:, None]).squeeze(-1)
    out["proposal_p"], out["proposal_q"] = p.gather(-1, prop[:, None]).squeeze(-1), q.gather(-1, prop[:, None]).squeeze(-1)
    return out


def close(a, b):
    a, b = a.double(), b.double()
    same_inf = (torch.isinf(a) & torch.isinf(b) & (a.sign() == b.sign()))
    fin = ~same_inf
    if bool((torch.isinf(a) | torch.isinf(b))[fin].any()) or bool(torch.isnan(a).any()):
        return math.inf
    return float(((a - b).abs() / (1 + b.abs()))[fin].max()) if bool(fin.any()) else 0.


mode = sys.argv[2]
if mode == "dense":
    worst, n, infs = 0., 0, 0
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        V = int(torch.randint(2, 60, (1,), generator=g))
        R = int(torch.randint(1, 6, (1,), generator=g))
        k = int(torch.randint(1, V + 1, (1,), generator=g))
        t = torch.randn(R, V, generator=g, dtype=torch.float64) * 4 + float(torch.randn(1, generator=g) * 500)
        s = torch.randn(R, V, generator=g, dtype=torch.float64) * 4 - float(torch.randn(1, generator=g) * 500)
        for x in (t, s):  # hard masks, never a whole row
            m = torch.rand(R, V, generator=g) < float(torch.rand(1, generator=g)) * .6
            m[:, int(torch.randint(0, V, (1,), generator=g))] = False
            x.masked_fill_(m, -math.inf)
        s[:, int(t[0].argmax())] = s[:, int(t[0].argmax())].clamp_min(-1e3)  # keep some overlap
        top = torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(R)])
        top = top.masked_fill(torch.rand(R, k, generator=g) < .2, -1)  # padding
        lab = torch.randint(0, V, (R,), generator=g)
        prop = torch.randint(0, V, (R,), generator=g)
        if bool(torch.isneginf(s).all(-1).any()) or bool(torch.isneginf(t).all(-1).any()):
            continue
        chunk = int(torch.randint(1, V + 3, (1,), generator=g))
        got = audit_logits(lambda lo, hi: t[:, lo:hi], lambda lo, hi: s[:, lo:hi], top, lab, prop, V, chunk)
        want = dense(t, s, top, lab, prop)
        for name, w in want.items():
            e = close(getattr(got, name), w)
            if e > 1e-9 and os.environ.get("S25_VERBOSE"):
                print("MISMATCH", seed, name, getattr(got, name).tolist(), w.tolist())
            worst = max(worst, e)
        infs += int(torch.isinf(got.full_kl).sum())
        n += 1
    print({"cases": n, "worst_rel_err": worst, "rows_with_inf_full_kl": infs})
elif mode == "memory":
    R, V, C = 256, int(os.environ.get("S25_VOCAB", "129280")), int(os.environ.get("S25_CHUNK", "4096"))

    def provider(seed: int):
        def f(lo: int, hi: int) -> torch.Tensor:
            g = torch.Generator().manual_seed(seed * 1_000_003 + lo)
            return torch.randn(R, hi - lo, generator=g) * 3
        return f
    torch.set_num_threads(4)
    top = torch.stack([torch.randperm(V)[:20] for _ in range(R)])
    lab = torch.randint(0, V, (R,))
    import threading
    import time

    def rss() -> int:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    provider(1)(0, C), provider(2)(0, C)  # warm allocator
    base, peak, done = rss(), [0], threading.Event()

    def sample() -> None:
        while not done.is_set():
            peak[0] = max(peak[0], rss())
            time.sleep(.0005)
    th = threading.Thread(target=sample)
    th.start()
    got = audit_logits(provider(1), provider(2), top, lab, lab, V, C)
    done.set()
    th.join()
    print({"rows": R, "vocab": V, "chunk": C, "peak_rss_growth_MiB": round((peak[0] - base) / 2**20, 1),
           "dense_fp64_rows_x_vocab_MiB": round(R * V * 8 / 2**20, 1), "one_chunk_fp64_MiB": round(R * C * 8 / 2**20, 1),
           "finite": bool(torch.isfinite(got.full_kl).all())})
else:
    PHASE[0] = "call"
    z = torch.randn(4, 50, dtype=torch.float64)
    audit_logits(lambda lo, hi: z[:, lo:hi], lambda lo, hi: z[:, lo:hi] * .5, torch.tensor([[0, 1], [2, -1], [3, 4], [5, 6]]),
                 torch.tensor([0, 1, 2, 3]), torch.tensor([9, 8, 7, 6]), 50, 7)
    net = [m for m in sys.modules if m.split(".")[0] in ("vllm", "requests", "urllib3", "httpx", "aiohttp", "zmq", "grpc")]
    print({"events": EVENTS, "network_client_modules_loaded": net})
