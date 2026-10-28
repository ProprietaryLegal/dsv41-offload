"""Multi-round speculative generation through the image's real verifier, on CPU (AUD1 item 1).

Each trial is one request: a prompt token, then rounds of (deterministic drafts -> the image's
rejection_sample() -> commit) until `length` tokens are generated, exactly as the V2 model runner
commits them (num_computed_tokens += accepted + 1; the next round's first row is the last committed
token; row i is keyed by position = start + i, model_runner/input_batch.prepare_pos_seq_lens).
Production's remote-DSpark shape: draft_logits=None (one-hot q), standard verification, T=1,
top_p=1, an unseeded request seed (random int64), a per-round draft window (0047 adaptive window)
and depth-policy dead rows passed as -1 (model_runner.sample -> invalid_drafts).

The target is a context-dependent toy (a Markov chain on a small vocabulary) so the exact joint
law of the generated tokens is known; the drafter is a deterministic function of the committed
context. Only synthetic numbers; no model, prompt or generated text from production.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from kernel_loader import DEFAULT_SRC, load

# vLLM #61161 hunk (rejection_sampler_utils.py, _rejection_kernel), applied to a copy for the
# positive control only. spec_step_seed is the PR's gumbel.py helper, defined in the copy.
_PRE = "            u = tl_rand32(seed, pos, includes_zero=False)\n"
_POST = (
    "            step_seed = spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed\n"
    "            u = tl_rand32(step_seed, pos, includes_zero=False)\n"
)
_HELPER = (
    "\n\n@triton.jit\n"
    "def spec_step_seed(seed, step):\n"
    "    return seed ^ (tl.cast(step, tl.int64) << 32)\n"
)
_ANCHOR = "from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n"


def write_pr61161_variant(dest: Path, src_root: Path = DEFAULT_SRC) -> Path:
    """Copy the image's rejection_sampler_utils.py with #61161's kernel change (control only)."""
    text = (src_root / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py").read_text()
    if text.count(_PRE) != 1 or text.count(_ANCHOR) != 1:
        raise RuntimeError("image kernel differs from the audited pre-#61161 source")
    out = dest / "rejection_sampler_utils_pr61161.py"
    out.write_text(text.replace(_ANCHOR, _ANCHOR + _HELPER).replace(_PRE, _POST))
    return out


@dataclass(frozen=True)
class Toy:
    """Context-dependent target p(x | prev), deterministic drafter, per-round window/depth."""

    target: tuple[tuple[float, ...], ...]  # rows: prev token -> distribution over the vocab
    draft_next: tuple[int, ...]  # deterministic proposal after token x
    max_k: int  # num_speculative_tokens (production: 5 at 1-2 streams, 4 at 3-4)
    length: int  # generated tokens per trial
    vary_window: bool = True  # 0047-style per-round window from committed history
    dead_rows: bool = True  # depth-policy rows past the depth -> -1
    prompt_token: int = 0
    prompt_pos: int = 37  # position of the prompt's last token (any offset)

    @property
    def vocab(self) -> int:
        return len(self.target)

    def joint(self) -> dict[tuple[int, ...], float]:
        """Exact law of the first `length` generated tokens."""
        out: dict[tuple[int, ...], float] = {(): 1.0}
        for _ in range(self.length):
            nxt: dict[tuple[int, ...], float] = {}
            for seq, pr in out.items():
                prev = seq[-1] if seq else self.prompt_token
                for x, px in enumerate(self.target[prev]):
                    if px > 0:
                        nxt[seq + (x,)] = pr * px
            out = nxt
        return out


# The review's first case (#61161 validation): context-free p=(.5,.5), drafter always 0, k=2.
PR_CASE = Toy(target=((0.5, 0.5), (0.5, 0.5)), draft_next=(0, 0), max_k=2, length=6,
              vary_window=False, dead_rows=False)
# Production-shaped case: context-dependent target, a drafter that is right about half the time,
# windows 1..5 and dead rows chosen from committed history only.
PROD_CASE = Toy(target=((0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.3, 0.2, 0.5)),
                draft_next=(0, 2, 2), max_k=5, length=4)


@dataclass
class Result:
    counts: dict[tuple[int, ...], int] = field(default_factory=dict)
    trials: int = 0
    rounds: int = 0
    accepted: int = 0
    # acceptance-uniform keys already consulted in an earlier round for a position that was not
    # committed (the #61161 condition), per the kernel's read set. The key is (seed, position),
    # or (seed ^ row << 32, position) for block rounds in the #61161-fixed copy.
    reread_keys: int = 0

    def merge(self, other: Result) -> None:
        for key, value in other.counts.items():
            self.counts[key] = self.counts.get(key, 0) + value
        self.trials += other.trials
        self.rounds += other.rounds
        self.accepted += other.accepted
        self.reread_keys += other.reread_keys


def _window(toy: Toy, seq: list[int], rnd: int) -> tuple[int, int]:
    """(window k, depth d <= k) from committed history only, like 0047 / the depth policy."""
    if not toy.vary_window:
        return toy.max_k, toy.max_k
    h = (sum((i + 1) * t for i, t in enumerate(seq)) * 2654435761 + rnd * 40503) & 0xFFFF
    k = 1 + h % toy.max_k
    d = k if not toy.dead_rows else 1 + (h >> 4) % k
    return k, d


def _run_chunk(args: tuple[Toy, str, int, int, str | None]) -> Result:
    toy, mode, n, chunk_seed, variant = args
    module = load(Path(variant) if variant else None)
    step_keyed = variant is not None and mode == "block"
    rng = random.Random(chunk_seed)
    seeds = [rng.getrandbits(63) - (1 << 62) for _ in range(n)]
    log_target = torch.tensor(toy.target, dtype=torch.float32).log()
    seqs: list[list[int]] = [[toy.prompt_token] for _ in range(n)]
    consulted: list[set[tuple[int, int]]] = [set() for _ in range(n)]
    res = Result(trials=n)
    rnd = 0
    while True:
        active = [r for r in range(n) if len(seqs[r]) - 1 < toy.length]
        if not active:
            break
        rnd += 1
        logits_rows: list[torch.Tensor] = []
        draft_sampled: list[int] = []
        pos: list[int] = []
        cu = [0]
        local_pos: list[int] = []
        expanded: list[int] = []
        for b, r in enumerate(active):
            seq = seqs[r]
            k, d = _window(toy, seq, rnd)
            start = toy.prompt_pos + len(seq) - 1
            row_tokens = [seq[-1]]
            for _ in range(k):
                row_tokens.append(toy.draft_next[row_tokens[-1]])
            for i, tok in enumerate(row_tokens):
                logits_rows.append(log_target[tok])
                # rows past the depth are invalid drafts (-1), as model_runner.sample passes them
                draft_sampled.append(-1 if (i > d) else tok)
                pos.append(start + i)
                local_pos.append(i)
                expanded.append(b)
            cu.append(cu[-1] + k + 1)
        nb = len(active)
        sampled, num_sampled = module.rejection_sample(
            torch.stack(logits_rows).contiguous(),
            None,  # remote DSpark capacity: draft_logits None -> one-hot q
            torch.tensor(draft_sampled, dtype=torch.int64),
            torch.tensor(cu, dtype=torch.int32),
            torch.tensor(pos, dtype=torch.int64),
            torch.arange(nb, dtype=torch.int32),
            torch.tensor(expanded, dtype=torch.int32),
            torch.tensor(local_pos, dtype=torch.int32),
            torch.ones(nb, dtype=torch.float32),  # T=1
            torch.tensor([seeds[r] for r in active], dtype=torch.int64),
            toy.max_k,
            use_block_verification=(mode == "block"),
        )
        for b, r in enumerate(active):
            m = int(num_sampled[b])
            k = cu[b + 1] - cu[b] - 1
            start = pos[cu[b]]
            # Rows whose acceptance uniform the kernel reads (rejection_sampler_utils.py
            # _rejection_kernel): standard stops after the first rejection or -1; block reads every
            # row up to the first -1. Rows are keyed by (seed, start + i).
            valid = 0
            while valid < k and draft_sampled[cu[b] + valid + 1] >= 0:
                valid += 1
            read_rows = valid if mode == "block" else min(m, valid)
            for i in range(read_rows):
                key = (i if step_keyed else 0, start + i)
                if key in consulted[r]:
                    res.reread_keys += 1
                consulted[r].add(key)
            new = [int(t) for t in sampled[b, :m]]
            seqs[r].extend(new)
            res.accepted += m - 1
            # committed positions can never be consulted again; drop them from the watch set
            committed_upto = start + m - 1
            consulted[r] = {c for c in consulted[r] if c[1] > committed_upto}
        res.rounds += 1
    for seq in seqs:
        gen = tuple(seq[1 : 1 + toy.length])
        res.counts[gen] = res.counts.get(gen, 0) + 1
    return res


def simulate(toy: Toy, mode: str, trials: int, seed: int, workers: int = 8,
             variant: Path | None = None) -> Result:
    if mode not in ("standard", "block"):
        raise ValueError(mode)
    chunks = max(1, min(workers, trials))
    sizes = [trials // chunks + (1 if i < trials % chunks else 0) for i in range(chunks)]
    jobs = [(toy, mode, s, seed * 1000 + i, str(variant) if variant else None)
            for i, s in enumerate(sizes) if s]
    total = Result()
    if len(jobs) == 1:
        total.merge(_run_chunk(jobs[0]))
        return total
    ctx = mp.get_context("spawn")
    env_ok = os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    if not env_ok:
        raise RuntimeError("TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES= are required")
    with ctx.Pool(len(jobs)) as pool:
        for part in pool.map(_run_chunk, jobs):
            total.merge(part)
    return total


def chi2_threshold(dof: int, alpha: float) -> float:
    """Wilson-Hilferty upper quantile of chi-square(dof)."""
    z = _norm_ppf(1.0 - alpha)
    c = 2.0 / (9.0 * dof)
    return dof * (1.0 - c + z * math.sqrt(c)) ** 3


def _norm_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_joint(toy: Toy, res: Result) -> tuple[float, int]:
    exact = toy.joint()
    stat = 0.0
    for key, pr in exact.items():
        e = pr * res.trials
        stat += (res.counts.get(key, 0) - e) ** 2 / e
    if any(key not in exact for key in res.counts):
        stat = math.inf
    return stat, len(exact) - 1


def marginals(toy: Toy, res: Result, token: int) -> list[tuple[float, float]]:
    """(observed P(x_t = token), exact) per generated position t."""
    exact = toy.joint()
    out = []
    for t in range(toy.length):
        obs = sum(c for k, c in res.counts.items() if k[t] == token) / res.trials
        ex = sum(p for k, p in exact.items() if k[t] == token)
        out.append((obs, ex))
    return out


def summary(toy: Toy, res: Result) -> dict[str, Any]:
    stat, dof = chi2_joint(toy, res)
    return {"trials": res.trials, "rounds": res.rounds, "accepted_drafts": res.accepted,
            "reread_keys": res.reread_keys, "chi2": round(stat, 2), "dof": dof,
            "marginals_token0": [(round(o, 4), round(e, 4)) for o, e in marginals(toy, res, 0)]}
