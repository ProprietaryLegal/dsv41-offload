"""PP4's verifier under multi-round sampled drafting at T=1, on CPU (R-PP4: the #61161 check on PP4's own path).

The kernel is the image's rejection_sampler_utils.py transformed by PP3's pinned prepare_verifier.utilities, which is
exactly the file PP4's overlay ships (pp4_prepare writes the same text): full-vocabulary FP32 probabilities q passed as
draft_logits (HAS_DRAFT_LOGITS == 2), standard verification only. Each trial drafts from a context-dependent proposal q
that differs from the target, samples every draft from q with a random stream owned by its round (never keyed by
position, the rule #61161 states for revisited-position draft noise), passes q's rows and commits exactly as the V2
model runner does (multiround.py). Speculative sampling is exact only if the generated tokens follow the target's law;
#61161's reuse would also show as re-read acceptance keys. Synthetic numbers only.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _path in (str(_HERE), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import multiround as mr
from kernel_loader import DEFAULT_SRC, load

SAMPLED_ENV = "VLLM_DSPARK_SAMPLED_TOP32"


@dataclass(frozen=True)
class SampledCase:
    """Target law and window/depth policy from `toy` (its draft_next is unused); drafts are drawn from `sample_q`."""

    toy: mr.Toy
    q: tuple[tuple[float, ...], ...]  # proposal row passed to the verifier: previous token -> q over the vocab
    sample_q: tuple[tuple[float, ...], ...] | None = None  # negative control only: draw from a different law than q


_TARGET = ((0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.3, 0.2, 0.5))
# q differs from the target on every row and gives token 2 zero mass after token 0 (disjoint support: log q = -inf).
_Q = ((0.7, 0.3, 0.0), (0.1, 0.6, 0.3), (0.2, 0.2, 0.6))
# Production-shaped: context-dependent target, sampled drafter, windows 1..5 and depth dead rows from committed history.
PP4_CASE = SampledCase(mr.Toy(target=_TARGET, draft_next=(0, 0, 0), max_k=5, length=4), q=_Q)
# PP4's measured configuration: fixed depth 5, every row valid.
PP4_FIXED5 = SampledCase(mr.Toy(target=_TARGET, draft_next=(0, 0, 0), max_k=5, length=4,
                                vary_window=False, dead_rows=False), q=_Q)
# Negative control: the verifier is told q but the drafts come from another law, which must bias the output.
MISMATCH = SampledCase(PP4_FIXED5.toy, q=_Q, sample_q=((0.1, 0.1, 0.8), (0.8, 0.1, 0.1), (0.1, 0.8, 0.1)))


def write_pp4_kernel(dest: Path, src_root: Path = DEFAULT_SRC) -> Path:
    """The PP4 verifier kernel, generated from the pinned image source by PP3's reviewed transformation."""
    from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3

    data = (src_root / pp3.UTILS).read_bytes()
    if hashlib.sha256(data).hexdigest() != pp3.PINS[pp3.UTILS]:
        raise RuntimeError("image kernel differs from PP3's pinned source")
    out = dest / "rejection_sampler_utils_pp4.py"
    out.write_text(pp3.utilities(data.decode()))
    return out


def _run_chunk(args: tuple[SampledCase, int, int, str]) -> mr.Result:
    case, n, chunk_seed, kernel = args
    os.environ[SAMPLED_ENV] = "1"
    module = load(Path(kernel))
    import triton.language as tl  # type: ignore[import-untyped]

    class _InterpreterDevice:
        # Triton's CPU interpreter has no libdevice (its log1p returns None). The q residual's only libdevice call,
        # log1p(-ratio), becomes log(1 - ratio): the same law at CPU rounding (GPU rounding is not part of the claim).
        @staticmethod
        def log1p(x: tl.tensor) -> tl.tensor:
            return tl.log(1.0 + x)

    module.tldevice = _InterpreterDevice()
    toy = case.toy
    draw_q = case.sample_q or case.q
    rng = random.Random(chunk_seed)
    seeds = [rng.getrandbits(63) - (1 << 62) for _ in range(n)]
    log_target = torch.tensor(toy.target, dtype=torch.float32).log()
    q_rows = torch.tensor(case.q, dtype=torch.float32)
    seqs: list[list[int]] = [[toy.prompt_token] for _ in range(n)]
    consulted: list[set[tuple[int, int]]] = [set() for _ in range(n)]
    res = mr.Result(trials=n)
    rnd = 0
    while True:
        active = [r for r in range(n) if len(seqs[r]) - 1 < toy.length]
        if not active:
            break
        rnd += 1
        nb = len(active)
        q = torch.zeros(nb, toy.max_k, toy.vocab, dtype=torch.float32)
        logits_rows: list[torch.Tensor] = []
        draft_sampled: list[int] = []
        pos: list[int] = []
        cu = [0]
        local_pos: list[int] = []
        expanded: list[int] = []
        for b, r in enumerate(active):
            seq = seqs[r]
            k, d = mr._window(toy, seq, rnd)
            start = toy.prompt_pos + len(seq) - 1
            # One draft stream per (request, round): a later round re-drafts a revisited position from new randomness.
            stream = random.Random(f"{seeds[r]}:{rnd}")
            row_tokens = [seq[-1]]
            for i in range(k):
                prev = row_tokens[-1]
                q[b, i] = q_rows[prev]
                row_tokens.append(stream.choices(range(toy.vocab), weights=draw_q[prev])[0])
            for i, tok in enumerate(row_tokens):
                logits_rows.append(log_target[tok])
                draft_sampled.append(-1 if i > d else tok)
                pos.append(start + i)
                local_pos.append(i)
                expanded.append(b)
            cu.append(cu[-1] + k + 1)
        sampled, num_sampled = module.rejection_sample(
            torch.stack(logits_rows).contiguous(),
            q,  # PP4: full-vocabulary FP32 q rows, HAS_DRAFT_LOGITS == 2
            torch.tensor(draft_sampled, dtype=torch.int64),
            torch.tensor(cu, dtype=torch.int32),
            torch.tensor(pos, dtype=torch.int64),
            torch.arange(nb, dtype=torch.int32),
            torch.tensor(expanded, dtype=torch.int32),
            torch.tensor(local_pos, dtype=torch.int32),
            torch.ones(nb, dtype=torch.float32),  # T=1
            torch.tensor([seeds[r] for r in active], dtype=torch.int64),
            toy.max_k,
            use_block_verification=False,
        )
        for b, r in enumerate(active):
            m = int(num_sampled[b])
            k = cu[b + 1] - cu[b] - 1
            start = pos[cu[b]]
            # Standard verification reads row i's acceptance uniform, keyed (seed, start + i), only while verifying.
            valid = 0
            while valid < k and draft_sampled[cu[b] + valid + 1] >= 0:
                valid += 1
            for i in range(min(m, valid)):
                key = (0, start + i)
                if key in consulted[r]:
                    res.reread_keys += 1
                consulted[r].add(key)
            seqs[r].extend(int(t) for t in sampled[b, :m])
            res.accepted += m - 1
            committed_upto = start + m - 1
            consulted[r] = {c for c in consulted[r] if c[1] > committed_upto}
        res.rounds += 1
    for seq in seqs:
        gen = tuple(seq[1 : 1 + toy.length])
        res.counts[gen] = res.counts.get(gen, 0) + 1
    return res


def simulate(case: SampledCase, trials: int, seed: int, kernel: Path, workers: int = 8) -> mr.Result:
    if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES= are required")
    chunks = max(1, min(workers, trials))
    sizes = [trials // chunks + (1 if i < trials % chunks else 0) for i in range(chunks)]
    jobs = [(case, s, seed * 1000 + i, str(kernel)) for i, s in enumerate(sizes) if s]
    total = mr.Result()
    if len(jobs) == 1:
        total.merge(_run_chunk(jobs[0]))
        return total
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for part in pool.map(_run_chunk, jobs):
            total.merge(part)
    return total
