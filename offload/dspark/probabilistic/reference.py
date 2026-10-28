"""Exact finite-vocabulary speculative sampling, standard tokenwise verification.

Fractions define probabilities; randrange uses integer rejection internally. Thus
there is no float acceptance threshold or CDF-rounding approximation here. This
is a small-vocabulary oracle, not a practical GPU implementation. Models and
processors are pure functions of the supplied prefix, never mutable KV/grammar
objects. Nothing loads a checkpoint, imports torch, or connects to a service.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

Weight = int | float | Fraction
Prefix = tuple[int, ...]


def _fraction(value: Weight) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError("weights must be real int/float/Fraction values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite weight")
    result = Fraction(value)
    if result < 0:
        raise ValueError("negative weight")
    return result


def _token(token: int, vocab_size: int) -> None:
    if type(token) is not int or not 0 <= token < vocab_size:
        raise ValueError(f"global token ID {token!r} outside vocabulary {vocab_size}")


@dataclass(frozen=True, init=False)
class Categorical:
    """Normalized exact distribution; unlisted global IDs have exactly zero mass.

    Sparse entries must be unique, including entries with zero weight. A full q
    is constructed with dense(); sparse_topk() chooses support BEFORE sampling.
    """

    vocab_size: int
    ids: tuple[int, ...]
    probabilities: tuple[Fraction, ...]
    tickets: tuple[int, ...]

    def __init__(self, vocab_size: int, ids: Sequence[int], weights: Sequence[Weight]):
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocabulary size must be a positive integer")
        if len(ids) != len(weights) or not ids:
            raise ValueError("support and weights must have matching nonempty lengths")
        for token in ids:
            _token(token, vocab_size)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate global token ID")
        entries = sorted(zip(ids, map(_fraction, weights)))
        mass = sum((weight for _, weight in entries), Fraction())
        if mass == 0:
            raise ValueError("empty effective support (all zero or all masked)")
        probs = tuple(weight / mass for _, weight in entries)
        denominator = math.lcm(*(p.denominator for p in probs))
        tickets = tuple(p.numerator * (denominator // p.denominator) for p in probs)
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "ids", tuple(token for token, _ in entries))
        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "tickets", tickets)

    @classmethod
    def dense(cls, weights: Sequence[Weight]) -> Categorical:
        return cls(len(weights), tuple(range(len(weights))), weights)

    def prob(self, token: int) -> Fraction:
        _token(token, self.vocab_size)
        # Intentionally simple and exact; production sparse lookup needs a kernel.
        return next((p for i, p in zip(self.ids, self.probabilities) if i == token), Fraction())

    def sample(self, rng: random.Random) -> int:
        draw = rng.randrange(sum(self.tickets))
        for token, count in zip(self.ids, self.tickets):
            if draw < count:
                return token
            draw -= count
        raise AssertionError("categorical integer sampler exhausted its support")

    def masked(self, allowed: Sequence[bool]) -> Categorical:
        if len(allowed) != self.vocab_size or any(type(x) is not bool for x in allowed):
            raise ValueError("mask must have one boolean per target vocabulary token")
        return Categorical(self.vocab_size, self.ids,
                           tuple(p if allowed[i] else 0 for i, p in zip(self.ids, self.probabilities)))

    def sparse_topk(self, k: int) -> Categorical:
        if type(k) is not int or not 1 <= k <= self.vocab_size:
            raise ValueError("top-k outside vocabulary")
        # Global lowest ID breaks ties, independent of TP ownership/order.
        selected = sorted(zip(self.ids, self.probabilities), key=lambda item: (-item[1], item[0]))[:k]
        return Categorical(self.vocab_size, tuple(i for i, _ in selected), tuple(p for _, p in selected))


def fp32_weights(vocab_size: int, ids: Sequence[int], weights: Sequence[Weight]) -> Categorical:
    """Quantize BEFORE sampling, then normalize the exact FP32 values identically.

    This is the semantics oracle for transported FP32 weights. CPU normalization
    here is rational, not an assertion of bit-parity with a future GPU softmax.
    """
    rounded: list[float] = []
    for weight in weights:
        _fraction(weight)
        try:
            rounded.append(struct.unpack("<f", struct.pack("<f", float(weight)))[0])
        except (OverflowError, struct.error) as error:
            raise ValueError("weight cannot be represented as finite FP32") from error
    return Categorical(vocab_size, ids, rounded)


def acceptance(p: Categorical, q: Categorical, token: int) -> Fraction:
    if p.vocab_size != q.vocab_size:
        raise ValueError("target/proposal vocabulary mismatch")
    qx = q.prob(token)
    if qx == 0:
        raise ValueError("proposal token has zero q mass")
    return min(Fraction(1), p.prob(token) / qx)


def residual(p: Categorical, q: Categorical) -> Categorical:
    """(p-q)+ over the FULL target vocabulary, never just transmitted support."""
    if p.vocab_size != q.vocab_size:
        raise ValueError("target/proposal vocabulary mismatch")
    return Categorical.dense(tuple(max(p.prob(i) - q.prob(i), Fraction())
                                   for i in range(p.vocab_size)))


@dataclass(frozen=True)
class Owner:
    request_id: str
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id or type(self.generation) is not int or self.generation < 0:
            raise ValueError("invalid request/generation owner")


class RandomStreams:
    """Request/generation/round-owned, domain-separated reproducible CPU streams.

    Creating streams consumes no process-global RNG. A verifier never uses the
    proposal generator as its acceptance or correction generator. Owner binding
    and block freshness are checked before draws. Not a GPU Philox replacement.
    """

    def __init__(self, seed: int, owner: Owner, round_id: int):
        if type(round_id) is not int or round_id < 0:
            raise ValueError("round ID must be nonnegative")
        self.owner = owner
        self.round_id = round_id
        self.proposal = self._make(seed, "proposal")
        self.acceptance = self._make(seed, "acceptance")
        self.correction = self._make(seed, "correction-or-bonus")

    def _make(self, seed: int, domain: str) -> random.Random:
        # Length-aware repr of a typed tuple prevents ambiguous field concatenation.
        key = repr((seed, self.owner.request_id, self.owner.generation, self.round_id, domain)).encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


Model = Callable[[Prefix], Categorical]
Processor = Callable[[Prefix, Categorical], Categorical]


def identity_processor(prefix: Prefix, distribution: Categorical) -> Categorical:
    return distribution


@dataclass(frozen=True)
class Proposal:
    token: int
    q: Categorical


@dataclass(frozen=True)
class Block:
    owner: Owner
    round_id: int
    prefix: Prefix
    context_id: str
    fixed_depth: int
    proposals: tuple[Proposal, ...]
    stop_ids: frozenset[int]


def propose(model: Model, prefix: Prefix, streams: RandomStreams, *, depth: int,
            context_id: str, stop_ids: frozenset[int] = frozenset()) -> Block:
    """Choose depth before any draws; q_i may depend on earlier sampled tokens."""
    if type(depth) is not int or not 1 <= depth <= 5:
        raise ValueError("reference fixed depth must be in 1..5")
    if not context_id:
        raise ValueError("processor/model context identity is required")
    proposals: list[Proposal] = []
    current = prefix
    for _ in range(depth):
        q = model(current)
        for token in (*current, *stop_ids):
            _token(token, q.vocab_size)
        token = q.sample(streams.proposal)
        proposals.append(Proposal(token, q))
        current += (token,)
        if token in stop_ids:
            break
    return Block(streams.owner, streams.round_id, prefix, context_id, depth, tuple(proposals), stop_ids)


@dataclass(frozen=True)
class Result:
    tokens: tuple[int, ...]
    accepted_drafts: int
    rejected: bool
    stopped: bool


class Verifier:
    """Single request session: commit only accepted prefix + correction/bonus.

    Processor callbacks must be pure functions of the *supplied* prefix. A real
    engine integration must clone/rollback grammar, penalty, KV and Markov state.
    Invalid state raises; no missing-q or target-only fallback is hidden here.
    """

    def __init__(self, owner: Owner, prefix: Prefix, context_id: str, target: Model, *,
                 processor: Processor = identity_processor, stop_ids: frozenset[int] = frozenset()):
        self.owner = owner
        self.prefix = prefix
        self.context_id = context_id
        self.target = target
        self.processor = processor
        self.stop_ids = stop_ids
        self.last_round = -1
        self.stopped = False
        self.failed = False

    def verify(self, block: Block, streams: RandomStreams) -> Result:
        if self.stopped:
            raise ValueError("request has already stopped")
        if self.failed:
            raise ValueError("failed verification requires explicit session reset")
        if block.owner != self.owner or streams.owner != self.owner:
            raise ValueError("request/generation RNG or block ownership mismatch")
        if block.round_id != streams.round_id or block.round_id <= self.last_round:
            raise ValueError("stale/reused round or wrong RNG round")
        if block.prefix != self.prefix or block.context_id != self.context_id:
            raise ValueError("stale prefix or changed model/processor context")
        if block.stop_ids != self.stop_ids:
            raise ValueError("stop policy mismatch")
        if type(block.fixed_depth) is not int or not 1 <= block.fixed_depth <= 5 or not 1 <= len(block.proposals) <= block.fixed_depth:
            raise ValueError("invalid fixed-depth block")
        if len(block.proposals) != block.fixed_depth and block.proposals[-1].token not in self.stop_ids:
            raise ValueError("depth may shorten only on a terminal token")
        vocab = block.proposals[0].q.vocab_size
        for proposal in block.proposals:
            if proposal.q.vocab_size != vocab or proposal.q.prob(proposal.token) == 0:
                raise ValueError("invalid proposal distribution/token")
        for token in (*self.prefix, *self.stop_ids):
            _token(token, vocab)

        # Poison this round before consuming RNG: a failed callback must not allow
        # a retry with partially advanced randomness. Recovery is explicit reset.
        self.last_round = block.round_id
        self.failed = True
        current = self.prefix
        output: list[int] = []
        accepted = 0
        rejected = False
        for proposal in block.proposals:
            p = self.processor(current, self.target(current))
            a = acceptance(p, proposal.q, proposal.token)
            if streams.acceptance.randrange(a.denominator) < a.numerator:
                token = proposal.token
                accepted += 1
            else:
                token = residual(p, proposal.q).sample(streams.correction)
                rejected = True
            output.append(token)
            current += (token,)
            if rejected or token in self.stop_ids:
                break
        if not rejected and output[-1] not in self.stop_ids:
            p = self.processor(current, self.target(current))
            if p.vocab_size != vocab:
                raise ValueError("bonus target vocabulary mismatch")
            token = p.sample(streams.correction)
            output.append(token)
            current += (token,)
        self.prefix = current
        self.stopped = output[-1] in self.stop_ids
        self.failed = False
        return Result(tuple(output), accepted, rejected, self.stopped)
