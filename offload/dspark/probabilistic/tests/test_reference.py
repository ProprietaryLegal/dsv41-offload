"""Synthetic distributions only. No model tensors, prompts or generated text."""

import itertools
import json
import math
import random
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F

import pytest

from offload.dspark.probabilistic.reference import (
    Categorical as D,
)
from offload.dspark.probabilistic.reference import (
    Owner,
    Proposal,
    RandomStreams,
    Verifier,
    acceptance,
    fp32_weights,
    propose,
    residual,
)


def law(p: D, q: D, *, actually_sampled: D | None = None) -> tuple[F, ...]:
    """Enumerate acceptance and correction outcomes, without Monte Carlo."""
    actual = q if actually_sampled is None else actually_sampled
    accepted = tuple(actual.prob(i) * acceptance(p, q, i) if actual.prob(i) else F()
                     for i in range(p.vocab_size))
    reject = 1 - sum(accepted)
    if not reject:
        return accepted
    r = residual(p, q)
    return tuple(a + reject * r.prob(i) for i, a in enumerate(accepted))


def test_exact_distribution_identity_exhaustive():
    # All 3-token distributions on the quarter grid, including zeros/disjoint support.
    grid = [D.dense(x) for x in itertools.product(range(5), repeat=3) if sum(x) == 4]
    for p, q in itertools.product(grid, repeat=2):
        assert law(p, q) == p.probabilities
        assert sum(q.prob(i) * acceptance(p, q, i) for i in q.ids if q.prob(i)) == sum(
            min(p.prob(i), q.prob(i)) for i in range(3))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_exact_sparse_identity(k):
    p = D.dense([1, 2, 3, 4])
    q = D.dense([7, 1, 2, 3]).sparse_topk(k)
    assert law(p, q) == p.probabilities
    assert sum(q.probabilities) == 1
    assert all(q.prob(i) == 0 for i in range(4) if i not in q.ids)


def test_residual_covers_full_target_vocabulary():
    p = D.dense([1, 2, 3, 4])
    q = D(4, [0], [1])
    assert residual(p, q).probabilities == (0, F(2, 9), F(1, 3), F(4, 9))
    assert law(p, q) == p.probabilities


def test_dropping_tail_after_sampling_is_not_exact():
    p, original = D.dense([1, 3, 6]), D.dense([7, 2, 1])
    # Even dropping only the smallest entry is wrong if sampling used original q.
    transmitted = original.sparse_topk(2)
    # The lost token cannot even be verified: fail instead of silently accepting.
    with pytest.raises(ValueError, match="zero q"):
        acceptance(p, transmitted, 2)
    # A different full q also breaks the law, even when every token is present.
    assert law(p, D.dense([8, 1, 1]), actually_sampled=original) != p.probabilities


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [math.nan, 1], [math.inf, 1],
                                    [-math.inf, 1], [True, 1], ["1", 2]])
def test_bad_weights(weights):
    with pytest.raises((ValueError, TypeError)):
        D.dense(weights)


@pytest.mark.parametrize("ids,weights", [([1, 1], [1, 0]), ([-1], [1]), ([3], [1]),
                                       ([1.0], [1]), ([True], [1]), ([1], [1, 2]), ([], [])])
def test_bad_sparse_support(ids, weights):
    with pytest.raises(ValueError):
        D(3, ids, weights)


def test_masks_and_nonfinite_wire_values():
    p = D.dense([1, 2, 3])
    assert p.masked([True, False, True]).probabilities == (F(1, 4), 0, F(3, 4))
    for mask in ([False, False, False], [True], [1, 0, 1]):
        with pytest.raises(ValueError):
            p.masked(mask)
    for weights in ([1e40, 1], [1e-50, 0], [math.nan, 1], [math.inf, 1]):
        with pytest.raises(ValueError):
            fp32_weights(2, [0, 1], weights)


def test_fp32_is_defined_before_sampling_and_normalized_on_both_sides():
    import struct
    weights = [0.10000000001, 0.2, 0.7]
    packed = struct.pack("<3f", *weights)
    sender = fp32_weights(4, [0, 2, 3], weights)
    receiver = fp32_weights(4, [0, 2, 3], struct.unpack("<3f", packed))
    assert sender == receiver
    assert sender != D(4, [0, 2, 3], weights)
    assert sender.prob(1) == 0
    assert sum(sender.probabilities) == 1
    assert law(D.dense([1, 2, 3, 4]), sender) == D.dense([1, 2, 3, 4]).probabilities


def test_sparse_order_ties_and_vocab_mismatch():
    assert D(4, [3, 1, 2, 0], [1, 1, 1, 1]).sparse_topk(2).ids == (0, 1)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        acceptance(D.dense([1, 2]), D.dense([1, 2, 3]), 0)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        residual(D.dense([1, 2]), D.dense([1, 2, 3]))
    with pytest.raises(ValueError, match="all zero"):
        residual(D.dense([1, 2]), D.dense([1, 2]))
    for k in (0, 5, True):
        with pytest.raises(ValueError):
            D.dense([1, 2, 3, 4]).sparse_topk(k)


DEFAULT_OWNER = Owner("request-a", 0)


def make_block(owner=DEFAULT_OWNER, round_id=0, prefix=(), depth=2,
               context_id="target-grammar-v1", stops=frozenset()):
    streams = RandomStreams(1234, owner, round_id)
    block = propose(lambda _: D.dense([5, 3, 2]), prefix, streams, depth=depth,
                    context_id=context_id, stop_ids=stops)
    return block, streams


def test_rng_ownership_and_global_rng_untouched():
    global_before = random.getstate()
    owner = Owner("request-a", 3)
    streams = RandomStreams(123, owner, 2)
    before = streams.acceptance.getstate(), streams.correction.getstate()
    propose(lambda _: D.dense([1, 2, 3]), (), streams, depth=5, context_id="v1")
    assert before == (streams.acceptance.getstate(), streams.correction.getstate())
    assert random.getstate() == global_before
    states = [RandomStreams(123, owner, 2).proposal.getstate(),
              RandomStreams(123, owner, 2).acceptance.getstate(),
              RandomStreams(123, owner, 2).correction.getstate(),
              RandomStreams(123, Owner("request-b", 3), 2).proposal.getstate(),
              RandomStreams(123, Owner("request-a", 4), 2).proposal.getstate(),
              RandomStreams(123, owner, 3).proposal.getstate()]
    assert len({repr(s) for s in states}) == len(states)


@pytest.mark.parametrize("bad", ["owner", "rng_owner", "generation", "round", "prefix",
                                "context", "stops", "short", "zero_q", "vocab"])
def test_invalid_state_refused_before_rng_draws(bad):
    block, streams = make_block()
    verifier = Verifier(block.owner, (), block.context_id, lambda _: D.dense([1, 1, 1]))
    if bad == "owner":
        block = replace(block, owner=Owner("other", 0))
    elif bad == "rng_owner":
        streams = RandomStreams(1234, Owner("other", 0), 0)
    elif bad == "generation":
        block = replace(block, owner=Owner(block.owner.request_id, 1))
    elif bad == "round":
        block = replace(block, round_id=1)
    elif bad == "prefix":
        block = replace(block, prefix=(1,))
    elif bad == "context":
        block = replace(block, context_id="other-grammar")
    elif bad == "stops":
        block = replace(block, stop_ids=frozenset({2}))
    elif bad == "short":
        block = replace(block, proposals=block.proposals[:1])
    elif bad == "zero_q":
        block = replace(block, proposals=(Proposal(0, D(3, [2], [1])),) * 2)
    elif bad == "vocab":
        block = replace(block, proposals=(block.proposals[0], Proposal(0, D.dense([1, 1]))))
    before = streams.acceptance.getstate(), streams.correction.getstate()
    with pytest.raises(ValueError):
        verifier.verify(block, streams)
    assert before == (streams.acceptance.getstate(), streams.correction.getstate())
    assert verifier.prefix == ()


def test_round_reuse_stale_prefix_and_interleaved_request_isolation():
    block, streams = make_block()
    verifier = Verifier(block.owner, (), block.context_id, lambda _: D.dense([1, 1, 1]))
    result = verifier.verify(block, streams)
    committed = verifier.prefix
    assert committed == result.tokens
    with pytest.raises(ValueError, match="stale/reused"):
        verifier.verify(block, streams)
    next_stream = RandomStreams(1234, block.owner, 1)
    with pytest.raises(ValueError, match="stale prefix"):
        verifier.verify(replace(block, round_id=1), next_stream)
    # An interleaved other request cannot advance this request's RNG or state.
    other, other_rng = make_block(owner=Owner("b", 0))
    other_result = Verifier(other.owner, (), other.context_id, lambda _: D.dense([1, 1, 1])).verify(other, other_rng)
    assert other_result.tokens
    assert verifier.prefix == committed
    replay, replay_rng = make_block()
    assert Verifier(replay.owner, (), replay.context_id, lambda _: D.dense([1, 1, 1])).verify(replay, replay_rng) == result


def test_rejection_discards_suffix_and_commits_correction_only():
    owner = Owner("reject", 0)
    streams = RandomStreams(0, owner, 0)
    block = propose(lambda _: D.dense([1, 0, 0]), (), streams, depth=5, context_id="v1")
    seen = []

    def target(prefix):
        seen.append(prefix)
        return D.dense([0, 0, 1])

    verifier = Verifier(owner, (), "v1", target)
    result = verifier.verify(block, streams)
    assert result.tokens == (2,)
    assert result.accepted_drafts == 0 and result.rejected
    assert verifier.prefix == (2,)
    assert seen == [()]  # no target work or state changes on a rejected suffix


def test_grammar_processors_follow_proposed_then_committed_prefix():
    owner = Owner("grammar", 0)
    streams = RandomStreams(0, owner, 0)
    # Draft can put mass on forbidden tokens; target grammar still wins exactly.
    block = propose(lambda _: D.dense([0, 1, 0]), (), streams, depth=3, context_id="alternating")
    seen = []

    def grammar(prefix, p):
        seen.append(prefix)
        return p.masked([len(prefix) % 2 == 0, len(prefix) % 2 == 1, False])

    verifier = Verifier(owner, (), "alternating", lambda _: D.dense([1, 1, 1]), processor=grammar)
    result = verifier.verify(block, streams)
    assert result.tokens == (0,) and result.rejected
    # Next block now conditions the grammar on committed token 0, not draft 1.
    streams2 = RandomStreams(0, owner, 1)
    block2 = propose(lambda _: D.dense([0, 1, 0]), verifier.prefix, streams2,
                     depth=1, context_id="alternating")
    result2 = verifier.verify(block2, streams2)
    assert result2.tokens == (1, 0) and not result2.rejected
    assert seen == [(), (0,), (0, 1)]


def test_all_masked_target_and_failed_round_cannot_retry():
    block, streams = make_block()
    verifier = Verifier(block.owner, (), block.context_id, lambda _: D.dense([1, 2, 3]),
                        processor=lambda prefix, p: p.masked([False] * 3))
    with pytest.raises(ValueError, match="all masked"):
        verifier.verify(block, streams)
    assert verifier.prefix == ()
    with pytest.raises(ValueError, match="explicit session reset"):
        verifier.verify(block, streams)


@pytest.mark.parametrize("stop_at", ["draft", "correction", "bonus"])
def test_stop_discards_later_tokens_and_prevents_further_rounds(stop_at):
    owner = Owner("stop", 0)
    streams = RandomStreams(0, owner, 0)
    stops = frozenset({2})
    calls = []

    def draft(prefix):
        calls.append(prefix)
        return D.dense([0, 0, 1]) if stop_at == "draft" else D.dense([1, 0, 0])

    block = propose(draft, (), streams, depth=5 if stop_at == "draft" else 1,
                    context_id="v1", stop_ids=stops)

    def target(prefix):
        return D.dense([1, 0, 0]) if stop_at == "bonus" and not prefix else D.dense([0, 0, 1])

    verifier = Verifier(owner, (), "v1", target, stop_ids=stops)
    result = verifier.verify(block, streams)
    assert result.stopped and result.tokens[-1] == 2
    if stop_at == "draft":
        assert calls == [()] and result.accepted_drafts == 1
    if stop_at == "bonus":
        assert result.tokens == (0, 2) and result.accepted_drafts == 1
    with pytest.raises(ValueError, match="already stopped"):
        verifier.verify(block, streams)


@pytest.mark.parametrize("depth", [0, 6, True])
def test_bad_depth(depth):
    with pytest.raises(ValueError, match="fixed depth"):
        propose(lambda _: D.dense([1, 2]), (), RandomStreams(0, Owner("x", 0), 0),
                depth=depth, context_id="v1")


@pytest.mark.parametrize("kind", ["full", "sparse2", "delta", "equal", "disjoint"])
def test_distribution_frequency(kind):
    p = D.dense([1, 2, 3, 4])
    q = D.dense([6, 1, 2, 1])
    if kind == "sparse2":
        q = q.sparse_topk(2)
    elif kind == "delta":
        q = q.sparse_topk(1)
    elif kind == "equal":
        q = p
    elif kind == "disjoint":
        p, q = D.dense([0, 2, 3, 5]), D.dense([1, 0, 0, 0])
    counts = Counter()
    accepted = 0
    n = 12000
    owner = Owner("frequency", 0)
    for sample in range(n):
        streams = RandomStreams(847, owner, sample)
        block = propose(lambda _: q, (), streams, depth=1, context_id="v1")
        result = Verifier(owner, (), "v1", lambda _: p).verify(block, streams)
        counts[result.tokens[0]] += 1
        accepted += result.accepted_drafts
    for token in range(4):
        expected = float(p.prob(token))
        assert abs(counts[token] / n - expected) <= 6 * math.sqrt(expected * (1 - expected) / n) + 1 / n
    expected_accept = float(sum(min(p.prob(i), q.prob(i)) for i in range(4)))
    assert abs(accepted / n - expected_accept) <= 6 * math.sqrt(expected_accept * (1 - expected_accept) / n) + 1 / n
    print(json.dumps({"evidence": "MEASURED synthetic CPU", "kind": kind, "samples": n,
                      "counts": [counts[i] for i in range(4)],
                      "p": [float(p.prob(i)) for i in range(4)],
                      "accepted": accepted, "expected_acceptance": expected_accept}, sort_keys=True))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("depth", [2, 5])
def test_conditional_two_token_joint_frequency(sparse, depth):
    # This detects using committed-trajectory probabilities after a differing draft.
    def target(prefix):
        return D.dense([3, 1]) if not prefix or prefix[-1] == 0 else D.dense([1, 4])

    def draft(prefix):
        q = D.dense([1, 4]) if not prefix or prefix[-1] == 0 else D.dense([4, 1])
        return q.sparse_topk(1) if sparse else q

    counts = Counter()
    n = 10000
    owner = Owner("joint", 0)
    for sample in range(n):
        session = Verifier(owner, (), "v1", target)
        round_id = 0
        while len(session.prefix) < 2:
            streams = RandomStreams(sample + 51, owner, round_id)
            block = propose(draft, session.prefix, streams, depth=depth, context_id="v1")
            session.verify(block, streams)
            round_id += 1
        counts[session.prefix[:2]] += 1
    for pair in itertools.product(range(2), repeat=2):
        expected = float(target(()).prob(pair[0]) * target(pair[:1]).prob(pair[1]))
        assert abs(counts[pair] / n - expected) < 6 * math.sqrt(expected * (1 - expected) / n) + 1 / n
    print(json.dumps({"evidence": "MEASURED synthetic CPU", "kind": "joint", "sparse": sparse,
                      "depth": depth, "samples": n,
                      "counts_00_01_10_11": [counts[pair] for pair in itertools.product(range(2), repeat=2)]}, sort_keys=True))


def test_appendix_a_selection_bias_regression():
    # Own toy distribution: q=p=uniform, but inspecting a future confidence
    # admits the first draw only if it is 0. Otherwise target-only resampling
    # gives first-token P(0)=3/4 rather than 1/2, despite an exact verifier.
    p = D.dense([1, 1])
    biased_zero = p.prob(0) + p.prob(1) * p.prob(0)
    assert biased_zero == F(3, 4) != p.prob(0)
    assert law(p, p) == p.probabilities
