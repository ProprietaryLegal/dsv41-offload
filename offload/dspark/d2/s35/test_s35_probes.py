"""S35 independent probes of offload/d2-integrate e774e91 (amendment 7 G1(b)). CPU only, synthetic data only.

No hosts, sockets, GPUs, W12 files or real receipts. The engine probes read the pinned image source
(/opt/dsv41/data/src-image, read only) and execute two small pure functions from it; they skip when that
reference copy is absent.

FIX-1 probes retain the bias evidence and check amendment 7b: the live counter amendment 7 binds to is not the quantity the K2 proxy
measures. (a) The production depth policy ``conf`` can verify depth 0; the scheduler still counts such a round in
``num_drafts`` with 0 accepted, so live position-0 acceptance is censored. (b) The proxy's position-0 rate depends on
the round boundaries of its own window-5 committed trajectory, so it differs from a live trajectory with smaller
windows on the same drafts and tokens.
FIX-2 probe: the pinned generator writes no rate row while paused; the live reference refuses a final rate row more
than 120 s older than the final snapshot.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import textwrap

import pytest
import torch

from offload.dspark.d2 import g1
from offload.dspark.d2.live_reference import RESTART, build_reference, timestamp
from offload.dspark.d2.train.tests.test_g1_live import metrics

REPO = Path(__file__).resolve().parents[4]
IMAGE = Path('/opt/dsv41/data/src-image/vllm')
PINNED_GENERATOR_SHA256 = '56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a'  # W12 w12.log GEN line
# Production remote-round costs (sparkhead main 21bed31 dsv41.env OFFLOAD_DSPARK_ENV: REMOTE_ROUND_MS, REMOTE_DRAFT_MS)
ROUND_MS = [28.79, 31.86, 34.44, 37.76, 41.56, 43.68]
DRAFT_MS = 4.4

needs_image = pytest.mark.skipif(not IMAGE.is_dir(), reason='pinned image reference copy absent')


def image_function(relative: str, name: str, cls: str | None = None) -> str:
    tree = ast.parse((IMAGE/relative).read_text())
    nodes = tree.body
    if cls is not None:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment((IMAGE/relative).read_text(), node) or '')


@needs_image
def test_fix1a_conf_policy_verifies_depth_zero_and_engine_counts_it_as_a_draft() -> None:
    ns: dict[str, object] = {'torch': torch}
    exec(image_function('v1/worker/gpu/spec_decode/depth_policy.py', 'choose_depth'), ns)
    choose_depth = ns['choose_depth']
    cost = torch.tensor([r + DRAFT_MS for r in ROUND_MS])
    conf = torch.tensor([[0.08, 0.5, 0.5, 0.5, 0.5], [0.10, 0.5, 0.5, 0.5, 0.5], [0.7, 0.6, 0.5, 0.4, 0.3]])
    depth = choose_depth(conf, cost).tolist()  # type: ignore[operator]
    assert depth[0] == 0 and depth[1] >= 1 and depth[2] >= 1
    # dead rows = local position > depth; row p carries draft p-1, so depth 0 invalidates draft 0
    dead = image_function('v1/worker/gpu/spec_decode/depth_policy.py', 'dead_rows', 'DSparkDepthPolicy')
    assert 'expanded_local_pos > self.depth[' in dead
    # the scheduler counts the round from the SCHEDULED window (>= AK_MIN 2), not from the verify depth
    sched = (IMAGE/'v1/core/sched/scheduler.py').read_text()
    assert 'num_draft_tokens = len(scheduled_spec_token_ids)' in sched
    assert 'depth' not in image_function('v1/core/sched/scheduler.py', 'make_spec_decoding_stats', 'Scheduler')
    ns2: dict[str, object] = {}
    src = image_function('v1/spec_decode/metrics.py', 'observe_draft', 'SpecDecodingStats')
    exec('class S:\n    def __init__(self):\n        self.num_spec_tokens = 5; self.num_drafts = 0; self.num_draft_tokens = 0\n'
         '        self.num_accepted_tokens = 0; self.num_accepted_tokens_per_pos = [0]*5; self.num_draft_tokens_per_pos = [0]*5\n'
         + textwrap.indent(src, '    '), ns2)
    s = ns2['S']()  # type: ignore[operator]
    s.observe_draft(num_draft_tokens=4, num_accepted_tokens=0)   # depth-0 round: window 4, nothing verifiable
    s.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
    assert s.num_drafts == 2 and s.num_accepted_tokens_per_pos[0] == 1
    ref = {'position0_acceptance': .5, 'status': 'VALID', 'sha256': 'probe'}
    assert g1.g1_verdict(.8, 'dev', live_reference=ref)['tripwire_passed'] is True  # type: ignore[arg-type]
    assert g1.g1_verdict(.2499, 'dev', live_reference=ref)['pass'] is False  # type: ignore[arg-type]
    # amendment 7's premise ("every round drafts at least one token, so position 0 is not censored") fails here:
    # live position-0 = P(depth >= 1 and draft 0 accepted); the proxy scores draft 0 on every round.


def sequence(pattern: str) -> tuple[dict[str, g1.Sequence], dict[str, dict[int, list[int]]]]:
    """Tokens 0..L-1 (token t at position t); drafts at every position. 'E' positions: all 5 drafts right;
    'H' positions: draft 0 wrong."""
    g = 1
    L = g + len(pattern) + 6
    tokens = list(range(L))
    drafts: dict[int, list[int]] = {}
    for P in range(g - 1, L - 2):
        easy = pattern[P - g + 1] == 'E' if 0 <= P - g + 1 < len(pattern) else False
        drafts[P] = [P + 2 + j for j in range(5)] if easy else [-1] * 5
    return {'s': {'tokens': tokens, 'gen_start': g, 'L': L, 'family': 'f'}}, {'s': drafts}


def live_like_p0(seq: g1.Sequence, drafts: dict[int, list[int]], window: int) -> float:
    tok, g, L = seq['tokens'], seq['gen_start'], seq['L']
    P, rounds, hits = g - 1, 0, 0
    while P <= L - 3 - window:
        ml = g1.match_len(drafts[P][:window], [tok[t] for t in range(P + 2, P + 2 + window)])
        assert ml is not None
        rounds += 1
        hits += ml > 0
        P += ml + 1
    return hits / rounds


def test_fix1b_proxy_position0_depends_on_its_own_trajectory_window() -> None:
    seqs, drafts = sequence(('E' * 30 + 'H' * 6) * 20)
    ref: dict[str, object] = {'position0_acceptance': 0.0, 'accepted_per_round': 0.0,
                              'per_position_acceptance': [0.0]*5, 'foreign_share': 0.,
                              'status': 'VALID', 'sha256': 'probe'}
    proxy = g1.score(seqs, drafts, 'dev', live_reference=ref)['position0_match_rate']  # type: ignore[arg-type]
    live4 = live_like_p0(seqs['s'], drafts['s'], 4)
    live2 = live_like_p0(seqs['s'], drafts['s'], 2)
    # same drafter, same tokens, same per-position draft quality: the statistic moves with the window by >> 0.03
    assert live4 - proxy > 0.03 and live2 - live4 > 0.03
    ref['position0_acceptance'] = live2
    report = g1.score(seqs, drafts, 'dev', live_reference=ref)  # type: ignore[arg-type]
    assert report['g1']['pass'] is True and report['g1']['tripwire_passed'] is True
    assert 'tolerance' not in report['g1']
    assert set(report['live_comparison']['biases']) == {'depth_0_censoring', 'window', 'split_coverage', 'foreign_share'}
    assert report['per_position_match_rate'][0] == proxy
    assert report['live_comparison']['position0_acceptance'] == live2


def test_fix2_generator_writes_no_rate_row_while_paused() -> None:
    path = REPO/'offload/dspark/d2/generate.py'
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_GENERATOR_SHA256
    tree = ast.parse(path.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'loop')
    paused = next(n for n in ast.walk(loop) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                  and n.test.id == 'why')
    assert isinstance(paused.body[-1], ast.Continue)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'rate' for b in paused.body for n in ast.walk(b))
    # the only rate row after a PAUSE is the forced one at driver exit (decode_close.sh pause sends C-c)
    tail = [ast.unparse(s) for s in loop.body if not isinstance(s, ast.While)]
    assert 'self.rate(force=True)' in tail


def test_fix2_final_snapshot_taken_after_flush_refuses(tmp_path: Path) -> None:
    first, final = tmp_path/'metrics-20261025T043300Z.prom', tmp_path/'metrics-20261025T061500Z.prom'
    first.write_text(metrics(80, 200))
    final.write_text(metrics())
    # generator exited at 06:00:00 (forced row); coordinator snapshots after flush/copy, before rollback
    rate = tmp_path/'rate-20261025T061500Z.json'
    rate.write_text(json.dumps({'t': '2026-10-25T06:00:00Z', 'gen_tokens': 248}))
    with pytest.raises(ValueError, match='stale'):
        build_reference([first, final], [rate], '2026-10-25T06:00:00Z', '2026-10-25T07:00:00Z')
    assert math.isclose(timestamp(RESTART), timestamp('2026-10-25T02:55:00Z'))


def test_attest_real_map_copied_as_resolution_refuses(tmp_path: Path) -> None:
    import hashlib
    import re
    from d2 import pass2_attest as A
    m = REPO/'offload/dspark/d2/PASS2-ATTEST-EVIDENCE.md'
    claims = A.parse(re.findall(r'^```json\n(.*?)\n```$', m.read_text(), re.M | re.S)[0])['claims']
    assert sum(c['status'] != 'MEASURED' for c in claims.values()) == 28  # 24 IMPLIED + 4 MISSING at e774e91
    res = tmp_path/'resolution.json'
    res.write_text(json.dumps({'schema': 'd2.pass2-resolution.v1', 'approved': True, 'coordinator': A.COORDINATOR,
                               'run': A.RUN, 'claims': claims}))
    d = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    with pytest.raises(ValueError):
        A.generate(m, d(m), res, d(res), tmp_path/'absent-manifest.json', tmp_path/'draft')
    assert not (tmp_path/'draft').exists()
    for name, claim in claims.items():  # no IMPLIED/MISSING claim proves even with a review scope added
        if claim['status'] != 'MEASURED':
            with pytest.raises(ValueError):
                A.prove(name, dict(claim, scope='s35'))


def test_obs_attest_claim_may_bind_to_the_resolution_itself(tmp_path: Path) -> None:
    """Observation: the pointer source may be the resolution file (or any coordinator-written file); binding proves
    equality, not evidence. PASS2-ATTEST.md states the coordinator's approval is the trust boundary."""
    from d2 import pass2_attest as A
    import hashlib
    res, other = tmp_path/'resolution.json', tmp_path/'any.txt'
    res.write_text(json.dumps({'claims': {'sampling.top_k': {'value': -1}}}))
    other.write_text('x')
    d = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    claim = {'value': -1, 'status': 'MEASURED', 'scope': 'self', 'sources': [
        {'path': str(res), 'sha256': d(res), 'pointer': '/claims/sampling.top_k/value'},
        {'path': str(other), 'sha256': d(other)}]}
    assert A.prove('sampling.top_k', claim) == -1
