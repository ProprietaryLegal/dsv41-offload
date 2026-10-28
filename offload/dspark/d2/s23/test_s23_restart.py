"""S23 probes of D2-FIX1 item 1 (driver) and item 2 (revision restart boundary) against the lane's fake vLLM.

Legacy = the generate.py actually running in production (e2b2186, loaded from git, unmodified), so the restart is
tested on a state directory written by the real old code, not a hand-made one.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from d2 import generate as G
from d2.sequence_audit import segment_coverage
from d2.tests.fakevllm import Fake, uninterrupted
from d2.tests.test_generate import _args, _prompts

ROOT = Path(__file__).resolve().parents[4]


def legacy_module(tmp_path: Path):
    repo = os.environ.get("S23_GIT_ROOT", str(ROOT))          # a git checkout holding e2b2186 (archived trees: set it)
    src = subprocess.run(["git", "-C", repo, "show", "e2b2186:offload/dspark/d2/generate.py"],
                         env={k: v for k, v in os.environ.items() if not k.startswith('GIT_')},
                         check=True, capture_output=True, text=True).stdout
    p = tmp_path / "legacy_generate_e2b2186.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("legacy_generate_e2b2186", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.MAX_STREAMS == 4 and "AUX_PRIORITY" not in vars(m)
    return m


def contiguous(s: dict) -> bool:
    pos = 0
    for g in s["segments"]:
        if g["start"] != pos:
            return False
        pos += g["n"]
    return pos == len(s["gen_ids"]) and [g["seg"] for g in s["segments"]] == list(range(len(s["segments"])))


def test_revision_restart_on_a_legacy_state_dir(tmp_path: Path) -> None:
    L = legacy_module(tmp_path)
    with Fake(token_delay=0.01) as f:
        a = _args(tmp_path, f.url, max_tokens=70, streams=2, spark_hours=1.0, i_mean_uncapped=False)
        d = L.Driver(a, _prompts(4))

        def stop_mid(drv):                                   # the coordinator's clean stop (TERM -> d.stop)
            if drv.gen_total() >= 30:
                drv.stop = True
        d.loop(hook=stop_mid)
        old = {pid: copy.deepcopy(s) for pid, s in d.st.items()}
        old_budget = json.load(open(os.path.join(a.state, "budget.json")))
        n_legacy_requests = len(f.requests)
        assert any(s["status"] == "partial" for s in old.values()), "need a partial at the boundary"
        assert all(r["priority"] == 5 for r in f.requests)
        # restart with the reviewed revision, same state, cap flag omitted (the documented restart command)
        b = _args(tmp_path, f.url, max_tokens=70, streams=3, spark_hours=None, i_mean_uncapped=False)
        d2 = G.Driver(b, _prompts(4))
        assert d2.cap_h == 1.0 and d2.busy_s == pytest.approx(old_budget["busy_s"], abs=0.05)
        d2.loop()
        new_requests = f.requests[n_legacy_requests:]
        assert new_requests and all(r["priority"] == 20 for r in new_requests)
        # identical sampling fields apart from priority (max_tokens = remaining room by design)
        keys = lambda r: {k: v for k, v in r.items() if k not in ("priority", "request_id", "max_tokens", "n_prompt")}
        assert {json.dumps(keys(r), sort_keys=True) for r in f.requests} == {json.dumps(keys(f.requests[0]), sort_keys=True)}
        for pid, s in d2.st.items():
            assert s["status"] == "done"
            assert s["gen_ids"] == uninterrupted(s["prompt_ids"], s["max_tokens"]), pid      # exact, no dup/loss
            assert contiguous(s), pid
            k = len(old[pid]["segments"])
            assert s["segments"][:k] == old[pid]["segments"], "legacy segments must stay byte-identical"
            assert all("priority" not in g and "generator_revision" not in g for g in s["segments"][:k])
            assert all(g["priority"] == 20 and g["generator_revision"] == "d2-decode-capture-option-ii-v2"
                       and g["sampling"]["temperature"] == 1.0 and g["sampling"]["top_p"] == 1.0
                       for g in s["segments"][k:])
        assert json.load(open(os.path.join(b.state, "budget.json")))["busy_s"] >= old_budget["busy_s"]


def test_masking_window_after_an_own_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production grace 3 s / poll 0.5 s: a foreign request arriving just as an own stream ends is seen only when the
    grace expires. One short sequence (1 s), the rest long (15 s): sparse completions, as in production."""
    from d2.tests import fakevllm
    lengths: dict = {}
    monkeypatch.setattr(fakevllm, "stop_len", lambda ids: lengths.setdefault(tuple(ids), 20 if not lengths else 300))
    with Fake(token_delay=0.05) as f:
        a = _args(tmp_path, f.url, max_tokens=400, streams=3, grace_s=3.0, poll_s=0.5, idle_s=0.0, backoff_s=60.0)
        d = G.Driver(a, _prompts(8))
        mark: dict = {}
        orig_abort = d.abort_all

        def abort_all(why: str) -> None:
            mark.setdefault("abort", (time.time(), why, d.obs["submits"]))
            orig_abort(why)
        d.abort_all = abort_all

        def hook(drv):
            if "t_other" not in mark and len(drv.ended) == 1 and drv.obs["submits"] >= 3:
                f.other = 1
                mark["t_other"] = time.time()
                mark["submits_at_other"] = drv.obs["submits"]
            if "abort" in mark:
                drv.stop = True
        d.loop(hook=hook)
        lag = mark["abort"][0] - mark["t_other"]
        print(f"S23 masking lag {lag:.2f} s; own submits while masked {mark['abort'][2] - mark['submits_at_other']}")
        assert "other request" in mark["abort"][1]
        assert lag <= 3.0 + 2 * 0.5 + 0.5


def test_masking_chains_over_back_to_back_completions(tmp_path: Path) -> None:
    """Chained completions must not extend the blind window via immediate replacement submissions."""
    with Fake(token_delay=0.02) as f:
        a = _args(tmp_path, f.url, max_tokens=400, streams=3, grace_s=3.0, poll_s=0.5, idle_s=0.0, backoff_s=60.0)
        d = G.Driver(a, _prompts(12))
        mark: dict = {}

        def hook(drv):
            if "t_other" not in mark and len(drv.ended) and drv.obs["submits"] >= 3:
                f.other = 1
                mark["t_other"], mark["s0"] = time.time(), drv.obs["submits"]
            if drv.positives and 'detected' not in mark:
                mark['detected'] = time.time()
                drv.stop = True
        d.loop(hook=hook)
        masked = (mark.get('detected') or time.time()) - mark["t_other"]
        print(f"S23 chained masking: foreign request unseen {masked:.2f} s, own submits meanwhile "
              f"{d.obs['submits'] - mark['s0']}, abort_events {d.obs['abort_events']}")
        assert 'detected' in mark
        assert masked <= 3.0 + 2 * 0.5 + 0.5
        assert d.obs['submits'] == mark['s0'], 'no replacement may perpetuate the grace'
        assert sum(s['status'] == 'done' for s in d.st.values()) >= 2, 'exercise chained completions'


def test_first_positive_poll_aborts(tmp_path: Path) -> None:
    with Fake(token_delay=0.15) as f:
        a = _args(tmp_path, f.url, max_tokens=400, streams=3, grace_s=3.0, poll_s=0.5, idle_s=0.0, backoff_s=60.0)
        d = G.Driver(a, _prompts(3))
        mark: dict = {}

        def hook(drv):
            if "t" not in mark and len(drv.streams) == 3 and drv.gen_total() > 20 and not drv.ended:
                f.other = 1
                mark["t"] = time.time()
            if drv.obs["abort_events"] and "lag" not in mark:
                mark["lag"] = time.time() - mark["t"]
                drv.stop = True
        d.loop(hook=hook)
        print(f"S23 first-poll abort lag {mark['lag']:.2f} s")
        assert mark["lag"] <= 0.5 + 0.5 + 0.3


def test_kill9_mid_stream_then_revision_restart(tmp_path: Path) -> None:
    """Process death between checkpoints: tokens up to the last checkpoint survive and stay exact; the dead segment's
    record is lost, so loading must mark its coverage unrecorded without inventing priority/sampling."""
    with Fake(token_delay=0.01) as f:
        prompts = _prompts(2)
        pf = tmp_path / "prompts.jsonl"
        pf.write_text("".join(json.dumps(p) + "\n" for p in prompts))
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('GIT_')},
                   PYTHONPATH=str(ROOT / "offload/dspark"), PYTHONDONTWRITEBYTECODE="1")
        # Exercise actual Driver checkpoints, not the production CLI's remote
        # capture-error monitor. The old CLI now exits before writing a budget
        # unless --capture-error-file is supplied; bypassing that safety gate
        # here also prevents accidental host access from this synthetic test.
        child = '''import json,sys
from pathlib import Path
from d2 import generate as G
from d2.tests.test_generate import _args
a = _args(Path(sys.argv[1]), sys.argv[2], max_tokens=400, streams=2,
          spark_hours=1.0, i_mean_uncapped=False, idle_s=0.0, poll_s=0.01, checkpoint_s=0.2)
prompts = [json.loads(line) for line in Path(sys.argv[3]).read_text().splitlines()]
G.Driver(a, prompts).loop()
'''
        cmd = [sys.executable, '-c', child, str(tmp_path), f.url, str(pf)]
        p = subprocess.Popen(cmd, env=env, cwd=str(ROOT / "offload/dspark"))
        seqs = tmp_path / "gen/seqs"
        t0 = time.time()
        checkpointed = False
        try:
            while time.time() - t0 < 20 and p.poll() is None:
                st = [json.load(open(x)) for x in seqs.glob("*.json")] if seqs.exists() else []
                if any(s["status"] == "running" and len(s["gen_ids"]) > 10 for s in st):
                    checkpointed = True
                    break
                time.sleep(0.05)
            assert checkpointed, 'child must actually write a running checkpoint before kill/restart'
        finally:
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
            p.wait()
        saved = {s["pid"]: s for s in (json.load(open(x)) for x in seqs.glob("*.json"))}
        a = _args(tmp_path, f.url, max_tokens=400, streams=2, spark_hours=None, i_mean_uncapped=False)
        d = G.Driver(a, prompts)
        d.loop()
        unrecorded = 0
        for pid, s in d.st.items():
            assert s["gen_ids"] == uninterrupted(s["prompt_ids"], s["max_tokens"]), pid
            assert contiguous(s)
            if saved.get(pid, {}).get('gen_ids'):
                unrecorded += 1
                assert s['segments'][0] == {'seg': 0, 'start': 0, 'n': len(saved[pid]['gen_ids']),
                                            'unrecorded': True}
                assert all(g['priority'] == 20 for g in s['segments'][1:])
                assert segment_coverage(s['segments'], len(s['gen_ids']))['unrecorded_tokens'] == len(
                    saved[pid]['gen_ids'])
        print(f"S23 kill-9: sequences with explicit unrecorded coverage {unrecorded}/{len(d.st)}")
        assert unrecorded >= 1
