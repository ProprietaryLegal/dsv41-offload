"""Generation driver against a fake vLLM: idle gate, abort on another request, partial kept and resumed exactly,
pause file, restart of a killed driver."""

import argparse
import json
import os
import threading
import time

from d2 import generate as G
from d2.tests.fakevllm import Fake, uninterrupted


def _args(tmp, url, **kw):
    a = argparse.Namespace(prompts="", state=str(tmp / "gen"), server=url, model="m", streams=2, max_tokens=60,
                           target_tokens=10 ** 9, spark_hours=0.0, i_mean_uncapped=True, splits=",".join(G.ORDER), seed=1, idle_s=0.15, backoff_s=0.3,
                           grace_s=0.05, poll_s=0.01, stagger_s=0.0, checkpoint_s=0.05, rate_every_s=0.1,
                           max_errors_in_row=3, pause_file=str(tmp / "PAUSE"), hold_host="", hold_path="",
                           hold_every_s=1.0, hold_stale_s=10.0, no_local_check=True)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _prompts(n):
    return [{"pid": f"p{i}", "split": "train" if i else "heldout_legal", "family": "f", "prompt_sha256": f"h{i}",
             "messages": [{"role": "user", "content": f"prompt {i}"}], "chat_template_kwargs": dict(G.CTK)}
            for i in range(n)]


def _check_exact(d):
    for pid, s in d.st.items():
        assert s["status"] == "done", (pid, s["status"])
        assert s["gen_ids"] == uninterrupted(s["prompt_ids"], s["max_tokens"]), pid
        assert s["tokens_sha256"] == G.tokens_sha256(s["prompt_ids"] + s["gen_ids"])
        assert s["L"] == len(s["prompt_ids"]) + len(s["gen_ids"]) and s["gen_start"] == len(s["prompt_ids"])


def test_runs_to_completion_with_exact_ids(tmp_path):
    with Fake() as f:
        a = _args(tmp_path, f.url)
        d = G.Driver(a, _prompts(5))
        d.loop()
        _check_exact(d)
        assert max(r["n_prompt"] for r in f.requests) > 0
        assert all(r["priority"] == 20 and r["temperature"] == 1.0 and r["top_p"] == 1.0 and r["return_token_ids"]
                   for r in f.requests)
        assert d.obs["abort_events"] == 0


def test_other_request_aborts_and_partials_resume_exactly(tmp_path):
    with Fake(token_delay=0.01) as f:
        a = _args(tmp_path, f.url, max_tokens=70)
        d = G.Driver(a, _prompts(4))
        fired = {"done": False}

        def hook(drv):
            if not fired["done"] and drv.gen_total() >= 30:
                fired["done"] = True
                f.other = 1                              # an interactive request appears

                def clear():
                    time.sleep(0.2)
                    f.other = 0
                threading.Thread(target=clear, daemon=True).start()
        d.loop(hook=hook)
        _check_exact(d)
        assert d.obs["abort_events"] >= 1 and d.obs["own_aborts"] >= 1
        segs = [g for s in d.st.values() for g in s["segments"]]
        assert any(g["finish"] == "aborted" for g in segs)
        resumed = [s for s in d.st.values() if len(s["segments"]) > 1]
        assert resumed and all(s["segments"][1]["start"] == s["segments"][0]["n"] for s in resumed)
        ev = [json.loads(x) for x in open(os.path.join(a.state, "events.jsonl"))]
        assert any(e["event"] == "abort_all" for e in ev)


def test_no_submit_while_busy_and_pause_file(tmp_path):
    with Fake() as f:
        a = _args(tmp_path, f.url)
        f.other = 1
        d = G.Driver(a, _prompts(2))
        t = threading.Thread(target=d.loop, daemon=True)
        t.start()
        time.sleep(0.5)
        assert not f.requests                            # never submitted while another request runs
        open(a.pause_file, "w").close()
        f.other = 0
        time.sleep(0.5)
        assert not f.requests                            # paused
        os.remove(a.pause_file)
        t.join(timeout=30)
        assert not t.is_alive()
        _check_exact(d)


def test_restart_resumes_a_killed_driver(tmp_path):
    with Fake() as f:
        a = _args(tmp_path, f.url)
        d = G.Driver(a, _prompts(1))
        d.ensure_prompt_ids("p0")
        full = uninterrupted(d.st["p0"]["prompt_ids"], a.max_tokens)
        s = d.st["p0"]
        s["gen_ids"] = full[:7]
        s["status"] = "running"                         # what a killed driver leaves behind
        d.save("p0")
        d.close()                                       # a killed process releases the state-dir lock
        d2 = G.Driver(a, _prompts(1))
        assert d2.st["p0"]["status"] == "partial"
        assert json.load(open(d2.path("p0")))["segments"] == [
            {"seg": 0, "start": 0, "n": 7, "unrecorded": True}]
        d2.loop()
        _check_exact(d2)
        assert d2.st["p0"]["segments"][0] == {"seg": 0, "start": 0, "n": 7, "unrecorded": True}
        assert d2.st["p0"]["segments"][1]["start"] == 7
        assert G.summary(a.state)["segments"] == 2


def test_target_tokens_stops_new_prompts(tmp_path):
    with Fake() as f:
        a = _args(tmp_path, f.url, target_tokens=50, streams=1)
        d = G.Driver(a, _prompts(6))
        d.loop()
        done = [s for s in d.st.values() if s["status"] == "done"]
        assert 1 <= len(done) < 6
        assert sum(len(s["gen_ids"]) for s in d.st.values()) >= 50
        assert all(s["status"] in ("done", "pending") for s in d.st.values())


def test_spark_hours_budget_stops_submissions(tmp_path):
    with Fake(token_delay=0.01) as f:
        a = _args(tmp_path, f.url, streams=1, spark_hours=0.4 / 3600)      # 0.4 s of own-stream time
        d = G.Driver(a, _prompts(6))
        d.loop()
        done = [s for s in d.st.values() if s["status"] == "done"]
        assert len(done) < 6 and all(s["status"] in ("done", "pending", "partial") for s in d.st.values())
        assert json.load(open(os.path.join(a.state, "budget.json")))["busy_s"] >= 0.4
        d2 = G.Driver(a, _prompts(6))                                   # the budget survives a restart
        assert not d2.budget_left()
