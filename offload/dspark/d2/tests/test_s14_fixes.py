"""Lane S14's FIX probes (offload/s14-d2data 21fb29f, d2/s14/test_s14_generate.py, test_s14_replay.py,
test_s14_prompts.py), turned into passing tests of the fixes F1-F11 (S14.report.md section 2). The fake server is S14's
(d2/tests/s14serve.py)."""

from __future__ import annotations

import argparse
import json
import os
import stat
import threading
import time

import pytest

from d2 import generate as G
from d2.tests.s14serve import Serve, uninterrupted


def args(tmp, url, **kw):
    """S14's production gate settings (poll 0.5 s, grace 3 s, stagger 2 s), short idle/backoff."""
    a = argparse.Namespace(prompts="", state=str(tmp / "gen"), server=url, model="m", streams=3, max_tokens=60,
                           target_tokens=10 ** 9, spark_hours=0.0, i_mean_uncapped=True, splits=",".join(G.ORDER),
                           seed=1, idle_s=0.3, backoff_s=0.3, grace_s=3.0, poll_s=0.5, stagger_s=2.0,
                           checkpoint_s=30.0, rate_every_s=60.0, max_errors_in_row=5, pause_file=str(tmp / "PAUSE"),
                           hold_host="", hold_path="", hold_every_s=5.0, hold_stale_s=10.0, no_local_check=True)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def prompts(n):
    return [{"pid": f"p{i}", "split": "train" if i else "heldout_legal", "family": "f", "prompt_sha256": f"h{i}",
             "messages": [{"role": "user", "content": f"prompt {i}"}], "chat_template_kwargs": dict(G.CTK)}
            for i in range(n)]


def start(d):
    t = threading.Thread(target=d.loop, daemon=True)
    t.start()
    return t


def wait_for(cond, timeout=20.0, step=0.01):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time()
        time.sleep(step)
    raise AssertionError("condition not reached")


def own_rids(s):
    with s.lock:
        return [x for x in s.running if not x.startswith("FG")]


def stop(d, t):
    d.stop = True
    t.join(timeout=60)
    assert not t.is_alive()


# ---------------------------------------------------------------------------------------------------------------- F1
@pytest.mark.parametrize("mode", ["500", "hang"])
def test_F1_metrics_unavailable_aborts_own_streams(tmp_path, mode):
    with Serve(slots=4, token_delay=0.01) as s:
        d = G.Driver(args(tmp_path, s.url, streams=1, max_tokens=5000), prompts(1))
        t = start(d)
        wait_for(lambda: len(own_rids(s)) == 1)
        rid = own_rids(s)[0]
        t0 = time.time()
        s.metrics_mode = mode
        wait_for(lambda: rid in s.disconnects, timeout=8.0)
        lat = s.disconnects[rid] - t0
        n = len([r for r in s.requests if r["path"] == "/v1/completions"])
        time.sleep(2.0)                                   # still blind: nothing new is submitted
        assert len([r for r in s.requests if r["path"] == "/v1/completions"]) == n
        s.metrics_mode = "ok"
        stop(d, t)
        assert lat < 3.0
        ev = [json.loads(x) for x in open(os.path.join(d.state_dir, "events.jsonl"))]
        assert any(e["event"] == "abort_all" and e["why"] == "metrics unavailable" for e in ev)
        assert d.st["p0"]["status"] == "partial" and d.st["p0"]["gen_ids"]


# ---------------------------------------------------------------------------------------------------------------- F2
def test_F2_spark_hours_cap_survives_a_restart_without_the_flag(tmp_path):
    with Serve(slots=4, token_delay=0.01, stop_at=40) as s:
        a = args(tmp_path, s.url, streams=1, spark_hours=0.4 / 3600, i_mean_uncapped=False, poll_s=0.05,
                 grace_s=0.1, stagger_s=0.0)
        d = G.Driver(a, prompts(8))
        d.loop()
        done1 = sum(1 for v in d.st.values() if v["status"] == "done")
        assert json.load(open(os.path.join(a.state, "budget.json")))["spark_hours"] == 0.4 / 3600
        b = args(tmp_path, s.url, streams=1, spark_hours=None, i_mean_uncapped=False, poll_s=0.05, grace_s=0.1,
                 stagger_s=0.0)
        d2 = G.Driver(b, prompts(8))                     # no flag: the saved cap holds
        assert d2.cap_h == 0.4 / 3600 and not d2.budget_left()
        d2.loop()
        assert sum(1 for v in d2.st.values() if v["status"] == "done") == done1
        c = args(tmp_path, s.url, streams=1, spark_hours=0.0, i_mean_uncapped=False)
        with pytest.raises(SystemExit, match="uncapped"):
            G.Driver(c, prompts(8))                      # 0 without the confirmation is refused


def test_F2_first_run_needs_an_explicit_cap(tmp_path):
    with Serve(slots=4) as s:
        with pytest.raises(SystemExit, match="required on the first run"):
            G.Driver(args(tmp_path, s.url, spark_hours=None, i_mean_uncapped=False), prompts(1))
        d = G.Driver(args(tmp_path, s.url, spark_hours=0.0, i_mean_uncapped=True), prompts(1))
        assert d.cap_h == 0.0 and d.budget_left()


# ---------------------------------------------------------------------------------------------------------------- F3
def test_F3_spark_hours_cap_is_not_overrun_by_running_streams(tmp_path):
    with Serve(slots=4, token_delay=0.01) as s:
        a = args(tmp_path, s.url, streams=1, max_tokens=300, spark_hours=0.5 / 3600, i_mean_uncapped=False,
                 poll_s=0.05, grace_s=0.1, stagger_s=0.0)
        d = G.Driver(a, prompts(2))
        d.loop()
        assert d.busy_s <= 0.5 + 0.5, f"busy {d.busy_s:.2f} s against a 0.5 s cap"
        st = d.st["p0"]
        assert st["status"] == "partial" and st["gen_ids"] and st["segments"][-1]["finish"] == "aborted"
        assert st["gen_ids"] == uninterrupted(st["prompt_ids"], 300, 10 ** 9)[: len(st["gen_ids"])]
        ev = [json.loads(x) for x in open(os.path.join(a.state, "events.jsonl"))]
        assert any(e["event"] == "abort_all" and e["why"] == "spark-hours budget" for e in ev)


# ---------------------------------------------------------------------------------------------------------------- F4
def test_F4_second_driver_on_the_same_state_is_refused(tmp_path):
    with Serve(slots=8, token_delay=0.01) as s:
        a = args(tmp_path, s.url, streams=2, max_tokens=5000, stagger_s=0.0)
        d = G.Driver(a, prompts(4))
        t = start(d)
        wait_for(lambda: len(own_rids(s)) == 2)
        try:
            with pytest.raises(SystemExit, match="LOCK"):
                G.Driver(a, prompts(4))
        finally:
            stop(d, t)
        G.Driver(a, prompts(4)).close()                  # released when the first driver's loop ended


# ---------------------------------------------------------------------------------------------------------------- F5
def test_F5_streams_above_3_are_refused(tmp_path):
    with Serve(slots=8, token_delay=0.01) as s:
        for n in (4, 5, 6, 0):
            with pytest.raises(SystemExit, match="--streams"):
                G.Driver(args(tmp_path, s.url, streams=n), prompts(6))
        d = G.Driver(args(tmp_path, s.url, streams=3, max_tokens=50, stagger_s=0.0, poll_s=0.05, grace_s=0.1),
                     prompts(6))
        d.loop()
        assert s.max_running_own <= 3


def test_F5_cli_default_is_3_and_refuses_4():
    ap = argparse.ArgumentParser()
    G.add_common(ap)
    assert ap.parse_args(["--state", "x"]).streams == 3
    with pytest.raises(SystemExit):
        ap.parse_args(["--state", "x", "--streams", "4"])


# ---------------------------------------------------------------------------------------------------------------- F6
def _fake_ssh(tmp_path, body: str) -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    p = bindir / "ssh"
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(bindir)


def test_F6_hold_check_does_not_blind_the_gate(tmp_path, monkeypatch):
    """A 6 s ssh: the gate still sees the foreground request through /metrics within a poll or two."""
    mark = tmp_path / "ssh-started"
    bindir = _fake_ssh(tmp_path, f"touch {mark}; sleep 6; exit 1")
    monkeypatch.setenv("PATH", bindir + os.pathsep + os.environ["PATH"])
    with Serve(slots=4, token_delay=0.01) as s:
        d = G.Driver(args(tmp_path, s.url, streams=1, max_tokens=5000, hold_host="sparkhead", hold_path="~/x/HOLD",
                          hold_every_s=3.0, hold_stale_s=10.0), prompts(1))
        t = start(d)
        wait_for(lambda: len(own_rids(s)) == 1, timeout=30)
        rid = own_rids(s)[0]
        if mark.exists():
            mark.unlink()
        wait_for(lambda: mark.exists(), timeout=30)        # an ssh check is in flight now
        t_fg = time.time()
        s.foreign(30.0)
        wait_for(lambda: rid in s.disconnects, timeout=30)
        lat = s.disconnects[rid] - t_fg
        stop(d, t)
        assert lat < 2.0


@pytest.mark.parametrize("body,why", [("exit 0", "hold"), ("exit 255", "hold_unknown"), ("sleep 30", "hold_stale")])
def test_F6_hold_unknown_and_stale_pause(tmp_path, monkeypatch, body, why):
    bindir = _fake_ssh(tmp_path, body)
    monkeypatch.setenv("PATH", bindir + os.pathsep + os.environ["PATH"])
    with Serve(slots=4, token_delay=0.01) as s:
        d = G.Driver(args(tmp_path, s.url, streams=1, hold_host="sparkhead", hold_path="~/x/HOLD", hold_every_s=0.2,
                          hold_stale_s=1.0, hold_timeout_s=20.0, poll_s=0.1), prompts(1))
        t = start(d)
        time.sleep(2.0)
        stop(d, t)
        assert not [r for r in s.requests if r["path"] == "/v1/completions"]
        ev = [json.loads(x) for x in open(os.path.join(d.state_dir, "events.jsonl"))]
        assert any(e["event"] == "pause" and e["why"] == why for e in ev), [e for e in ev if e["event"] == "pause"]


# ---------------------------------------------------------------------------------------------------------------- F7
def test_F7_serve_restarts_are_pauses_not_errors(tmp_path):
    """S14's three-drop probe: the serve goes away three times while 2 own streams run. No error is counted, the
    driver keeps running, nothing is marked failed, every partial resumes exactly."""
    with Serve(slots=4, token_delay=0.01, stop_at=400) as s:
        d = G.Driver(args(tmp_path, s.url, streams=2, max_tokens=500, stagger_s=0.0, poll_s=0.1, grace_s=0.2,
                          idle_s=0.3, backoff_s=0.3), prompts(2))
        out = {}

        def run():
            try:
                d.loop()
                out["exit"] = "returned"
            except RuntimeError as e:
                out["exit"] = f"RuntimeError: {e}"
        t = threading.Thread(target=run, daemon=True)
        t.start()
        for _ in range(3):
            wait_for(lambda: len(own_rids(s)) == 2, timeout=30)
            time.sleep(0.3)
            s.die = True
            time.sleep(1.0)
            s.die = False
        t.join(timeout=120)
        assert out.get("exit") == "returned", out
        assert d.errors_in_row == 0
        for pid, st in d.st.items():
            assert st["status"] == "done" and not [e for e in st["errors"] if "seg" in e]
            assert st["gen_ids"] == uninterrupted(st["prompt_ids"], 500, 400), pid
        ev = [json.loads(x) for x in open(os.path.join(d.state_dir, "events.jsonl"))]
        assert any(e["event"] in ("stream_lost_server_down", "abort_all") for e in ev)


def test_F7_genuine_errors_keep_the_partial_as_stopped(tmp_path):
    """Three errors with the serve up: a prompt that has tokens becomes `stopped` (replayable), not `failed`."""
    from d2 import replay as RP
    with Serve(slots=4, token_delay=0.01) as s:
        d = G.Driver(args(tmp_path, s.url, streams=1, max_errors_in_row=10), prompts(1))
        d.ensure_prompt_ids("p0")
        d.st["p0"]["gen_ids"] = [11, 12, 13]
        d.st["p0"]["status"] = "partial"

        class Err:
            top20, logprob_error = False, None
            body = {"priority": G.AUX_PRIORITY}
            pid, seg, request_id, prompt_ids, t_submit, t_first, t_end, usage = "p0", 0, "x", [], 0.0, None, 1.0, None
            echo, finish, error = None, None, "HTTP 400: bad request"

            def is_alive(self):
                return False

            def snapshot(self):
                return []

            def snapshot_pair(self):
                return self.snapshot(), []
        for k in range(3):
            d.streams["p0"] = Err()
            d.reap()
        assert d.st["p0"]["status"] == "stopped" and d.st["p0"]["gen_ids"] == [11, 12, 13]
        d.close()
        seq = RP.load_seq(d.path("p0"))
        assert seq and seq["finish_reason"] == "error" and len(seq["tokens"]) == len(d.st["p0"]["prompt_ids"]) + 3


# ---------------------------------------------------------------------------------------------------------------- F8
PROMPTS = os.path.join(os.environ.get('D2_PRIVATE_TEST_ROOT', '/opt/dsv41/data/d2'),
                       'prompts/prompts.jsonl')


@pytest.mark.skipif(not os.path.exists(PROMPTS), reason="the private prompt set lives on Controller")
def test_F8_no_heldout_or_dev_prompt_shares_content_with_train():
    from d2 import prompts as P
    ps = [json.loads(x) for x in open(PROMPTS)]
    assert not {p["pid"] for p in ps} & set(P.EXCLUDED_PIDS)
    P.check_disjoint(ps)
    assert P.near_duplicates(ps) == []
    with open(os.path.join(os.path.dirname(PROMPTS), "manifest.json")) as fh:
        man = json.load(fh)
    assert man["n_prompts"] == len(ps) == 538 and man["excluded_pids"] == P.EXCLUDED_PIDS


def test_F8_near_duplicate_check_catches_a_refiled_document():
    from d2 import prompts as P
    words = [f"w{i}" for i in range(400)]
    doc = " ".join(words)
    other = " ".join(f"z{i}" for i in range(400))

    def pr(pid, split, text):
        return {"pid": pid, "split": split, "messages": [{"role": "user", "content": text}]}
    ps = [pr("a", "train", doc), pr("b", "dev", "Summarize: " + doc[:1500]), pr("c", "dev", other)]
    bad = P.near_duplicates(ps)
    assert [x[0] for x in bad] == ["b"]
    with pytest.raises(RuntimeError, match="share content"):
        P.check_near_duplicates(ps)


# ---------------------------------------------------------------------------------------------------------------- F9
def _eq_rec(j, same, decode_rows, kl=None, pos1_same=True, ctl_kl=None, ctl_same=True):
    d = [1, 2, 3, 4, 5]
    r = d if same else ([1, 2, 9, 4, 5] if pos1_same else [7, 2, 3, 4, 5])
    rec = {"key": f"r{j:06d}_b0", "req": "acap0", "P": j, "realized": d, "engine": d,
           "D": {"drafts": d, "margin": [1.0] * 5}, "R": {"drafts": r, "margin": [1.0] * 5},
           "decode_rows_in_window": decode_rows, "main_x_P_rel": 0.0}
    if kl is not None:
        rec["kl_D_R"] = [kl] * 5
    if ctl_kl is not None:                       # the noise control arm N (coordinator amendment 2026-10-23 14:59Z)
        rec["N"] = {"drafts": d if ctl_same else [1, 2, 8, 4, 5], "margin": [1.0] * 5}
        rec["kl_D_N"] = [ctl_kl] * 5
        rec["control"] = {"sigma": 2.0 ** -8, "seed": 20261023}
    return rec


def _report(tmp_path, recs):
    import types
    from d2 import equiv as E
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    with open(work / "drafts-0of1.jsonl", "w") as fh:
        for x in recs:
            fh.write(json.dumps(x) + "\n")
    out = tmp_path / "gate.json"
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(out)))
    return json.load(open(out))


def test_F9_equivalence_gate_counts_only_blocks_whose_arms_differ(tmp_path):
    """S14's probe: 192 real comparisons at 98.96 % equal + 8 identical-input blocks -> 99.0 % over all; must fail."""
    recs = [_eq_rec(j, j < 8 + 190, 0 if j < 8 else 128) for j in range(200)]
    rep = _report(tmp_path, recs)
    assert rep["blocks_inputs_differ"] == 192 and rep["blocks_inputs_identical"] == 8
    assert rep["block_equal_frac_inputs_identical"] == 1.0 and rep["block_equal_frac"] == round(190 / 192, 4)
    # S14's probe reads these two names (offload/s14-d2data 21fb29f test_s14_replay.py)
    assert rep["block_equal_frac_with_decode_rows"] == round(190 / 192, 4)
    assert rep["block_equal_frac_all_blocks"] == round(198 / 200, 4)
    assert rep["decode_rows_in_window_bins"]["0-0"]["blocks"] == 8
    assert rep["decode_rows_in_window_bins"]["128-128"]["blocks"] == 192
    assert not rep["gate"]["pass"] and rep["gate"]["verdict"] == "fail"
    assert not rep["gate"]["rule2_bf16_noise"]["kl_recorded"]


def test_F9_calibrated_bf16_noise_rule(tmp_path):
    """The BF16-noise rule calibrated to the same-run noise control (coordinator amendment 2026-10-23 14:59Z)."""
    # 97 % equal, every mismatch past position 1, KL at the control's level: pass_bf16_noise, and says so
    recs = [_eq_rec(j, j >= 3, 128, kl=1.2e-2, ctl_kl=1e-2, ctl_same=j >= 4) for j in range(100)]
    rep = _report(tmp_path, recs)
    assert rep["gate"]["verdict"] == "pass_bf16_noise" and rep["gate"]["pass"]
    assert rep["gate"]["rule2_bf16_noise"]["control"]["kl_median_per_position"] == [1e-2] * 5
    # KL 10x the control: fail
    recs = [_eq_rec(j, j >= 3, 128, kl=1e-1, ctl_kl=1e-2) for j in range(100)]
    assert _report(tmp_path, recs)["gate"]["verdict"] == "fail"
    # tiny KL but position-1 disagreements on 6 % while the control agrees on 100 % (amendment 2 backstop: 0.95): fail
    recs = [_eq_rec(j, j >= 6, 128, kl=1e-5, pos1_same=False, ctl_kl=1e-5) for j in range(100)]
    assert _report(tmp_path, recs)["gate"]["verdict"] == "fail"
    # no control recorded: the rule cannot be met
    recs = [_eq_rec(j, j >= 3, 128, kl=1e-5) for j in range(100)]
    r2 = _report(tmp_path, recs)["gate"]
    assert r2["verdict"] == "fail" and r2["rule2_bf16_noise"]["reason"] == "noise control not recorded"


# --------------------------------------------------------------------------------------------------------------- F10
def test_F10_dataset_build_requires_a_passed_equivalence_gate(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())

    def build(dest, **kw):
        a = argparse.Namespace(replay=str(out), out=str(tmp_path / dest), prompt_manifest=str(out / "prompt-manifest.json"),
                               taps_source="prefill_replay", splits="", allow_skips=False, equiv_gate="",
                               override_equiv_gate="", override_plumbing="")
        for k, v in kw.items():
            setattr(a, k, v)
        D.cmd_build(a)
        return json.load(open(tmp_path / dest / "MANIFEST.json"))
    with pytest.raises(SystemExit, match="equiv-gate"):
        build("s1")                                       # no gate result: refused (S14's probe)
    with pytest.raises(SystemExit, match="did not pass"):
        build("s2", equiv_gate=_gate(tmp_path, False, out))    # a failed gate: refused
    assert not (tmp_path / "s1").exists() and not (tmp_path / "s2").exists()
    man = build("s3", equiv_gate=_gate(tmp_path, True, out))
    g = man["checks"]["equivalence_gate"]
    assert g["pass"] is True and g["verdict"] == "pass" and len(g["sha256"]) == 64 and g["override"] is None
    assert all(v["plumbing_errors"] == 0 for v in man["checks"]["g1_plumbing"]["capture_set"].values())
    man = build("s4", override_equiv_gate="coordinator: test override")
    assert man["checks"]["equivalence_gate"]["override"] == "coordinator: test override"
    with pytest.raises(SystemExit, match="holds prefill_replay taps"):
        build("s5", taps_source="decode_capture")      # S16 F10-b: the label cannot bypass the gate on replay taps


def test_F10_build_refuses_to_publish_on_plumbing_errors(tmp_path, monkeypatch):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())
    monkeypatch.setattr(D, "plumbing", lambda d, stride=16: {"train": {"plumbing_errors": 1, "first": ["x"]}})
    a = argparse.Namespace(replay=str(out), out=str(tmp_path / "s"), prompt_manifest=str(out / "prompt-manifest.json"), taps_source="prefill_replay",
                           splits="", allow_skips=False, equiv_gate=_gate(tmp_path, True, out), override_equiv_gate="",
                           override_plumbing="")
    with pytest.raises(SystemExit, match="plumbing failed"):
        D.cmd_build(a)
    assert not (tmp_path / "s").exists() and (tmp_path / "s.partial").exists()


# --------------------------------------------------------------------------------------------------------------- F11
def test_F11_replayed_twice_does_not_block_the_copy(tmp_path):
    """S14's probe: the overlay finished a sequence, replay.py lost the response and re-sent it. The re-send is
    recorded once (kind already_complete), nothing is rewritten, and the copy goes through and records it."""
    import types
    import torch
    from d2 import replay as RP
    from d2.tests.test_replay import _FakeServe
    from d2.tests.test_replay_overlay import _call, _load, _round
    spark = tmp_path / "spark"
    spark.mkdir()
    ns = _load(spark, env=True)
    spec = ns["DSparkSpeculator"]()
    prompt, gen = list(range(1, 60)), list(range(100, 141))
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    st = {"schema": "d2.genstate.v1", "pid": "d2p0007", "split": "train", "family": "f", "prompt_sha256": "h",
          "status": "done", "prompt_ids": prompt, "gen_ids": gen, "segments": [{}], "finish_reason": "stop"}
    st["tokens_sha256"] = RP.tokens_sha256(prompt + gen)
    json.dump(st, open(seqs / "d2p0007.json", "w"))
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    e = RP.read_plan(str(out))[0]
    rid = f"cmpl-{e['request_id']}-0-1a2b3c4d"
    first = None
    for k in range(3):                                   # the replay, then two re-sent replays (chunked)
        for lo, n in ((0, 30), (30, e["L"] - 30)):
            ib, aux = _round([(rid, 1 + k, lo, n)])
            ib.input_ids[:n] = torch.tensor((prompt + gen)[lo: lo + n], dtype=torch.int32)
            _call(spec, ib, aux)
        if first is None:
            first = (spark / "d2p0007" / "taps.bin").read_bytes()
    assert (spark / "d2p0007" / "taps.bin").read_bytes() == first          # the first replay is kept
    errs = [json.loads(x) for x in open(spark / "ERRORS.jsonl")]
    assert len(errs) == 1 and errs[0]["kind"] == "already_complete" and errs[0]["seq"] == "d2p0007"
    srv = _FakeServe()
    try:
        a = types.SimpleNamespace(out=str(out), metrics=srv.url + "/metrics", remote_dir=str(spark), ssh="bash -c",
                                  rate=0, limit=0)
        RP.cmd_copy(a)
    finally:
        srv.httpd.shutdown()
    done = json.load(open(out / "taps" / "d2p0007" / "DONE.json"))
    assert done["duplicate_replay"] is True and done["rows_missing"] == 0
    rec = [json.loads(x) for x in open(out / "copy_receipts.jsonl")]
    assert rec[0]["duplicate_replay"] is True


def test_F11_other_overlay_errors_still_block_the_copy(tmp_path):
    import types
    from d2 import replay as RP
    from d2.tests.test_replay import _FakeServe, _gen_state
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    _gen_state(seqs, "d2p0001", "train", list(range(1, 300)), list(range(500, 540)))
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    spark = tmp_path / "spark"
    spark.mkdir()
    (spark / "ERRORS.jsonl").write_text(json.dumps({"kind": "error", "seq": "d2p0001", "error": "x"}) + "\n")
    srv = _FakeServe()
    try:
        with pytest.raises(SystemExit, match="recorded errors"):
            RP.cmd_copy(types.SimpleNamespace(out=str(out), metrics=srv.url + "/metrics", remote_dir=str(spark),
                                              ssh="bash -c", rate=0, limit=0))
    finally:
        srv.httpd.shutdown()


def test_F10_namespace_without_gate_fields_is_refused(tmp_path):
    """S14's original probe (21fb29f) calls cmd_build with a namespace that has no equiv_gate / override fields: that is
    'no gate given' and must refuse with SystemExit, not an AttributeError."""
    import types
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _recs
    out = _fake_replay(tmp_path, _recs())
    a = types.SimpleNamespace(replay=str(out), out=str(tmp_path / "shards"), prompt_manifest=str(out / "prompt-manifest.json"),
                              taps_source="prefill_replay", splits="", allow_skips=False)
    with pytest.raises(SystemExit, match="--equiv-gate"):
        D.cmd_build(a)
    assert not (tmp_path / "shards").exists()
