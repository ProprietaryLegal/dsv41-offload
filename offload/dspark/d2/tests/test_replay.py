"""replay.py on fakes: plan from generation states + acap exports, prompt_logprobs parsing (ties, outside-top-20 tokens,
broken responses), a run against a fake server, and the direct-link copy through copy_to_x86host with a local shell."""

import hashlib
import http.server
import json
import math
import os
import random
import sys
import threading
import types

import numpy as np
import pytest
import torch

from d2 import replay as RP
from d2 import shards as S

DSPARK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DSPARK not in sys.path:
    sys.path.insert(0, DSPARK)


def _lp_entry(tok: int, p: int, outside: bool) -> dict:
    """A position's prompt_logprobs: 20 ranked entries, plus the committed token (rank > 20) when outside."""
    rng = random.Random(p)
    ids = rng.sample(range(100, 5000), 25)
    if not outside:
        ids[rng.randrange(20)] = tok
    else:
        ids = [i for i in ids if i != tok][:20]
    logits = sorted((rng.gauss(0, 2) for _ in range(21)), reverse=True)
    z = math.log(sum(math.exp(x) for x in logits) + 0.05)
    e = {str(i): {"logprob": logits[k] - z, "rank": k + 1, "decoded_token": "x"} for k, i in enumerate(ids[:20])}
    if outside:
        e[str(tok)] = {"logprob": logits[20] - z - 1.0, "rank": 37, "decoded_token": "x"}
    return e


def _response(tokens, outside_every=7):
    pl = [None] + [_lp_entry(t, p, p % outside_every == 0) for p, t in enumerate(tokens) if p > 0]
    return {"model": "m", "created": 1, "choices": [{"prompt_token_ids": list(tokens), "prompt_logprobs": pl}],
            "usage": {"prompt_tokens": len(tokens), "prompt_tokens_details": {"cached_tokens": 0}}}


def test_parse_logprobs_matches_shard_rules():
    toks = [0] + [100 + (i * 37) % 4000 for i in range(1, 300)]
    arr, chk = RP.parse_logprobs(_response(toks), toks, 200)
    rec = S.SeqRecord({"seq_id": "x"}, torch.tensor(toks, dtype=torch.int32), 200,
                      torch.randn(300 - S.r0_of(200), S.TAPW).to(torch.bfloat16),
                      *(torch.from_numpy(arr[k]) for k in ("top_ids", "top_lp", "tok_lp", "tok_rank", "resid")))
    assert S.check_record(rec) == []
    out = [j for j in range(100) if (200 + j) % 7 == 0]
    assert all(arr["tok_rank"][j] > 20 for j in out) and all(arr["tok_rank"][j] <= 20 for j in range(100)
                                                              if j not in out)
    assert chk["ties_at_edge"] == 0


def test_parse_logprobs_refuses_broken_responses():
    toks = [0] + list(range(100, 160))
    bad = _response(toks)
    bad["choices"][0]["prompt_token_ids"][5] += 1
    with pytest.raises(ValueError, match="different token ids"):
        RP.parse_logprobs(bad, toks, 30)
    bad = _response(toks)
    del bad["choices"][0]["prompt_logprobs"][40][str(toks[40])]
    with pytest.raises(ValueError, match="own logprob"):
        RP.parse_logprobs(bad, toks, 30)
    bad = _response(toks)
    bad["choices"][0]["prompt_logprobs"].pop()
    with pytest.raises(ValueError, match="entries"):
        RP.parse_logprobs(bad, toks, 30)


def _gen_state(d, pid, split, prompt, gen, status="done"):
    s = {"schema": "d2.genstate.v1", "pid": pid, "split": split, "family": "f", "prompt_sha256": "h" + pid,
         "status": status, "prompt_ids": prompt, "gen_ids": gen, "segments": [{}],
         "finish_reason": "stop" if status == "done" else None,
         "tokens_sha256": RP.tokens_sha256(prompt + gen)}
    json.dump(s, open(os.path.join(d, pid + ".json"), "w"))


def test_plan_entries(tmp_path):
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    _gen_state(seqs, "d2p0001", "train", list(range(1, 400)), list(range(500, 700)))
    _gen_state(seqs, "d2p0002", "heldout_legal", list(range(1, 50)), list(range(500, 530)), status="partial")
    _gen_state(seqs, "d2p0003", "train", list(range(1, 50)), [], status="pending")
    acap = tmp_path / "acap"
    acap.mkdir()
    json.dump({"schema": "d2.acap.v1", "seq_id": "acap0", "tokens": list(range(1, 900)), "gen_start": 300,
               "prompt_sha256": "a"}, open(acap / "acap0.json", "w"))
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap=str(acap), out=str(out)))
    plan = RP.read_plan(str(out))
    assert [e["seq_id"] for e in plan] == ["acap0", "d2p0002", "d2p0001"]
    by = {e["seq_id"]: e for e in plan}
    assert by["acap0"]["from"] == 0 and by["acap0"]["request_id"] == "d2r.acap0.0.899"
    assert by["d2p0001"]["from"] == 399 - 128 and by["d2p0001"]["L"] == 599
    assert by["d2p0002"]["from"] == 0 and by["d2p0002"]["finish_reason"] == "aborted"
    assert oct(os.stat(out).st_mode & 0o777) == "0o700"


class _FakeServe:
    def __init__(self, cached=0):
        fake = self
        self.bodies = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, ctype="application/json"):
                b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send({"data": [{"id": RP.MODEL, "root": RP.MODEL_ROOT}]})
                elif self.path == "/metrics":
                    self._send(b'vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\n',
                               "text/plain")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append({k: v for k, v in body.items() if k != "prompt"})
                r = _response(body["prompt"])
                r["usage"]["prompt_tokens_details"]["cached_tokens"] = cached
                self._send(r)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def test_run_writes_arrays(tmp_path):
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    for i in range(3):
        _gen_state(seqs, f"d2p000{i}", "train", list(range(1, 200 + 50 * i)), [100 + (j * 31) % 3000 for j in range(90)])
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    srv = _FakeServe()
    a = types.SimpleNamespace(out=str(out), server=srv.url, model=RP.MODEL, inflight=2, timeout=60, limit=0,
                              allow_production=False, check_remote=False, remote_dir="", ssh="")
    RP.cmd_run(a)
    for e in RP.read_plan(str(out)):
        z = np.load(out / "logprobs" / (e["seq_id"] + ".npz"))
        assert z["top_ids"].shape == (90, 20) and z["resid"].min() >= 0
        meta = json.load(open(out / "logprobs" / (e["seq_id"] + ".json")))
        assert meta["request_id"] == e["request_id"]
    assert all(b["prompt_logprobs"] == 20 and b["max_tokens"] == 1 and b["priority"] == 5 and b["cache_salt"]
               and b["request_id"].startswith("d2r.") for b in srv.bodies)
    assert len({b["cache_salt"] for b in srv.bodies}) == 3
    RP.cmd_run(a)                                     # resumable: nothing left to do
    assert len(srv.bodies) == 3
    srv.httpd.shutdown()


def test_run_refuses_cached_prefill_and_production(tmp_path):
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    _gen_state(seqs, "d2p0000", "train", list(range(1, 200)), list(range(300, 340)))
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    srv = _FakeServe(cached=64)
    a = types.SimpleNamespace(out=str(out), server=srv.url, model=RP.MODEL, inflight=1, timeout=60, limit=0,
                              allow_production=False, check_remote=False, remote_dir="", ssh="")
    with pytest.raises(RuntimeError, match="cached prompt tokens"):
        RP.cmd_run(a)
    assert not list((out / "logprobs").glob("*.npz"))  # rejected replies must not become resumable data
    a.server = "http://HEAD_IP:8000"
    with pytest.raises(SystemExit, match="refusing :8000"):
        RP.check_server(a)
    srv.httpd.shutdown()


def test_copy_through_copy_to_x86host(tmp_path):
    """The copy path end to end with a local shell standing in for ssh to the Spark."""
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    prompt, gen = list(range(1, 300)), [100 + j for j in range(60)]
    _gen_state(seqs, "d2p0007", "train", prompt, gen)
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    e = RP.read_plan(str(out))[0]
    remote = tmp_path / "spark" / "d2p0007"
    remote.mkdir(parents=True)
    rows = e["L"] - e["from"]
    taps = os.urandom(rows * RP.ROW_BYTES)
    ids = np.asarray((prompt + gen)[e["from"]:], dtype="<i4").tobytes()
    (remote / "taps.bin").write_bytes(taps)
    (remote / "ids.bin").write_bytes(ids)
    (remote / "ledger.jsonl").write_text('{"round": 0}\n')
    done = {"seq": "d2p0007", "from": e["from"], "end": e["L"], "rows": rows, "rows_missing": 0, "in_order": True,
            "zero_rows": 0, "sha256_taps": hashlib.sha256(taps).hexdigest(), "sha256_ids": hashlib.sha256(ids).hexdigest()}
    (remote / "DONE.json").write_text(json.dumps(done))
    srv = _FakeServe()
    a = types.SimpleNamespace(out=str(out), metrics=srv.url + "/metrics", remote_dir=str(tmp_path / "spark"),
                              ssh="bash -c", rate=0, limit=0)
    RP.cmd_copy(a)
    d = out / "taps" / "d2p0007"
    assert (d / "taps.bin").read_bytes() == taps and (d / "ids.bin").read_bytes() == ids
    assert json.load(open(d / "DONE.json"))["sha256_taps"] == done["sha256_taps"]
    rec = [json.loads(x) for x in open(out / "copy_receipts.jsonl")]
    assert rec[0]["taps"]["sha256"] == done["sha256_taps"]
    srv.httpd.shutdown()
