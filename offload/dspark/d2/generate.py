"""D2 on-policy generation on production :8000, idle-gated, at production settings, with exact token ids.

  run     (coordinator, tmux, after window 5)
          cd offload/dspark && python3 -m d2.generate run --prompts /opt/dsv41/data/d2/prompts/prompts.jsonl \
              --state /opt/dsv41/data/d2/gen
  status  python3 -m d2.generate status --state /opt/dsv41/data/d2/gen
  smoke   (lane D2-DATA, once) python3 -m d2.generate smoke --prompts ... --state /opt/dsv41/data/d2/gen-smoke

Settings (production): T=1, top_p=1 (no top_k/min_p: the served model ships no generation_config.json, so none apply),
thinking on and reasoning_effort high through chat_template_kwargs, priority 20. Exact ids: the prompt is rendered by
the server's own template (POST /tokenize, chat form) and sent to /v1/completions as token ids with return_token_ids;
the stream's token_ids deltas are the sampled ids, and the echoed prompt_token_ids must equal what was sent.

Idle gate (the brief): submit only when running + waiting - own == 0 has held for >= 60 s (120 s after an abort);
own streams 1..3, default 3 (one sequence slot stays free for NDM's requests); when another request appears (seen on a 0.5 s poll), abort every own stream at
once (closing the connection makes the server abort) and keep the received tokens. A partial response is an exact T=1
sample of its prefix, so the prompt is later resumed from prompt + received tokens (same token-id path), and the
sequence stays one exact sample. Pauses (own streams aborted, nothing submitted): sparkhead ~/dsv41-deploy/HOLD exists,
the PAUSE file exists, or /metrics does not answer with both gauges within 1.5 s (an error or a hang).

window.sh open waits for an idle serve BEFORE it writes HOLD, so the coordinator touches the PAUSE file first.

State: <state>/seqs/<pid>.json (one per prompt, atomic writes, mode 600): prompt ids, generated ids, segments,
status pending|running|partial|done|stopped|failed (stopped: 3 errors with the serve up, partial kept and replayable;
failed: 3 errors and nothing generated). A stream lost while /metrics is down is a pause, never an error. Logs: <state>/events.jsonl, <state>/rate.jsonl. Restarting `run` resumes
(a `running` state from a killed driver counts as partial). Proprietary output (legal-v2a): Controller only, mode 700.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import generation_top20 as T20

SERVER = "http://HEAD_IP:8000"
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
PAUSE_FILE = "/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-GEN.PAUSE"
HOLD_HOST, HOLD_PATH = "sparkhead", "~/dsv41-deploy/HOLD"
CTK = {"thinking": True, "reasoning_effort": "high"}
ORDER = ("heldout_legal", "heldout_general", "dev", "train")
SCHEMA = "d2.genstate.v1"
MAX_STREAMS = 3          # D2-FIX1: leave one of the four sequence slots for legal work
AUX_PRIORITY = 20       # installed image CPU proof: 0 < 10 < 20
GENERATOR_REVISION = "d2-decode-capture-option-ii-v2"
DEFAULT_STREAMS = 3      # coordinator 2026-10-23 (S14 F5): one sequence slot stays free, so NDM's requests start at once


def now() -> float:
    return time.time()


def utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t if t is not None else now()))


def tokens_sha256(ids: list[int]) -> str:
    import array
    a = array.array("i", ids)
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha256(a.tobytes()).hexdigest()


def atomic_json(path: str, obj, mode: int = 0o600) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(obj))           # same bytes as json.dump, ~2.7x faster: the gate thread checkpoints (S32)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------ the server
class Server:
    def __init__(self, base: str, model: str = MODEL, timeout: float = 900.0):
        u = urllib.parse.urlparse(base)
        self.base, self.host, self.port = base.rstrip("/"), u.hostname, u.port or 80
        self.model, self.timeout = model, timeout

    def metrics(self, timeout: float = 1.5) -> tuple[float, float] | None:
        """(running, waiting) summed over all series; None unless the page answers with BOTH gauges within `timeout`
        (an error or a hang both read as blind: the gate then aborts own streams)."""
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=timeout) as r:
                body = r.read().decode()
        except OSError:
            return None
        run = re.findall(r"^vllm:num_requests_running(?:\{[^\n]*\})?\s+([\d.eE+-]+)$", body, re.M)
        wait = re.findall(r"^vllm:num_requests_waiting(?:\{[^\n]*\})?\s+([\d.eE+-]+)$", body, re.M)
        if not run or not wait:
            return None
        return sum(float(x) for x in run), sum(float(x) for x in wait)

    def post_json(self, path: str, body: dict, timeout: float = 120.0) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tokenize_chat(self, messages: list[dict], ctk: dict) -> list[int]:
        r = self.post_json("/tokenize", {"model": self.model, "messages": messages, "add_generation_prompt": True,
                                         "chat_template_kwargs": ctk})
        if r.get("count") != len(r["tokens"]):
            raise RuntimeError(f"/tokenize count {r.get('count')} != {len(r['tokens'])} tokens")
        return [int(x) for x in r["tokens"]]


def hold_exists(host: str = HOLD_HOST, path: str = HOLD_PATH, timeout_s: float = 8.0) -> bool | None:
    """True/False, or None when sparkhead does not answer (treated as a pause: the window protocol is unknown)."""
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, f"test -e {path}"],
                           timeout=timeout_s, capture_output=True)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode in (0, 1):
        return r.returncode == 0
    return None


class HoldPoller(threading.Thread):
    """S14 F6: the ssh HOLD check runs here, off the gate loop, every `every_s`; the gate reads the last result and its
    age (a slow ssh never delays the /metrics polls)."""

    def __init__(self, host: str, path: str, every_s: float, timeout_s: float):
        super().__init__(daemon=True, name="d2g-hold")
        self.host, self.path, self.every_s, self.timeout_s = host, path, every_s, timeout_s
        self.result: tuple[float, bool | None] = (0.0, None)      # (completion time, HOLD?) ; time 0 = no result yet
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            v = hold_exists(self.host, self.path, self.timeout_s)
            self.result = (now(), v)
            self.stopped.wait(self.every_s)


# ------------------------------------------------------------------------------------------------------- a stream
class Stream(threading.Thread):
    """One segment of one prompt: a streamed /v1/completions request on token ids."""

    def __init__(self, server: Server, pid: str, seg: int, prompt_ids: list[int], max_tokens: int,
                 extra: dict | None = None, *, top20: bool = False, pause_file: str = PAUSE_FILE):
        super().__init__(daemon=True, name=f"d2g-{pid}-{seg}")
        self.server, self.pid, self.seg = server, pid, seg
        self.prompt_ids, self.max_tokens = prompt_ids, max_tokens
        self.request_id = f"d2c.{pid}.{seg}.{max(0, len(prompt_ids) - 128)}.{len(prompt_ids)}"
        self.body = {"model": server.model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 1.0,
                     "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
                     "return_token_ids": True, "priority": AUX_PRIORITY, "request_id": self.request_id, **(extra or {})}
        self.top20, self.pause_file = top20, pause_file
        if top20:
            self.body.update(logprobs=20, return_tokens_as_token_ids=True)
        self.logprob_rows: list[T20.Row] = []
        self.logprob_error: str | None = None
        self.tokens: list[int] = []
        self.lock = threading.Lock()
        self.echo: list[int] | None = None
        self.finish: str | None = None
        self.error: str | None = None
        self.usage: dict | None = None
        self.t_submit = now()
        self.t_first: float | None = None
        self.t_end: float | None = None
        self.aborted = False
        self.conn: http.client.HTTPConnection | None = None

    def n(self) -> int:
        with self.lock:
            return len(self.tokens)

    def snapshot(self) -> list[int]:
        with self.lock:
            return list(self.tokens)

    def snapshot_pair(self) -> tuple[list[int], list[T20.Row]]:
        with self.lock:
            return list(self.tokens), list(self.logprob_rows)

    def abort(self) -> None:
        self.aborted = True
        c = self.conn
        if c is not None and c.sock is not None:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass                        # already closed by the server or the reader: the abort holds either way

    def run(self) -> None:
        conn = http.client.HTTPConnection(self.server.host, self.server.port, timeout=self.server.timeout)
        self.conn = conn
        try:
            conn.request("POST", "/v1/completions", body=json.dumps(self.body),
                         headers={"Content-Type": "application/json"})
            if self.aborted:                    # abort() raced the connect: close now, the server aborts
                return
            resp = conn.getresponse()
            if resp.status != 200:
                self.error = f"HTTP {resp.status}: {resp.read()[:400]!r}"
                return
            while not self.aborted:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                ch = json.loads(data)
                if "error" in ch:
                    self.error = f"stream error: {json.dumps(ch['error'])[:400]}"
                    break
                for c in ch.get("choices") or []:
                    if c.get("prompt_token_ids") is not None:
                        self.echo = [int(x) for x in c["prompt_token_ids"]]
                    ids = c.get("token_ids") or []
                    if ids:
                        validated: list[T20.Row] = []
                        if self.top20:
                            try:
                                validated = T20.rows(ids, c.get("logprobs"))
                            except (ValueError, TypeError, KeyError) as exc:
                                self.logprob_error = f"generation top20: {exc}"
                                T20.latch(Path(self.pause_file), self.logprob_error)
                                raise ValueError(self.logprob_error) from exc
                        with self.lock:
                            if not self.tokens:
                                self.t_first = now()
                            self.tokens.extend(int(x) for x in ids)
                            self.logprob_rows.extend(validated)
                    if c.get("finish_reason"):
                        self.finish = c["finish_reason"]
                if ch.get("usage"):
                    self.usage = ch["usage"]
        except (OSError, http.client.HTTPException, ValueError) as e:
            if not self.aborted:
                self.error = f"{type(e).__name__}: {e}"
        finally:
            self.t_end = now()
            try:
                conn.close()
            except OSError:
                pass
            if self.aborted and self.finish is None:
                self.finish = "aborted"
            elif self.finish is None and self.error is None:
                self.error = "stream ended without a finish_reason"


# ------------------------------------------------------------------------------------------------------ the driver
class Driver:
    def __init__(self, a, prompts: list[dict]):
        self.a = a
        self.server = Server(a.server, a.model)
        self.state_dir = a.state
        self.seq_dir = os.path.join(a.state, "seqs")
        os.makedirs(self.seq_dir, mode=0o700, exist_ok=True)
        os.chmod(a.state, 0o700)
        os.chmod(self.seq_dir, 0o700)
        # S14 F4: one driver per state dir (a second one would count the first as foreground traffic, overwrite its
        # per-prompt files and double the budget)
        self._lock_fh = open(os.path.join(a.state, "LOCK"), "a")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fh.close()
            raise SystemExit(f"another d2.generate driver holds {a.state}/LOCK; refusing to start a second one")
        try:
            self._init(a, prompts)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop the HOLD poller and release the state-dir lock (the loop does this when it ends; a killed process
        releases the lock by itself)."""
        if getattr(self, "hold", None) is not None:
            self.hold.stopped.set()
        if self._lock_fh is not None and not self._lock_fh.closed:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()

    def _init(self, a, prompts: list[dict]) -> None:
        if not 1 <= a.streams <= MAX_STREAMS:
            raise SystemExit(f"--streams {a.streams}: own streams must be 1..{MAX_STREAMS} (the serve has 4 sequences)")
        self.events = open(os.path.join(a.state, "events.jsonl"), "a")
        self.ratef = open(os.path.join(a.state, "rate.jsonl"), "a")
        for f in (self.events, self.ratef):
            os.chmod(f.name, 0o600)
        self.prompts = {p["pid"]: p for p in prompts}
        self.order = [p["pid"] for p in prompts]
        self.st: dict[str, dict] = {pid: self.load(pid) for pid in self.order}
        self.streams: dict[str, Stream] = {}
        self.ended: collections.deque = collections.deque()      # (t_end) of recently ended own streams
        self.last_other = now()           # the quiet clock starts at launch: the first submit waits idle_s
        self.backoff = False
        self.positives = 0
        self.hold: HoldPoller | None = None
        self.stop = False
        self.errors_in_row = 0
        self.local = None
        self.t0 = now()
        self.budget_path = os.path.join(a.state, "budget.json")
        saved: dict = {}
        if os.path.exists(self.budget_path):
            with open(self.budget_path) as fh:
                saved = json.load(fh)
        self.busy_s = float(saved.get("busy_s", 0.0))
        self.cap_h = self.resolve_cap(saved)
        self.save_budget()
        self._last_tick = self.t0
        self.obs = {"polls": 0, "idle_polls": 0, "own_aborts": 0, "abort_events": 0, "submits": 0, "pauses": 0,
                    "tokenize_local_mismatch": 0, "gauge_samples": []}
        self.tok_hist: collections.deque = collections.deque()   # (t, total generated tokens)
        self._down = False
        self._last_rate = 0.0
        self._last_submit = 0.0

    # ---- state
    def path(self, pid: str) -> str:
        return os.path.join(self.seq_dir, f"{pid}.json")

    def load(self, pid: str) -> dict:
        p = self.prompts[pid]
        fp = self.path(pid)
        if os.path.exists(fp):
            with open(fp) as fh:
                s = json.load(fh)
            if s["prompt_sha256"] != p["prompt_sha256"]:
                raise RuntimeError(f"{pid}: state prompt_sha256 differs from the prompt set (different prompt build?)")
            if s.get("top20", "off") != getattr(self.a, "top20", "off"):
                raise ValueError(f"{pid}: cannot change top20 mode on existing state")
            if s.get("top20") == "generation":
                T20.validate_saved(s["gen_ids"], s["logprob_rows"])
            if s["status"] == "running":
                s["status"] = "partial" if s["gen_ids"] else "pending"   # the driver died mid-stream
            covered = max((g["start"] + g["n"] for g in s["segments"]), default=0)
            if len(s["gen_ids"]) > covered:
                # A checkpoint saves received IDs before reap can record the dead request's provenance.
                s["segments"].append({"seg": len(s["segments"]), "start": covered,
                                      "n": len(s["gen_ids"]) - covered, "unrecorded": True})
                atomic_json(fp, s)
            return s
        return {"schema": SCHEMA, "pid": pid, "split": p["split"], "family": p["family"],
                "prompt_sha256": p["prompt_sha256"], "status": "pending", "prompt_ids": None, "gen_ids": [],
                "top20": getattr(self.a, "top20", "off"), "logprob_rows": [],
                "segments": [], "errors": [], "max_tokens": self.a.max_tokens, "local_ids_match": None}

    def save(self, pid: str) -> None:
        atomic_json(self.path(pid), self.st[pid])

    def event(self, kind: str, **kw) -> None:
        self.events.write(json.dumps({"t": utc(), "event": kind, **kw}) + "\n")
        self.events.flush()

    # ---- the Spark-time cap (S14 F2): persisted in budget.json, kept by a restart that omits the flag
    def resolve_cap(self, saved: dict) -> float:
        cli = getattr(self.a, "spark_hours", None)
        old = saved.get("spark_hours")
        if cli is None:
            if old is None:
                raise SystemExit("--spark-hours is required on the first run of a state dir (for no cap: "
                                 "--spark-hours 0 --i-mean-uncapped)")
            cap = float(old)
        else:
            cap = float(cli)
            if old is not None and float(old) != cap:
                self.event("spark_hours_changed", old=old, new=cap)
        if cap < 0:
            raise SystemExit(f"--spark-hours {cap} is negative")
        if cap == 0 and not getattr(self.a, "i_mean_uncapped", False):
            raise SystemExit("refusing to run uncapped: pass --spark-hours 0 --i-mean-uncapped explicitly")
        return cap

    def save_budget(self) -> None:
        atomic_json(self.budget_path, {"busy_s": round(self.busy_s, 1), "spark_hours": self.cap_h, "t": utc()})

    # ---- accounting
    def gen_total(self) -> int:
        return sum(len(s["gen_ids"]) for s in self.st.values()) + sum(st.n() for st in self.streams.values())

    def budget_left(self) -> bool:
        """--spark-hours: the time the Spark spends on own streams (>= 1 running), summed across restarts."""
        return not self.cap_h or self.busy_s < self.cap_h * 3600

    def next_pid(self) -> str | None:
        if not self.budget_left():
            return None
        busy = set(self.streams)
        for pid in self.order:                         # resume partials first: sequences end complete
            if pid not in busy and self.st[pid]["status"] == "partial":
                return pid
        if self.gen_total() >= self.a.target_tokens:
            return None
        for pid in self.order:
            if pid not in busy and self.st[pid]["status"] == "pending":
                return pid
        return None

    def work_left(self) -> bool:
        if not self.budget_left():
            return False
        if any(s["status"] == "partial" for s in self.st.values()):
            return True
        return self.gen_total() < self.a.target_tokens and any(s["status"] == "pending" for s in self.st.values())

    # ---- gate inputs
    def paused(self) -> str | None:
        if os.path.exists(self.a.pause_file):
            return "pause_file"
        if self.a.hold_host:
            if self.hold is None:
                self.hold = HoldPoller(self.a.hold_host, self.a.hold_path, self.a.hold_every_s,
                                       getattr(self.a, "hold_timeout_s", 8.0))
                self.hold.start()
                t_end = now() + 1.0                  # the first answer usually takes ~0.3 s: wait for it once
                while self.hold.result[0] == 0.0 and now() < t_end:
                    time.sleep(0.02)
            t, v = self.hold.result
            if now() - t > self.a.hold_stale_s:
                return "hold_stale"                  # no fresh answer from sparkhead: treated as HOLD
            if v is None:
                return "hold_unknown"
            if v:
                return "hold"
        return None

    def own(self) -> int:
        while self.ended and now() - self.ended[0] > self.a.grace_s:
            self.ended.popleft()
        return len(self.streams) + len(self.ended)

    # ---- actions
    def abort_all(self, why: str) -> None:
        if not self.streams:
            return
        for s in self.streams.values():
            s.abort()
        self.obs["own_aborts"] += len(self.streams)
        self.obs["abort_events"] += 1
        self.event("abort_all", why=why, streams=sorted(self.streams), tokens={k: v.n() for k, v in self.streams.items()})
        deadline = now() + 30
        for s in list(self.streams.values()):
            s.join(timeout=max(0.1, deadline - now()))
        self.reap()

    def ensure_prompt_ids(self, pid: str) -> None:
        s = self.st[pid]
        if s["prompt_ids"] is not None:
            return
        p = self.prompts[pid]
        ids = self.server.tokenize_chat(p["messages"], p["chat_template_kwargs"])
        s["prompt_ids"] = ids
        s["prompt_tokens"] = len(ids)
        if self.local is not None:
            loc = self.local.ids(p["messages"], p["chat_template_kwargs"])
            s["local_ids_match"] = loc == ids
            if loc != ids:
                self.obs["tokenize_local_mismatch"] += 1
                self.event("tokenize_local_mismatch", pid=pid, server=len(ids), local=len(loc))
        self.save(pid)

    def submit(self, pid: str) -> None:
        self.ensure_prompt_ids(pid)
        s = self.st[pid]
        room = s["max_tokens"] - len(s["gen_ids"])
        if room <= 0:
            s["status"] = "done"
            self.finalize(pid, "length")
            return
        seg = len(s["segments"])
        st = Stream(self.server, pid, seg, s["prompt_ids"] + s["gen_ids"], room,
                    top20=getattr(self.a, "top20", "off") == "generation", pause_file=self.a.pause_file)
        self.streams[pid] = st
        s["status"] = "running"
        self.save(pid)
        st.start()
        self.obs["submits"] += 1
        self.event("submit", pid=pid, seg=seg, prompt_tokens=len(st.prompt_ids), max_tokens=room)

    def finalize(self, pid: str, reason: str) -> None:
        s = self.st[pid]
        s["finish_reason"] = reason
        s["L"] = len(s["prompt_ids"]) + len(s["gen_ids"])
        s["gen_start"] = len(s["prompt_ids"])
        s["tokens_sha256"] = tokens_sha256(s["prompt_ids"] + s["gen_ids"])
        s["done_utc"] = utc()
        self.save(pid)

    def reap(self) -> None:
        for pid, st in list(self.streams.items()):
            if st.is_alive():
                continue
            del self.streams[pid]
            self.ended.append(st.t_end or now())
            s = self.st[pid]
            got, lp_rows = st.snapshot_pair()
            seg = {"seg": st.seg, "request_id": st.request_id, "start": len(s["gen_ids"]), "n": len(got),
                   "finish": st.finish, "error": st.error, "t_submit": utc(st.t_submit),
                   "ttft_s": round(st.t_first - st.t_submit, 3) if st.t_first else None,
                   "decode_s": round(st.t_end - st.t_first, 3) if st.t_first and st.t_end else None,
                   "usage": st.usage, "priority": st.body["priority"], "generator_revision": GENERATOR_REVISION,
                   "sampling": {k: v for k, v in st.body.items() if k not in
                                ("prompt", "model", "request_id", "stream", "stream_options", "return_token_ids")}}
            if st.echo is not None and st.echo != st.prompt_ids:
                s["errors"].append({"t": utc(), "error": "echoed prompt_token_ids differ from the ids sent"})
                s["status"] = "failed"
                self.save(pid)
                raise RuntimeError(f"{pid} seg {st.seg}: the server echoed different prompt ids (exactness broken)")
            if got and st.echo is None:
                seg["error"] = (seg["error"] or "") + " no prompt_token_ids echo"
            s["gen_ids"].extend(got)
            if st.top20:
                s["logprob_rows"].extend(lp_rows)
            if st.logprob_error:
                self.event("top20_invalid", pid=pid, seg=st.seg, error=st.logprob_error, kept=len(got))
            s["segments"].append(seg)
            if st.error and st.finish not in ("stop", "length") and self.server.metrics() is None:
                # S14 F7: the serve went away (a restart, a window opened without the PAUSE file): a pause, not an
                # error. The received tokens are an exact prefix and stay; the loop backs off until /metrics answers.
                s["status"] = "partial" if s["gen_ids"] else "pending"
                self.backoff = True
                self.last_other = now()
                self.event("stream_lost_server_down", pid=pid, seg=st.seg, error=st.error, kept=len(got))
            elif st.error and st.finish not in ("stop", "length"):
                s["errors"].append({"t": utc(), "seg": st.seg, "error": st.error})
                self.errors_in_row += 1
                fails = sum(1 for e in s["errors"] if "seg" in e)
                if fails >= 3:
                    # never throw away a valid partial: `stopped` is terminal but replayable (replay.load_seq)
                    s["status"] = "stopped" if s["gen_ids"] else "failed"
                    if s["gen_ids"]:
                        self.finalize(pid, "error")
                else:
                    s["status"] = "partial" if s["gen_ids"] else "pending"
                self.event("stream_error", pid=pid, seg=st.seg, error=st.error, kept=len(got))
            elif st.finish in ("stop", "length"):
                s["status"] = "done"
                self.errors_in_row = 0
                self.finalize(pid, st.finish)
                self.event("done", pid=pid, gen=len(s["gen_ids"]), finish=st.finish, segments=len(s["segments"]))
            else:                                   # aborted by us
                s["status"] = "partial" if s["gen_ids"] else "pending"
                self.event("aborted", pid=pid, seg=st.seg, kept=len(got), gen=len(s["gen_ids"]))
            self.save(pid)
            if self.errors_in_row >= self.a.max_errors_in_row:
                raise RuntimeError(f"{self.errors_in_row} stream errors in a row; last: {st.error}")

    def checkpoint_running(self) -> None:
        for pid, st in self.streams.items():
            s = dict(self.st[pid])
            got, lp_rows = st.snapshot_pair()
            s["gen_ids"] = s["gen_ids"] + got
            if st.top20:
                s["logprob_rows"] = s["logprob_rows"] + lp_rows
            s["status"] = "running"
            atomic_json(self.path(pid), s)

    def rate(self, force: bool = False) -> None:
        t = now()
        tot = self.gen_total()
        self.tok_hist.append((t, tot))
        while self.tok_hist and t - self.tok_hist[0][0] > 300:
            self.tok_hist.popleft()
        if not force and t - self._last_rate < self.a.rate_every_s:
            return
        self._last_rate = t
        t_old, n_old = self.tok_hist[0]
        by = collections.Counter(s["status"] for s in self.st.values())
        rec = {"t": utc(t), "elapsed_s": round(t - self.t0), "busy_h": round(self.busy_s / 3600, 3),
               "own_streams": len(self.streams), "gen_tokens": tot,
               "tok_s_5min": round((tot - n_old) / max(1e-9, t - t_old), 2),
               "idle_frac": round(self.obs["idle_polls"] / max(1, self.obs["polls"]), 4),
               "status": dict(by), "submits": self.obs["submits"], "own_aborts": self.obs["own_aborts"],
               "abort_events": self.obs["abort_events"], "pauses": self.obs["pauses"],
               "tokenize_local_mismatch": self.obs["tokenize_local_mismatch"]}
        self.ratef.write(json.dumps(rec) + "\n")
        self.ratef.flush()

    # ---- the loop
    def loop(self, hook=None) -> None:
        last_ck = now()
        paused_why = None
        while not self.stop:
            t = now()
            if self.streams:
                self.busy_s += t - self._last_tick
            self._last_tick = t
            self.reap()
            if self.streams and not self.budget_left():
                # S14 F3: the cap stops running streams too (their partials are valid data, replayed as they are)
                self.abort_all("spark-hours budget")
            why = self.paused()
            if why:
                if paused_why != why:
                    self.obs["pauses"] += 1
                    self.event("pause", why=why)
                    paused_why = why
                self.abort_all(why)
                self.last_other = now()
                self.backoff = True
                time.sleep(self.a.poll_s)
                continue
            if paused_why:
                self.event("resume", after=paused_why)
                paused_why = None
            m = self.server.metrics()
            if m is None:
                # blind (S14 F1): another request could be running unseen, so own streams stop at once (partials kept)
                if not self._down:
                    self.event("server_not_answering")
                    self._down = True
                self.abort_all("metrics unavailable")
                self.backoff = True
                self.last_other = now()
                time.sleep(self.a.poll_s * 2)
                continue
            if self._down:
                self.event("server_answering")
                self._down = False
            running, waiting = m
            own = self.own()
            other = running + waiting - own
            self.obs["polls"] += 1
            if len(self.obs["gauge_samples"]) < 2000 and self.streams:
                self.obs["gauge_samples"].append((round(t - self.t0, 2), running, waiting, own))
            if other > 0:
                self.positives += 1
                self.last_other = t
            else:
                self.positives = 0
                self.obs["idle_polls"] += 1
            if self.positives >= 1 and self.streams:
                self.abort_all(f"other request (running {running:g}, waiting {waiting:g}, own {own})")
                self.backoff = True
            need = self.a.backoff_s if self.backoff else self.a.idle_s
            quiet = t - self.last_other
            if hook is not None:
                hook(self)
            # Do not add a replacement while ended requests still inflate the aggregate own count.
            # Without this bound, back-to-back completions can mask foreign demand indefinitely.
            if quiet >= need and other <= 0 and not self.ended and len(self.streams) < self.a.streams:
                pid = self.next_pid()
                if pid is not None and t - self._last_submit >= self.a.stagger_s:
                    self.submit(pid)
                    self._last_submit = now()
                    self.backoff = False
            if not self.streams and not self.work_left():
                self.event("finished", gen_tokens=self.gen_total())
                break
            if t - last_ck >= self.a.checkpoint_s:
                self.checkpoint_running()
                self.save_budget()
                last_ck = t
            self.rate()
            time.sleep(self.a.poll_s)
        if self.streams:
            self.abort_all("driver stop")
        self.save_budget()
        self.rate(force=True)
        self.close()


def load_prompts(path: str, splits: tuple[str, ...], seed: int) -> list[dict]:
    with open(path) as fh:
        ps = [json.loads(line) for line in fh]
    out = []
    for split in splits:
        sub = [p for p in ps if p["split"] == split]
        if split == "train":
            random.Random(seed).shuffle(sub)           # a prefix of the run is a random sample of the train set
        out.extend(sub)
    return out


def make_local():
    try:
        from .prompts import Renderer
        return Renderer()
    except (ImportError, OSError) as e:
        print(f"local renderer unavailable ({e}); /tokenize ids are used without a local cross-check", flush=True)
        return None


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--top20", choices=("off", "generation"), default="off")
    ap.add_argument("--prompts", default="/opt/dsv41/data/d2/prompts/prompts.jsonl")
    ap.add_argument("--state", required=True)
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--streams", type=int, default=DEFAULT_STREAMS, choices=range(1, MAX_STREAMS + 1),
                    metavar=f"1..{MAX_STREAMS}")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--spark-hours", type=float, default=None,
                    help="cap on Spark time with own streams running, persisted in <state>/budget.json: required on "
                         "the first run; a restart without it keeps the saved cap; 0 = uncapped, only with "
                         "--i-mean-uncapped. Partial sequences left at the cap are valid data, replayed as they are")
    ap.add_argument("--i-mean-uncapped", action="store_true", help="confirms --spark-hours 0")
    ap.add_argument("--splits", default=",".join(ORDER))
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--backoff-s", type=float, default=120.0)
    ap.add_argument("--grace-s", type=float, default=3.0, help="an ended own stream still counts as own this long")
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--stagger-s", type=float, default=2.0)
    ap.add_argument("--checkpoint-s", type=float, default=30.0)
    ap.add_argument("--rate-every-s", type=float, default=60.0)
    ap.add_argument("--max-errors-in-row", type=int, default=5)
    ap.add_argument("--pause-file", default=PAUSE_FILE)
    ap.add_argument("--capture-error-file", default="", help="required absolute ERROR_COUNT.json path on capture host")
    ap.add_argument("--capture-error-host", default="sparkhead")
    ap.add_argument("--hold-host", default=HOLD_HOST, help="'' disables the HOLD check")
    ap.add_argument("--hold-path", default=HOLD_PATH)
    ap.add_argument("--hold-every-s", type=float, default=5.0, help="HOLD poller period (a thread, off the gate)")
    ap.add_argument("--hold-stale-s", type=float, default=10.0, help="an older HOLD answer counts as HOLD")
    ap.add_argument("--hold-timeout-s", type=float, default=8.0, help="ssh timeout of one HOLD check")
    ap.add_argument("--no-local-check", action="store_true")


def cmd_run(a) -> None:
    if os.path.realpath(a.state) == "/opt/dsv41/data/d2/gen":
        raise SystemExit("decode capture requires a NEW second-pass state directory; historical gen is read-only")
    if not a.capture_error_file:
        raise SystemExit("decode capture run requires --capture-error-file for automatic PAUSE")
    from .decode_control import remote_monitor
    monitor = remote_monitor(a.capture_error_host, a.capture_error_file, a.pause_file)
    prompts = load_prompts(a.prompts, tuple(a.splits.split(",")), a.seed)
    d = Driver(a, prompts)
    if not a.no_local_check:
        d.local = make_local()

    def on_signal(signum, frame):
        d.stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    d.event("start", argv=sys.argv[1:], prompts=len(prompts), gen_tokens=d.gen_total(),
            generator_revision=GENERATOR_REVISION, priority=AUX_PRIORITY, max_streams=MAX_STREAMS)
    print(f"{utc()} d2.generate: {len(prompts)} prompts, {d.gen_total()} tokens already generated; state {a.state}",
          flush=True)
    monitor.start()
    try:
        d.loop()
    finally:
        monitor.close()
    d.event("stop", gen_tokens=d.gen_total())
    print(json.dumps(summary(a.state), indent=1), flush=True)


def summary(state: str) -> dict:
    seqs = []
    for f in sorted(os.listdir(os.path.join(state, "seqs"))):
        if f.endswith(".json"):
            with open(os.path.join(state, "seqs", f)) as fh:
                seqs.append(json.load(fh))
    by = collections.defaultdict(lambda: collections.Counter())
    tok = collections.Counter()
    for s in seqs:
        by[s["split"]][s["status"]] += 1
        tok[s["split"]] += len(s["gen_ids"])
    segs = [g for s in seqs for g in s["segments"]]
    dec = [(g["n"], g["decode_s"]) for g in segs if g.get("decode_s") and g["n"] > 1]
    rate = None
    rp = os.path.join(state, "rate.jsonl")
    if os.path.exists(rp):
        with open(rp) as fh:
            lines = fh.readlines()
        rate = json.loads(lines[-1]) if lines else None
    return {"sequences": len(seqs), "by_split": {k: dict(v) for k, v in by.items()},
            "gen_tokens": dict(tok), "gen_tokens_total": sum(tok.values()),
            "segments": len(segs), "aborted_segments": sum(1 for g in segs if g.get("finish") == "aborted"),
            "per_stream_decode_tok_s_mean": round(sum(n for n, _ in dec) / max(1e-9, sum(s for _, s in dec)), 2)
            if dec else None,
            "finish": dict(collections.Counter(s.get("finish_reason") for s in seqs if s["status"] == "done")),
            "last_rate": rate}


def cmd_status(a) -> None:
    print(json.dumps(summary(a.state), indent=1))


# ---------------------------------------------------------------------------------------------------------- smoke
def cmd_smoke(a) -> None:
    """One gated smoke of <= 3 prompts at max_tokens <= 512: template equivalence (/tokenize vs local vs one 1-token
    chat completion), exact echo, a deliberate abort + resume of one stream, TTFT and decode rates."""
    if a.n > 3 or a.max_tokens > 512:
        raise SystemExit("the brief allows one smoke of <= 3 prompts with max_tokens <= 512")
    with open(a.prompts) as fh:
        ps = [json.loads(line) for line in fh]
    train = [p for p in ps if p["split"] == "train"]
    legal = sorted((p for p in train if not p["family"].startswith("general_")), key=lambda p: p["est_tokens"])
    gen = sorted((p for p in train if p["family"].startswith("general_")), key=lambda p: p["est_tokens"])
    mid = min(legal, key=lambda p: abs(p["est_tokens"] - 6000))
    pick = [legal[0], gen[0], mid][: a.n]
    a.target_tokens = 10 ** 9
    a.spark_hours, a.i_mean_uncapped = 0.0, True      # the smoke is bounded by <= 3 prompts x <= 512 tokens
    d = Driver(a, pick)
    d.local = make_local()
    rep: dict = {"server": a.server, "prompts": [{"pid": p["pid"], "family": p["family"], "est_tokens": p["est_tokens"]}
                                                  for p in pick]}
    # 1) the gate first: nothing is sent before idle_s of quiet
    print(f"{utc()} smoke: waiting for {a.idle_s:.0f} s of idle on {a.server}", flush=True)
    t_wait = now()
    quiet_since = None
    while True:
        if d.paused():
            raise SystemExit(f"smoke refused: paused ({d.paused()})")
        m = d.server.metrics()
        if m is None:
            raise SystemExit("smoke refused: /metrics does not answer with both gauges")
        if m[0] + m[1] == 0:
            quiet_since = quiet_since or now()
            if now() - quiet_since >= a.idle_s:
                break
        else:
            quiet_since = None
        if now() - t_wait > a.wait_max_s:
            raise SystemExit(f"smoke: no {a.idle_s:.0f} s idle window within {a.wait_max_s:.0f} s")
        time.sleep(a.poll_s)
    rep["waited_for_idle_s"] = round(now() - t_wait, 1)
    # 2) template equivalence: server /tokenize vs the local engine encoder, then one 1-token chat completion
    eq = []
    for p in pick:
        d.ensure_prompt_ids(p["pid"])
        eq.append({"pid": p["pid"], "server_tokens": len(d.st[p["pid"]]["prompt_ids"]),
                   "local_equal": d.st[p["pid"]]["local_ids_match"]})
    rep["tokenize_vs_local"] = eq
    p0 = pick[0]
    m = d.server.metrics()
    if m is None or m[0] + m[1] != 0:
        raise SystemExit("smoke: the serve became busy before the chat check; not sending")
    chat = d.server.post_json("/v1/chat/completions", {
        "model": a.model, "messages": p0["messages"], "chat_template_kwargs": p0["chat_template_kwargs"],
        "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "return_token_ids": True, "priority": AUX_PRIORITY,
        "request_id": f"d2g.smokechat.{p0['pid']}"}, timeout=300)
    cids = chat.get("prompt_token_ids")
    rep["chat_vs_tokenize"] = {"pid": p0["pid"], "chat_prompt_tokens": len(cids) if cids else None,
                               "equal": cids == d.st[p0["pid"]]["prompt_ids"],
                               "cached_tokens": ((chat.get("usage") or {}).get("prompt_tokens_details") or {}).get(
                                   "cached_tokens")}
    if cids != d.st[p0["pid"]]["prompt_ids"]:
        raise SystemExit(f"smoke: chat completion rendered {len(cids or [])} ids, /tokenize {len(d.st[p0['pid']]['prompt_ids'])}"
                         " (template paths differ: stop and investigate)")
    # 3) streams through the production loop; stream of pick[0] is aborted once after >= 128 tokens, then resumed
    state = {"aborted": False}

    def hook(drv: Driver) -> None:
        st = drv.streams.get(pick[0]["pid"])
        if st is not None and not state["aborted"] and st.n() >= 128:
            state["aborted"] = True
            st.abort()
            st.join(timeout=30)
            drv.event("smoke_deliberate_abort", pid=pick[0]["pid"], kept=st.n())
    a.idle_s = 0.0           # the 60 s idle requirement was met above; the loop still aborts on any other request
    d.last_other = now() - 1
    d.loop(hook=hook)
    seqs = [d.st[p["pid"]] for p in pick]
    rep["streams"] = []
    for s in seqs:
        for g in s["segments"]:
            rep["streams"].append({"pid": s["pid"], "seg": g["seg"], "n": g["n"], "finish": g["finish"],
                                   "prompt_tokens": len(s["prompt_ids"]) + g["start"], "ttft_s": g["ttft_s"],
                                   "decode_tok_s": round((g["n"] - 1) / g["decode_s"], 2) if g.get("decode_s") else None,
                                   "cached_tokens": ((g.get("usage") or {}).get("prompt_tokens_details") or {}).get(
                                       "cached_tokens"), "error": g["error"]})
    rep["resume_exact"] = all(s["status"] == "done" for s in seqs) and state["aborted"]
    rep["abort_events_other_requests"] = d.obs["abort_events"]          # the deliberate abort is not one of these
    rep["gauge_samples"] = d.obs["gauge_samples"][:: max(1, len(d.obs["gauge_samples"]) // 40)]
    rep["wall_s"] = round(now() - d.t0, 1)
    rep["generated_tokens"] = sum(len(s["gen_ids"]) for s in seqs)
    with open(os.path.join(a.state, "smoke_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    os.chmod(os.path.join(a.state, "smoke_report.json"), 0o600)
    print(json.dumps(rep, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    add_common(r)
    s = sp.add_parser("status")
    s.add_argument("--state", required=True)
    k = sp.add_parser("smoke")
    add_common(k)
    k.add_argument("--n", type=int, default=3)
    k.add_argument("--wait-max-s", type=float, default=3600.0)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "status":
        cmd_status(a)
    else:
        if a.max_tokens == 16384:
            a.max_tokens = 512
        cmd_smoke(a)


if __name__ == "__main__":
    main()
