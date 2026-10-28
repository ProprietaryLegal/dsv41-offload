"""(Copied from lane S14, offload/s14-d2data 21fb29f, d2/s14/fakeserve.py.) S14 adversarial fake of the production vLLM server for d2.generate: a fixed number of slots (requests beyond them
wait), foreign "interactive" requests that occupy slots, /metrics failure modes (HTTP 500, hang), multi-token chunks
(speculative rounds), server-side aborts, a registration lag, and a log of every request and of the moment the server
saw each client disconnect. Tokens are a deterministic function of the prefix (as in the lane's fake), so a resumed
sequence must equal the uninterrupted one."""

from __future__ import annotations

import hashlib
import http.server
import json
import select
import socket
import threading
import time

EOS = 1


def next_token(prefix: list[int]) -> int:
    h = hashlib.sha256(json.dumps(prefix[-4:] + [len(prefix)]).encode()).digest()
    return 10 + int.from_bytes(h[:4], "little") % 5000


def uninterrupted(prompt_ids: list[int], max_tokens: int, stop_at: int) -> list[int]:
    seq, out = list(prompt_ids), []
    while len(out) < max_tokens:
        t = EOS if len(out) >= stop_at else next_token(seq)
        seq.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


class Serve:
    def __init__(self, slots: int = 4, token_delay: float = 0.01, chunk_tokens: int = 1, stop_at: int = 10 ** 9,
                 register_lag: float = 0.0):
        self.slots, self.token_delay, self.chunk_tokens = slots, token_delay, chunk_tokens
        self.stop_at, self.register_lag = stop_at, register_lag
        self.lock = threading.Lock()
        self.running: list[str] = []          # request ids holding a slot (own and foreign)
        self.waiting: list[str] = []          # registered, no slot yet
        self.metrics_mode = "ok"              # ok | 500 | hang
        self.requests: list[dict] = []        # every POST (path, summary, time)
        self.gets: list[tuple[str, float]] = []
        self.disconnects: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self.server_abort_after: dict[str, int] = {}   # request_id -> abort after n tokens (finish_reason "abort")
        self.stop_map: dict[str, int] = {}              # pid -> generated length at which EOS comes (default stop_at)
        self.die = False                                # True: streams drop without a finish, /metrics fails
        self.prompt_len: dict[tuple, int] = {}
        self.max_running_own = 0
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                srv.gets.append((self.path, time.time()))
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                if srv.metrics_mode == "500" or srv.die:
                    self.send_error(500)
                    return
                if srv.metrics_mode == "hang":
                    time.sleep(10)
                with srv.lock:
                    r, w = len(srv.running), len(srv.waiting)
                body = (f'vllm:num_requests_running{{engine="0",model_name="m"}} {float(r)}\n'
                        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {float(w)}\n'
                        f'vllm:num_requests_waiting_by_reason{{engine="0",model_name="m",reason="capacity"}} {float(w)}\n'
                        ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _closed(self) -> bool:
                r, _, _ = select.select([self.connection], [], [], 0)
                if not r:
                    return False
                try:
                    return self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                srv.requests.append({"path": self.path, "t": time.time(),
                                     **{k: v for k, v in body.items() if k not in ("prompt", "messages")},
                                     "n_prompt": len(body.get("prompt") or [])})
                if self.path == "/tokenize":
                    text = json.dumps(body["messages"], sort_keys=True)
                    ids = [0] + [10 + b % 200 for b in hashlib.sha256(text.encode()).digest()]
                    with srv.lock:
                        srv.prompt_len[tuple(ids)] = len(ids)
                    out = json.dumps({"tokens": ids, "count": len(ids), "max_model_len": 10 ** 6}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                rid = body["request_id"]
                pid = rid.split(".")[1] if rid.startswith(("d2g.", "d2c.")) else rid
                stop_at = srv.stop_map.get(pid, srv.stop_at)
                ids = [int(x) for x in body["prompt"]]
                orig = next((n for k, n in srv.prompt_len.items() if tuple(ids[: len(k)]) == k), len(ids))
                base_gen = len(ids) - orig
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(srv.register_lag)
                with srv.lock:
                    srv.waiting.append(rid)
                try:
                    while True:                                   # wait for a slot
                        if self._closed():
                            srv.disconnects[rid] = time.time()
                            return
                        with srv.lock:
                            if len(srv.running) < srv.slots and srv.waiting and srv.waiting[0] == rid:
                                srv.waiting.pop(0)
                                srv.running.append(rid)
                                srv.max_running_own = max(srv.max_running_own,
                                                          sum(1 for x in srv.running if not x.startswith("FG")))
                                break
                        time.sleep(0.005)
                    seq, n_out, first = list(ids), 0, True
                    while n_out < body["max_tokens"]:
                        if self._closed():
                            srv.disconnects[rid] = time.time()
                            return
                        if srv.die:
                            return                                # the server went away mid-stream
                        time.sleep(srv.token_delay * srv.chunk_tokens)
                        new, fin = [], None
                        for _ in range(srv.chunk_tokens):
                            t = EOS if base_gen + n_out >= stop_at else next_token(seq)
                            seq.append(t)
                            new.append(t)
                            n_out += 1
                            if t == EOS:
                                fin = "stop"
                                break
                            if n_out >= body["max_tokens"]:
                                fin = "length"
                                break
                            if srv.server_abort_after.get(rid) == n_out:
                                fin = "abort"
                                break
                        ch = {"choices": [{"index": 0, "text": "", "token_ids": new, "finish_reason": fin,
                                           **({"prompt_token_ids": ids} if first else {})}]}
                        first = False
                        self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                        self.wfile.flush()
                        if fin:
                            break
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    srv.finished[rid] = time.time()
                except (BrokenPipeError, ConnectionResetError):
                    srv.disconnects.setdefault(rid, time.time())
                    return
                finally:
                    with srv.lock:
                        if rid in srv.running:
                            srv.running.remove(rid)
                        if rid in srv.waiting:
                            srv.waiting.remove(rid)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    # foreign (interactive) traffic: occupies a slot (or waits for one) for `seconds`
    def foreign(self, seconds: float, name: str = "FG0") -> threading.Thread:
        def run():
            with self.lock:
                self.waiting.append(name)
            while True:
                with self.lock:
                    if len(self.running) < self.slots and self.waiting and self.waiting[0] == name:
                        self.waiting.pop(0)
                        self.running.append(name)
                        self.finished[name + ".start"] = time.time()
                        break
                time.sleep(0.005)
            time.sleep(seconds)
            with self.lock:
                self.running.remove(name)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def own_running(self) -> int:
        with self.lock:
            return sum(1 for x in self.running + self.waiting if not x.startswith("FG"))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
