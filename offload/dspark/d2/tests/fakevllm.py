"""A small fake of the vLLM OpenAI server for driver tests: /metrics, /tokenize, streamed /v1/completions with
return_token_ids, client-disconnect aborts, and injectable "other" requests. Tokens are a deterministic function of the
prefix, so an aborted-and-resumed sequence must equal the uninterrupted one."""

from __future__ import annotations

import hashlib
import http.server
import json
import math
import select
import socket
import threading
import time

EOS = 1


def next_token(prefix: list[int]) -> int:
    h = hashlib.sha256(json.dumps(prefix[-4:] + [len(prefix)]).encode()).digest()
    return 10 + int.from_bytes(h[:4], "little") % 5000


def completion_logprobs(token: int, position: int) -> dict[str, object]:
    """Pinned image: sampled entry first, then top20, dict dedup, token_id keys.

    Cycle through sample inside, outside, and a boundary tie outside top20.
    """
    outside = position % 3 != 0
    ids = list(range(10000, 10020)) if outside else [token] + list(range(10000, 10019))
    values = [-math.log(100) - i * .1 for i in range(20)]
    own = values[-1] if position % 3 == 2 else (values[-1] - 1 if outside else values[0])
    top = {f'token_id:{token}': own}
    top.update({f'token_id:{k}': v for k, v in zip(ids, values)})
    return {'tokens': [f'token_id:{token}'], 'token_logprobs': [own],
            'top_logprobs': [top], 'text_offset': [position]}


def stop_len(prompt_ids: list[int]) -> int:
    """Generated length at which EOS comes, from the original prompt only (so resumes agree)."""
    return 20 + sum(prompt_ids[:8]) % 50


class Fake:
    def __init__(self, token_delay: float = 0.004):
        self.token_delay = token_delay
        self.malformed_at: int | None = None
        self.missing_at: int | None = None
        self.active: dict[str, int] = {}
        self.other = 0
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.prompt_len: dict[tuple, int] = {}   # tokenize registry: first prompt ids -> original length
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                with fake.lock:
                    run = len(fake.active) + fake.other
                body = (f'vllm:num_requests_running{{engine="0",model_name="m"}} {float(run)}\n'
                        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} 0.0\n').encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    text = json.dumps(body["messages"], sort_keys=True)
                    ids = [0] + [10 + b % 200 for b in hashlib.sha256(text.encode()).digest()]
                    with fake.lock:
                        fake.prompt_len[tuple(ids)] = len(ids)
                    out = json.dumps({"tokens": ids, "count": len(ids), "max_model_len": 1000}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                fake.requests.append({k: v for k, v in body.items() if k != "prompt"} | {"n_prompt": len(body["prompt"])})
                rid = body["request_id"]
                ids = [int(x) for x in body["prompt"]]
                orig = next((n for k, n in fake.prompt_len.items() if tuple(ids[: len(k)]) == k), len(ids))
                base_gen = len(ids) - orig
                with fake.lock:
                    fake.active[rid] = 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                seq = list(ids)
                n_out = 0
                first = True
                try:
                    while n_out < body["max_tokens"]:
                        r, _, _ = select.select([self.connection], [], [], 0)
                        if r:
                            try:
                                if self.connection.recv(1, socket.MSG_PEEK) == b"":
                                    return                       # client closed: the request is aborted
                            except OSError:
                                return
                        time.sleep(fake.token_delay)
                        if base_gen + n_out >= stop_len(ids[:orig]):
                            t = EOS
                        else:
                            t = next_token(seq)
                        seq.append(t)
                        n_out += 1
                        fin = "stop" if t == EOS else ("length" if n_out >= body["max_tokens"] else None)
                        ch = {"choices": [{"index": 0, "text": "", "token_ids": [t], "finish_reason": fin,
                                           **({"prompt_token_ids": ids} if first else {})}]}
                        if body.get("logprobs") == 20:
                            lp = completion_logprobs(t, base_gen + n_out - 1)
                            if fake.malformed_at == base_gen + n_out:
                                lp['top_logprobs'] = [{}]
                            if fake.missing_at != base_gen + n_out:
                                ch['choices'][0]['logprobs'] = lp
                        first = False
                        self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                        self.wfile.flush()
                        if fin:
                            break
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    with fake.lock:
                        fake.active.pop(rid, None)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def uninterrupted(prompt_ids: list[int], max_tokens: int) -> list[int]:
    seq, out = list(prompt_ids), []
    while len(out) < max_tokens:
        t = EOS if len(out) >= stop_len(prompt_ids) else next_token(seq)
        seq.append(t)
        out.append(t)
        if t == EOS:
            break
    return out
