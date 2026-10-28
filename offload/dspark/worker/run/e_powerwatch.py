#!/usr/bin/env python3
"""Lane E power watchdog for sustained pair runs (run ON x86host, alongside the job; enforces the per-card and total power caps).

Reads the coordinator's 1 s power log (tmux dsv41-powerlog -> ~/dsv41-offload/prod/run/gpu-power-*.csv; never a
second logger) and stops the watched tmux session at once on:
  * a stale log (no new sample for STALE_S seconds: the logger or the host is in trouble),
  * a card missing from a sample, or a reading that is not a number ([N/A], [GPU is lost]),
  * any card above its own power limit + 5 W, or all four together above TOTAL_W (default: the sum of their caps) for 3 samples in a row,
  * more than MEM_MIB of GPU memory on GPU-c4e7a371 held by processes that were not there when the watchdog started
    (the coordinator's budget while the live worker shares that card: 14 GiB).
Writes every sample of the window (the four cards' draw) and the verdict to OUT (JSON lines), so the run's power
trace goes with its results. Exits 0 when the watched session ends normally, 3 after stopping it.
Usage: e_powerwatch.py TMUX_SESSION OUT.jsonl [TOTAL_W=850] [STALE_S=6]
"""

import glob
import json
import os
import subprocess
import sys
import time

sess, out_path = sys.argv[1], sys.argv[2]
MEM_MIB = 14 * 1024
SHARED_UUID = "GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93"
TOTAL_W = float(sys.argv[3]) if len(sys.argv) > 3 else 850.0      # the four cards' caps: 325 + 325 + 100 + 100 W
STALE_S = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
log = sorted(glob.glob(os.path.expanduser("~/dsv41-offload/prod/run/gpu-power-*.csv")), key=os.path.getmtime)[-1]


def apps_on_shared() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[1] == SHARED_UUID:
            res[int(p[0])] = int(p[2])
    return res


PREEXISTING = set(apps_on_shared())             # the live worker (and anything else that was there before us)


def alive() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", sess], capture_output=True).returncode == 0


# started BEFORE the job (so the snapshot above holds only others' processes): wait for the job's session to appear
for _ in range(600):
    if alive():
        break
    time.sleep(0.1)
else:
    print(f"tmux session {sess} did not appear within 60 s", flush=True)
    sys.exit(2)
print(f"watching {sess}; processes already on {SHARED_UUID}: {sorted(PREEXISTING)}", flush=True)


def stop(reason: str, out) -> None:
    out.write(json.dumps({"t": time.time(), "verdict": "STOPPED", "reason": reason}) + "\n")
    out.flush()
    subprocess.run(["tmux", "send-keys", "-t", sess, "C-c"], capture_output=True)
    time.sleep(3)
    subprocess.run(["tmux", "kill-session", "-t", sess], capture_output=True)
    print("STOPPED:", reason, flush=True)
    sys.exit(3)


with open(log) as f, open(out_path, "a") as out:
    f.seek(0, 2)
    last_new = time.time()
    over = 0
    cur_ts, cur, first = None, {}, True

    def close_group():
        global over, first
        if sorted(cur) == ["0", "1", "2", "3"]:
            tot = sum(cur.values())
            out.write(json.dumps({"ts": cur_ts, "w": cur, "total": round(tot, 1)}) + "\n")
            out.flush()
            over = over + 1 if tot > TOTAL_W else 0
            if over >= 3:
                stop(f"four cards together {tot:.0f} W > {TOTAL_W} W for 3 samples", out)
        elif not first:                          # the first group may be cut by where reading started
            stop(f"sample {cur_ts} has {len(cur)} cards, not 4", out)
        first = False

    last_mem = 0.0
    while True:
        if time.time() - last_mem > 2.0:
            last_mem = time.time()
            ours = sum(m for pid, m in apps_on_shared().items() if pid not in PREEXISTING)
            if ours > MEM_MIB:
                stop(f"{ours} MiB of ours on {SHARED_UUID} > {MEM_MIB} MiB (the live worker shares the card)", out)
        line = f.readline()
        if not line:
            if not alive():
                out.write(json.dumps({"t": time.time(), "verdict": "job ended"}) + "\n")
                print("job ended", flush=True)
                sys.exit(0)
            if time.time() - last_new > STALE_S:
                stop(f"power log {log} stale for {time.time() - last_new:.0f} s", out)
            time.sleep(0.2)
            continue
        last_new = time.time()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        ts, idx = parts[0], parts[1]
        try:
            draw = float(parts[3].split()[0])
            limit = float(parts[4].split()[0])
        except ValueError:
            stop(f"unreadable power sample: {line.strip()}", out)
        if draw > limit + 5.0:
            stop(f"GPU {idx} draws {draw} W above its {limit} W limit", out)
        if idx == "0":                           # the logger prints cards 0-3 in order; a sample set starts at 0
            if cur_ts is not None:               # (its timestamps can differ by a millisecond inside one set)
                close_group()
            cur_ts, cur = ts, {}
        if cur_ts is not None:
            cur[idx] = draw
