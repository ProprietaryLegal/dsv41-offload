#!/usr/bin/env python3
"""Print a table from ofl_pingpong / engram microbench JSON lines (one object per line)."""
import json
import sys

for path in sys.argv[1:]:
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        L = d["lat_us"]
        print(f"{d['backend']:5} {d['label']:22} req={d['req']:7} resp={d['resp']:9} bad={d['bad_bytes']} "
              f"p50={L['p50']:9.1f} p90={L['p90']:9.1f} p99={L['p99']:9.1f} max={L['max']:9.1f} us "
              f"srv={d['server_us_mean']:8.1f} MB/s={d['resp_MBps']:7.0f}")
