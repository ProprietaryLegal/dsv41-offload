#!/usr/bin/env python3
"""EngramRemoteTable (offload/engram/spark/engram_remote.cpp) against the image's EngramDiskTable, on a GPU host.

Runs the engine's lookup pattern without vLLM: ids on the device -> D2H into pinned ids -> enqueue_read (spin-mode
stream memory ops answered by the remote proxy thread) -> a device copy of the staging rows on the same stream, eagerly
and inside a captured CUDA graph replayed with new ids; plus read_rows. Every row is compared with the image's
EngramDiskTable reading the same shard. Failure injection (--inject): a server that freezes (SIGSTOP) or dies must be
answered from the exact local rows, and remote reads must resume after it is back.

    CUDA_VISIBLE_DEVICES=GPU-240876ad-... python3 test_remote_ext.py --image-common DIR --root REPO \\
        --shard /opt/dsv41/x86host-data/engram/model-00047-of-00048.safetensors --manifest REPO/manifest.json \\
        --endpoint rdma://10.202.28.1:18610@10.202.28.1 --row-start 0 --num-rows 192001740 [--inject PID]
"""

import argparse
import hashlib
import json
import os
import signal
import struct
import sys
import time

import torch
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--image-common", required=True, help="dir with the image's engram_disk.cpp")
ap.add_argument("--root", required=True, help="dsv41-deploy checkout (offload/ + manifest.json)")
ap.add_argument("--shard", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--endpoint", required=True)
ap.add_argument("--row-start", type=int, required=True)
ap.add_argument("--num-rows", type=int, required=True)
ap.add_argument("--max-rows", type=int, default=49152)
ap.add_argument("--inject", type=int, default=0, help="pid of a test server to SIGSTOP / kill (failure tests)")
ap.add_argument("--restart-cmd", default="", help="shell command that restarts the injected server")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--out", default="")
a = ap.parse_args()

fails = 0
results = []


def check(name, ok, **info):
    global fails
    fails += 0 if ok else 1
    results.append({"check": name, "pass": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info) if info else "", flush=True)


root = os.path.join(a.root, "offload")
here = a.image_common
srcs = [os.path.join(root, "engram", "spark", "engram_remote.cpp"), os.path.join(root, "engram", "engram_proto.cpp"),
        os.path.join(root, "transport", "ofl.cpp")]
h = hashlib.md5()
for f in srcs + [os.path.join(here, "engram_disk.cpp"), os.path.join(root, "engram", "engram_client.h")]:
    h.update(open(f, "rb").read())
t0 = time.time()
extra_inc = [p for p in os.environ.get("EXTRA_INC", "").split(":") if p]  # x86host: the extracted verbs headers
remote_ext = load(name=f"vllm_engram_remote_{h.hexdigest()[:10]}", sources=srcs,
                  extra_include_paths=[here, os.path.join(root, "engram"), os.path.join(root, "transport"), *extra_inc],
                  extra_cflags=["-O2"], extra_ldflags=[os.environ.get("IBVERBS_LIB", "-libverbs")], with_cuda=True,
                  verbose=False)
disk_ext = load(name="vllm_engram_disk", sources=[os.path.join(here, "engram_disk.cpp")], extra_cflags=["-O2"],
                with_cuda=True, verbose=False)
print(f"built both extensions in {time.time() - t0:.1f} s", flush=True)


def tensor_offsets(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    w = next(k for k in hdr if k.endswith(".engram.embed.weight"))
    s = w[: -len("weight")] + "scale"
    return 8 + n + hdr[w]["data_offsets"][0], 8 + n + hdr[s]["data_offsets"][0]


off_w, off_s = tensor_offsets(a.shard)
R = a.max_rows


def registered(nbytes):  # as the overlay's _allocate_registered_staging
    import mmap

    import numpy as np

    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    owner = np.frombuffer(m, dtype=np.uint8)
    owner[:: mmap.PAGESIZE] = 0
    t = torch.from_numpy(owner)
    r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 3)
    assert r.value == 0, r
    assert t.is_pinned()
    registered.keep.append((m, owner))
    return t


registered.keep = []

torch.cuda.init()
dev = torch.device("cuda")
storage = registered(R * 264)
st_w, st_s = storage[: R * 256].view(R, 256), storage[R * 256:].view(R, 8)
ids_pinned = torch.empty(R, dtype=torch.int64, pin_memory=True)

remote = remote_ext.EngramRemoteTable(a.shard, off_w, a.shard, off_s, a.row_start, a.num_rows, 256, 8, 0, 64,
                                      endpoint=a.endpoint, manifest=a.manifest, timeout_ms=100, retry_ms=1000,
                                      name="test_remote_ext", required=True)
print("remote:", remote.register_staging(st_w.data_ptr(), st_s.data_ptr(), R), flush=True)
local = disk_ext.EngramDiskTable(a.shard, off_w, a.shard, off_s, a.row_start, a.num_rows, 256, 8, 0, 64)
exp_w = torch.empty(R, 256, dtype=torch.uint8, pin_memory=True)
exp_s = torch.empty(R, 8, dtype=torch.uint8, pin_memory=True)
g = torch.Generator().manual_seed(1234)


def make_ids(n):
    ids = torch.randint(0, a.num_rows, (n,), generator=g, dtype=torch.int64)
    k = max(1, n // 20)
    ids[torch.randint(0, n, (k,), generator=g)] = -1                 # padding
    ids[torch.randint(0, n, (k,), generator=g)] = a.num_rows + 3     # past the range
    ids[torch.randint(0, n, (k,), generator=g)] = ids[: k].clone()   # duplicates
    ids[0], ids[-1] = 0, a.num_rows - 1                              # boundaries
    return ids


def expected(ids_cpu, n):
    ids_pinned[:n].copy_(ids_cpu)
    local.read_rows(ids_pinned.data_ptr(), exp_w.data_ptr(), exp_s.data_ptr(), n)
    return exp_w[:n].clone(), exp_s[:n].clone()


def compare(got_w, got_s, ew, es):
    bad = (got_w != ew).any(dim=1) | (got_s != es).any(dim=1)
    return int(bad.sum())


# 1. read_rows (synchronous)
for n in (288, 4096, 49152):
    ids = make_ids(n)
    ew, es = expected(ids, n)
    st_w[:n].fill_(0xA5)
    st_s[:n].fill_(0xA5)
    ids_pinned[:n].copy_(ids)
    remote.read_rows(ids_pinned.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
    check(f"read_rows {n} rows == EngramDiskTable", compare(st_w[:n], st_s[:n], ew, es) == 0)
check("read_rows went remote (no fallback)", remote.remote_stats()["fallbacks"] == 0, stats={
    k: v for k, v in remote.remote_stats().items() if not k.startswith("hist")})

# 2. spin mode on a stream, eager: ids D2H -> enqueue_read -> device copy of the staging rows (same stream)
print("mode:", remote.enable_spin(1000, 256, 2000, 512), flush=True)
stream = torch.cuda.Stream()
out_w = torch.empty(R, 256, dtype=torch.uint8, device=dev)
out_s = torch.empty(R, 8, dtype=torch.uint8, device=dev)
for n in (288, 4096, 16384, 49152):
    for it in range(3):
        ids = make_ids(n)
        ew, es = expected(ids, n)
        ids_dev = ids.to(dev)
        st_w[:n].fill_(0xA5)
        st_s[:n].fill_(0xA5)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            ids_pinned[:n].copy_(ids_dev, non_blocking=True)
            remote.enqueue_read(stream.cuda_stream, ids_pinned.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
            out_w[:n].copy_(st_w[:n], non_blocking=True)
            out_s[:n].copy_(st_s[:n], non_blocking=True)
        stream.synchronize()
        bad = compare(out_w[:n].cpu(), out_s[:n].cpu(), ew, es)
        if it == 2 or bad:
            check(f"spin eager {n} rows == EngramDiskTable (stream order)", bad == 0, iteration=it)

# 3. CUDA graph: capture once (n fixed), replay with new device ids
for n in (288, 4096):
    ids_dev = make_ids(n).to(dev)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ids_pinned[:n].copy_(ids_dev, non_blocking=True)
        remote.enqueue_read(torch.cuda.current_stream().cuda_stream, ids_pinned.data_ptr(), st_w.data_ptr(),
                            st_s.data_ptr(), n)
        out_w[:n].copy_(st_w[:n], non_blocking=True)
        out_s[:n].copy_(st_s[:n], non_blocking=True)
    torch.cuda.synchronize()
    bad_total = 0
    t_rep = []
    for it in range(a.iters):
        ids = make_ids(n)
        ew, es = expected(ids, n)
        ids_dev.copy_(ids.to(dev))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        t_rep.append((time.perf_counter() - t1) * 1e6)
        bad_total += compare(out_w[:n].cpu(), out_s[:n].cpu(), ew, es)
    t_rep.sort()
    check(f"CUDA graph replay {n} rows x {a.iters} == EngramDiskTable", bad_total == 0,
          replay_us_p50=round(t_rep[len(t_rep) // 2], 1))
    del graph

st = remote.remote_stats()
check("spin + graph reads all remote", st["fallbacks"] == 0, requests=st["requests"], rows=st["rows"])

# 4. failure injection: a frozen server (timeouts) and a dead one; exact local rows meanwhile, then remote again
def spin_lookup(n):
    ids = make_ids(n)
    ew, es = expected(ids, n)
    ids_dev = ids.to(dev)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        ids_pinned[:n].copy_(ids_dev, non_blocking=True)
        remote.enqueue_read(stream.cuda_stream, ids_pinned.data_ptr(), st_w.data_ptr(), st_s.data_ptr(), n)
        out_w[:n].copy_(st_w[:n], non_blocking=True)
        out_s[:n].copy_(st_s[:n], non_blocking=True)
    t1 = time.perf_counter()
    stream.synchronize()
    return compare(out_w[:n].cpu(), out_s[:n].cpu(), ew, es), (time.perf_counter() - t1) * 1e3


if a.inject:
    before = remote.remote_stats()
    os.kill(a.inject, signal.SIGSTOP)
    try:
        bad1, ms1 = spin_lookup(4096)
        bad2, ms2 = spin_lookup(4096)
    finally:
        os.kill(a.inject, signal.SIGCONT)
    st = remote.remote_stats()
    check("frozen server: exact rows from the local fallback", bad1 == 0 and bad2 == 0
          and st["fallbacks"] >= before["fallbacks"] + 2, first_ms=round(ms1, 1), second_ms=round(ms2, 1),
          transport_errors=st["transport_errors"] - before["transport_errors"])
    time.sleep(2.5)  # retry_ms = 1000
    bad3, _ = spin_lookup(4096)
    st2 = remote.remote_stats()
    check("after SIGCONT: reconnected and remote again", bad3 == 0 and st2["reconnects"] >= 1
          and st2["requests"] > st["requests"], reconnects=st2["reconnects"])
    os.kill(a.inject, signal.SIGTERM)
    time.sleep(1.0)
    bad4, ms_dead = spin_lookup(16384)
    st3 = remote.remote_stats()
    _, ms4 = spin_lookup(4096)  # down: straight to the local reader
    check("dead server: exact rows from the local fallback", bad4 == 0 and st3["fallbacks"] > st2["fallbacks"],
          transport_errors=st3["transport_errors"] - st2["transport_errors"], failing_lookup_ms=round(ms_dead, 1),
          local_ms=round(ms4, 1))
    if a.restart_cmd:
        os.system(a.restart_cmd)
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(1.5)
            bad5, _ = spin_lookup(288)
            if remote.remote_stats()["requests"] > st3["requests"]:
                break
        st4 = remote.remote_stats()
        check("restarted server: remote again", bad5 == 0 and st4["requests"] > st3["requests"],
              reconnects=st4["reconnects"])

st = remote.remote_stats()
summary = {k: v for k, v in st.items() if not k.startswith("hist")}
print("remote_stats:", json.dumps(summary), flush=True)
print("spin_stats:", remote.spin_stats(), "describe:", remote.describe(), flush=True)
if a.out:
    with open(a.out, "w") as f:
        json.dump({"results": results, "remote_stats": summary, "mode": remote.mode()}, f, indent=1)
print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILED", flush=True)
sys.exit(1 if fails else 0)
