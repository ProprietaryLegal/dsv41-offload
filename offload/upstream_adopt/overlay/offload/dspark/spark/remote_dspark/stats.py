# SPDX-License-Identifier: Apache-2.0
"""Counters and latency histograms of the remote DSpark proxy (rank 0)."""

from __future__ import annotations

import json
import math
import os
import threading
import time


class Histogram:
    """Log-spaced histogram of microsecond values (8 bins per octave, 1 µs .. ~1 h)."""

    BINS_PER_OCTAVE = 8

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.n = 0
        self.total = 0.0
        self.max = 0.0

    def add(self, us: float) -> None:
        us = max(float(us), 1.0)
        b = int(math.log2(us) * self.BINS_PER_OCTAVE)
        self.counts[b] = self.counts.get(b, 0) + 1
        self.n += 1
        self.total += us
        if us > self.max:
            self.max = us

    def quantile(self, q: float) -> float:
        if self.n == 0:
            return 0.0
        target = q * self.n
        acc = 0
        for b in sorted(self.counts):
            acc += self.counts[b]
            if acc >= target:
                return 2.0 ** ((b + 1) / self.BINS_PER_OCTAVE)  # upper edge of the bin
        return self.max

    def summary(self) -> dict:
        return {
            "n": self.n,
            "mean_us": round(self.total / self.n, 1) if self.n else 0.0,
            "p50_us": round(self.quantile(0.5), 1),
            "p90_us": round(self.quantile(0.9), 1),
            "p99_us": round(self.quantile(0.99), 1),
            "max_us": round(self.max, 1),
        }


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counters: dict[str, int] = {}
        self.hists: dict[str, Histogram] = {}
        self.extra: dict[str, object] = {}
        self.started = time.time()

    def inc(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def observe(self, key: str, us: float) -> None:
        with self.lock:
            h = self.hists.get(key)
            if h is None:
                h = self.hists[key] = Histogram()
            h.add(us)

    def set(self, key: str, value: object) -> None:
        with self.lock:
            self.extra[key] = value

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "counters": dict(sorted(self.counters.items())),
                "latency": {k: h.summary() for k, h in sorted(self.hists.items())},
                **{k: v for k, v in self.extra.items()},
            }

    def write_json(self, path: str) -> None:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f, indent=1, default=str)
        os.replace(tmp, path)
