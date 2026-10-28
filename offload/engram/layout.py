#!/usr/bin/env python3
"""Per-rank Engram row ranges, computed exactly as the engine does (common/engram.py: EngramLayout primes,
ParallelEngramEmbedding head sharding: part_n_hash_cols = cdiv(n_hash_cols, tp), head_start = rank * part,
vocab_start_idx = sum(head_sizes[:head_start]), part_num_embeddings = vocab_end - vocab_start).

    python3 layout.py config.json [--tp 2]          -> JSON {layer: [{rank, row_start, num_rows}, ...]}

No torch/vllm import: the prime search below is a copy of _is_prime / find_next_prime / EngramLayout.__init__.
"""

import argparse
import json


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def layout(tc: dict, tp: int) -> dict:
    layer_ids = list(tc["engram_layer_ids"])
    num_emb = list(tc["engram_num_embeddings"])
    max_ngram, n_heads, vocab = tc["engram_max_ngram_size"], tc["engram_n_heads"], tc["engram_vocab_size"]
    seen: set = set()
    out = {}
    for li, layer in enumerate(layer_ids):
        sizes = []
        for _ in range(max_ngram - 1):
            cur = vocab - 1
            for _ in range(n_heads):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
        n_cols = len(sizes)
        part = -(-n_cols // tp)
        assert sum(sizes) <= num_emb[li], (layer, sum(sizes), num_emb[li])
        ranks = []
        for r in range(tp):
            hs, he = r * part, min(r * part + part, n_cols)
            start, end = sum(sizes[:hs]), sum(sizes[:he])
            ranks.append({"rank": r, "head_start": hs, "row_start": start, "num_rows": end - start})
        out[str(layer)] = {"num_embeddings": num_emb[li], "sum_head_sizes": sum(sizes), "head_sizes": sizes,
                           "ranks": ranks}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tp", type=int, default=2)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    print(json.dumps(layout(cfg.get("text_config", cfg), a.tp), indent=1))
