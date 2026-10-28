#!/usr/bin/env python3
"""How many Engram row ids repeat inside one lookup, on real legal text (for the dedup decision).

Replicates the engine's hashing on the CPU (common/engram.py: build_compressed_token_map, compute_hash_multipliers,
EngramLayout primes/offsets via layout.py, _hash_ids_kernel's rolling XOR over n-grams 2..4 with pad blocking), for one
request streamed in prefill chunks of --chunk tokens, and reports per rank and layer the fraction of a lookup's row ids
that are repeats (rows the server could send once), plus the wire bytes that would save.

    python3 dup_rate.py --tokenizer DIR --config config.json --text opinions_cl.jsonl [--tokens 300000] [--chunk 4096]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layout import layout  # noqa: E402


def build_compressed_token_map(tokenizer):  # verbatim logic of common/engram.py
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def multipliers(layer_ids, max_ngram, vocab):
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab) // 2)
    return np.stack([np.random.default_rng(10007 * l).integers(0, bound, size=(max_ngram,), dtype=np.int64) * 2 + 1
                     for l in layer_ids])


def hashes(c, li, mult, primes, offsets, pad_id, max_ngram, n_heads):
    T = len(c)
    pos = np.arange(T)
    rolling = np.zeros(T, dtype=np.int64)
    blocked = np.zeros(T, dtype=bool)
    out = np.empty((T, (max_ngram - 1) * n_heads), dtype=np.int64)
    for shift in range(max_ngram):
        lb = pos - shift
        src = np.where(lb >= 0, c[np.maximum(lb, 0)], 0)
        blocked |= lb < 0
        value = np.where(blocked, pad_id, src).astype(np.int64)
        rolling ^= value * mult[li, shift]
        if shift > 0:
            for h in range(n_heads):
                col = (shift - 1) * n_heads + h
                out[:, col] = rolling % primes[col] + offsets[col]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=300000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=2)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tc = json.load(open(a.config))
    tc = tc.get("text_config", tc)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tmap, vocab = build_compressed_token_map(tok)
    assert vocab == tc["engram_compressed_vocab_size"], (vocab, tc["engram_compressed_vocab_size"])
    tmap = np.array(tmap, dtype=np.int64)
    pad_id = int(tmap[tc["engram_pad_token_id"]])
    lay = layout(tc, a.tp)
    layer_ids = list(tc["engram_layer_ids"])
    mult = multipliers(layer_ids, tc["engram_max_ngram_size"], vocab)
    texts = []
    for path in a.text:
        for line in open(path):
            texts.append(json.loads(line)["text"])
    ids = []
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False))
        if len(ids) >= a.tokens:
            break
    ids = np.array(ids[: a.tokens], dtype=np.int64)
    c = tmap[ids]
    res = {"tokens": int(len(ids)), "chunk": a.chunk, "compressed_vocab": vocab, "layers": {}}
    for li, layer in enumerate(layer_ids):
        L = lay[str(layer)]
        sizes = L["head_sizes"]
        offsets = np.cumsum([0] + sizes[:-1])
        h = hashes(c, li, mult, np.array(sizes, dtype=np.int64), offsets, pad_id, tc["engram_max_ngram_size"],
                   tc["engram_n_heads"])
        assert h.min() >= 0 and h.max() < L["num_embeddings"]
        per_rank = []
        for r in L["ranks"]:
            cols = slice(r["head_start"], r["head_start"] + len(sizes) // a.tp)
            local = h[:, cols] - r["row_start"]
            assert local.min() >= 0 and local.max() < r["num_rows"]
            dups = []
            for s0 in range(0, len(local) - a.chunk + 1, a.chunk):
                blk = local[s0: s0 + a.chunk].ravel()
                dups.append(1 - len(np.unique(blk)) / len(blk))
            by_n = []
            for n in range(3):  # per n-gram size, whole stream
                sub = local[:, n * tc["engram_n_heads"]:(n + 1) * tc["engram_n_heads"]] if a.tp == 1 else None
                by_n.append(None if sub is None else float(1 - len(np.unique(sub)) / sub.size))
            per_rank.append({"rank": r["rank"], "rows_per_chunk": a.chunk * local.shape[1], "chunks": len(dups),
                             "dup_frac_mean": float(np.mean(dups)), "dup_frac_min": float(np.min(dups)),
                             "dup_frac_max": float(np.max(dups)),
                             "wire_bytes_saved_per_chunk": float(np.mean(dups)) * a.chunk * local.shape[1] * 264})
        res["layers"][str(layer)] = per_rank
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
