"""D2 shards (schema d2.shard.v1, DATASET.md): writer, reader, integrity checks and a V1 CaptureSet adapter.

Proprietary content (legal-v2a token ids, taps, target probabilities): shards live only on x86host
/opt/dsv41/x86host-data/d2/ (mode 700). Nothing here prints token ids or text.

  Shard(path)            reader: tensors (taps lazily, by slice), sidecar metadata, blocks() per DATASET.md section 4
  write_shard(...)       writer used by dataset.py build and dataset.py synth
  check_record(rec)      DATASET.md section 8 rules that one sequence can be checked against on its own
  ShardCaptureSet(...)   V1 verify.captures.CaptureSet interface over shards (rows, samples, plumbing, window,
                         window_status, arm_inputs), so V1 run_ref / analyze run on D2 data unchanged
"""

from __future__ import annotations

import collections.abc
import dataclasses
import hashlib
import json
import os
from typing import Iterator, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMA = "d2.shard.v1"
MANIFEST_SCHEMA = "d2.manifest.v1"
W = 128                      # drafter window (native config sliding_window)
K = 5                        # dspark_block_size
NOISE = 128799               # dspark_noise_token_id
TOPK = 20
H = 5120
TAP_LAYERS = (37, 38, 39)
TAPW = 3 * H
VOCAB = 129280
SEQ_FIELDS = ("tok_off", "L", "gen_start", "r0", "row_off", "n_rows", "span_off", "n_span")
SPLITS = ("train", "dev", "heldout_legal", "heldout_general", "acap")
TAPS_SOURCES = ("prefill_replay", "decode_capture", "synthetic")
MAX_SHARD_TAP_BYTES = 2 * 1024 ** 3


def tokens_sha256(tokens: Sequence[int] | torch.Tensor) -> str:
    """sha256 of the token ids as int32 little-endian (the identity used across prompt, generation and shards)."""
    t = torch.as_tensor(tokens, dtype=torch.int32) if not isinstance(tokens, torch.Tensor) else tokens.to(torch.int32)
    return hashlib.sha256(t.contiguous().numpy().astype("<i4", copy=False).tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def r0_of(gen_start: int) -> int:
    return max(0, gen_start - W)


@dataclasses.dataclass
class SeqRecord:
    """One sequence ready to be written (DATASET.md section 3)."""
    meta: dict                 # seq_id, split, family, source, source_docs, prompt_sha256, finish_reason, segments, ...
    tokens: torch.Tensor       # int32 [L]
    gen_start: int
    taps: torch.Tensor         # bf16 [L - r0, TAPW]
    top_ids: torch.Tensor      # int32 [L - gen_start, TOPK]
    top_lp: torch.Tensor       # float32 [L - gen_start, TOPK]
    tok_lp: torch.Tensor       # float32 [L - gen_start]
    tok_rank: torch.Tensor     # int32 [L - gen_start]
    resid: torch.Tensor        # float32 [L - gen_start]

    @property
    def L(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def r0(self) -> int:
        return r0_of(self.gen_start)


def check_record(rec: SeqRecord) -> list[str]:
    """DATASET.md section 8 rules 3 and 5 for one sequence; returns the violations (empty = clean)."""
    err: list[str] = []
    sid = rec.meta.get("seq_id", "?")
    L, g, r0 = rec.L, rec.gen_start, rec.r0
    if not 1 <= g <= L - 1:
        return [f"{sid}: gen_start {g} outside 1..L-1 (L={L})"]
    n_span = L - g
    shapes = {"taps": (L - r0, TAPW), "top_ids": (n_span, TOPK), "top_lp": (n_span, TOPK), "tok_lp": (n_span,),
              "tok_rank": (n_span,), "resid": (n_span,)}
    dtypes = {"taps": torch.bfloat16, "top_ids": torch.int32, "top_lp": torch.float32, "tok_lp": torch.float32,
              "tok_rank": torch.int32, "resid": torch.float32}
    for name, shape in shapes.items():
        t = getattr(rec, name)
        if tuple(t.shape) != shape or t.dtype != dtypes[name]:
            err.append(f"{sid}: {name} {tuple(t.shape)} {t.dtype}, want {shape} {dtypes[name]}")
    if rec.tokens.dtype != torch.int32 or rec.tokens.dim() != 1:
        err.append(f"{sid}: tokens {rec.tokens.dtype} dim {rec.tokens.dim()}, want int32 [L]")
    if err:
        return err
    if not bool(torch.isfinite(rec.taps).all()):
        err.append(f"{sid}: non-finite taps")
    zero = (rec.taps.view(-1, 3, TAPW // 3) == 0).all(-1).any(-1)
    if bool(zero.any()):
        err.append(f"{sid}: {int(zero.sum())} taps rows with a layer all zero (first at position "
                   f"{rec.r0 + int(zero.nonzero()[0])}; trimmed eager-prefill rows?)")
    if int(rec.tokens.min()) < 0 or int(rec.tokens.max()) >= VOCAB:
        err.append(f"{sid}: token id outside 0..{VOCAB - 1}")
    lp = rec.top_lp
    if not bool(torch.isfinite(lp).all()) or bool((lp > 1e-6).any()):
        err.append(f"{sid}: top_lp not finite or above 0")
    if bool((lp[:, 1:] > lp[:, :-1] + 1e-6).any()):
        err.append(f"{sid}: top_lp not non-increasing")
    if bool(((rec.top_ids < 0) | (rec.top_ids >= VOCAB)).any()):
        err.append(f"{sid}: top_ids outside the vocabulary")
    srt = rec.top_ids.sort(dim=-1).values
    if bool((srt[:, 1:] == srt[:, :-1]).any()):
        err.append(f"{sid}: repeated id within a top-20 row")
    mass = lp.double().exp().sum(-1)
    if bool((mass > 1 + 1e-4).any()):
        err.append(f"{sid}: top-20 mass above 1 (max {float(mass.max()):.6f})")
    want_resid = (1 - mass).clamp(0, 1).float()
    if not torch.allclose(rec.resid, want_resid, atol=1e-5):
        err.append(f"{sid}: resid != max(0, 1 - sum exp(top_lp))")
    span_tok = rec.tokens[g:].long()
    if not bool(torch.isfinite(rec.tok_lp).all()) or bool((rec.tok_lp > 1e-6).any()):
        err.append(f"{sid}: tok_lp not finite or above 0")
    hit = rec.top_ids.long() == span_tok[:, None]
    in_top = hit.any(-1)
    if not torch.equal(in_top, rec.tok_rank <= TOPK):
        err.append(f"{sid}: tok_rank <= {TOPK} disagrees with the committed token's presence in top_ids")
    if bool((rec.tok_rank < 1).any()):
        err.append(f"{sid}: tok_rank below 1")
    if bool(in_top.any()):
        at = hit.float().argmax(-1)
        idx = in_top.nonzero().flatten()
        lp_hit = lp[idx, at[idx]]
        if not torch.allclose(lp_hit, rec.tok_lp[idx], atol=1e-5):
            err.append(f"{sid}: tok_lp != top_lp at the committed token")
        rk_hit = at[idx].to(torch.int32) + 1
        # ties: a tied neighbour may sit before the committed token in the row; the rank must still point at a tie
        bad = (rk_hit != rec.tok_rank[idx]) & (rec.tok_rank[idx] <= TOPK)   # rank > TOPK here is reported above
        if bool(bad.any()):
            j = idx[bad]
            if not torch.allclose(lp[j, rec.tok_rank[j].long() - 1], rec.tok_lp[j], atol=1e-5):
                err.append(f"{sid}: tok_rank != position of the committed token in top_ids")
    out_top = ~in_top
    if bool(out_top.any()) and bool((rec.tok_lp[out_top] > lp[out_top, -1] + 1e-5).any()):
        err.append(f"{sid}: a committed token outside the top-20 has a logprob above the 20th")
    return err


def write_shard(base: str, split: str, records: list[SeqRecord], taps_source: str,
                extra_meta: dict | None = None) -> dict:
    """Write <base>.safetensors + <base>.json (DATASET.md section 6); returns the manifest file entry."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not in {SPLITS}")
    if taps_source not in TAPS_SOURCES:
        raise ValueError(f"taps_source {taps_source!r} not in {TAPS_SOURCES}")
    if not records:
        raise ValueError("empty shard")
    errs = [e for r in records for e in check_record(r)]
    if errs:
        raise ValueError(f"{len(errs)} integrity errors, first: {errs[0]}")
    for r in records:
        if r.meta.get("split") != split:
            raise ValueError(f"{r.meta.get('seq_id')}: split {r.meta.get('split')!r} written into a {split} shard")
    idx, metas = [], []
    tok_off = row_off = span_off = 0
    for i, r in enumerate(records):
        L, g, r0 = r.L, r.gen_start, r.r0
        idx.append([tok_off, L, g, r0, row_off, L - r0, span_off, L - g])
        m = dict(r.meta)
        m.update(index=i, L=L, gen_start=g, n_gen=L - g, tokens_sha256=tokens_sha256(r.tokens))
        metas.append(m)
        tok_off += L
        row_off += L - r0
        span_off += L - g
    if row_off * TAPW * 2 > MAX_SHARD_TAP_BYTES:
        raise ValueError(f"shard taps {row_off * TAPW * 2} bytes above {MAX_SHARD_TAP_BYTES}")
    tensors = {
        "seq_index": torch.tensor(idx, dtype=torch.int64),
        "tokens": torch.cat([r.tokens for r in records]).contiguous(),
        "taps": torch.cat([r.taps for r in records]).contiguous(),
        "row_pos": torch.cat([torch.arange(r.r0, r.L, dtype=torch.int32) for r in records]),
        "top_ids": torch.cat([r.top_ids for r in records]).contiguous(),
        "top_lp": torch.cat([r.top_lp for r in records]).contiguous(),
        "tok_lp": torch.cat([r.tok_lp for r in records]).contiguous(),
        "tok_rank": torch.cat([r.tok_rank for r in records]).contiguous(),
        "resid": torch.cat([r.resid for r in records]).contiguous(),
    }
    path = base + ".safetensors"
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={"schema": SCHEMA, "split": split})
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    digest = sha256_file(path)
    side = {"schema": SCHEMA, "split": split, "file": os.path.basename(path), "sha256": digest, "W": W, "K": K,
            "topk": TOPK, "tap_layers": list(TAP_LAYERS), "hidden": H, "noise_token": NOISE,
            "taps_source": taps_source, **(extra_meta or {}), "sequences": metas}
    side_path = base + ".json"
    with open(side_path + ".tmp", "w") as fh:
        json.dump(side, fh, indent=1)
    os.chmod(side_path + ".tmp", 0o600)
    os.replace(side_path + ".tmp", side_path)
    return {"file": path, "sidecar": side_path, "sha256": digest, "sidecar_sha256": sha256_file(side_path),
            "bytes": os.path.getsize(path), "n_seq": len(records), "n_rows": row_off, "n_span": span_off}


class Shard:
    """Read one shard. Small tensors are loaded; taps stay in the file and are read by slice."""

    SMALL = ("seq_index", "tokens", "row_pos", "top_ids", "top_lp", "tok_lp", "tok_rank", "resid")

    def __init__(self, path: str, verify_sha256: bool = False):
        if not path.endswith(".safetensors"):
            raise ValueError(f"{path}: not a .safetensors shard")
        self.path = path
        with open(path[: -len(".safetensors")] + ".json") as fh:
            self.side = json.load(fh)
        if self.side.get("schema") != SCHEMA:
            raise ValueError(f"{path}: sidecar schema {self.side.get('schema')!r}, want {SCHEMA}")
        if verify_sha256 and sha256_file(path) != self.side["sha256"]:
            raise ValueError(f"{path}: sha256 differs from its sidecar")
        self._f = safe_open(path, framework="pt", device="cpu")
        md = self._f.metadata() or {}
        if md.get("schema") != SCHEMA or md.get("split") != self.side["split"]:
            raise ValueError(f"{path}: header metadata {md} disagrees with the sidecar")
        self.split = self.side["split"]
        self.t = {n: self._f.get_tensor(n) for n in self.SMALL}
        self.meta = self.side["sequences"]
        if self.t["seq_index"].shape != (len(self.meta), len(SEQ_FIELDS)):
            raise ValueError(f"{path}: seq_index {tuple(self.t['seq_index'].shape)} vs {len(self.meta)} sequences")
        self._taps = self._f.get_slice("taps")
        if list(self._taps.get_shape()) != [int(self.t["row_pos"].shape[0]), TAPW]:
            raise ValueError(f"{path}: taps shape {self._taps.get_shape()}")

    def __len__(self) -> int:
        return len(self.meta)

    def seq(self, i: int) -> dict:
        """seq_index fields of sequence i (ints) merged over its sidecar metadata."""
        v = self.t["seq_index"][i].tolist()
        return {**self.meta[i], **dict(zip(SEQ_FIELDS, v))}

    def tokens(self, i: int) -> torch.Tensor:
        s = self.seq(i)
        return self.t["tokens"][s["tok_off"]: s["tok_off"] + s["L"]]

    def taps(self, i: int, p0: int, p1: int) -> torch.Tensor:
        """bf16 [p1 - p0, TAPW] taps rows of positions p0 .. p1-1 of sequence i."""
        s = self.seq(i)
        if not s["r0"] <= p0 <= p1 <= s["L"]:
            raise IndexError(f"rows {p0}..{p1} outside {s['r0']}..{s['L']} of sequence {i}")
        a = s["row_off"] + p0 - s["r0"]
        return self._taps[a: a + (p1 - p0)]

    def span(self, i: int) -> dict[str, torch.Tensor]:
        """Target arrays over sequence i's span (row j = position gen_start + j)."""
        s = self.seq(i)
        a, n = s["span_off"], s["n_span"]
        return {k: self.t[k][a: a + n] for k in ("top_ids", "top_lp", "tok_lp", "tok_rank", "resid")}

    def blocks(self, i: int) -> range:
        """Every anchor-1 position P of sequence i's training blocks (DATASET.md section 4)."""
        s = self.seq(i)
        return range(s["gen_start"] - 1, s["L"] - 2)

    def block(self, i: int, P: int) -> dict:
        """One training block: window rows, anchor, and per draft position its target or None (partial block)."""
        s = self.seq(i)
        if not s["gen_start"] - 1 <= P <= s["L"] - 3:
            raise IndexError(f"P {P} outside the block range of sequence {i}")
        tok = self.tokens(i)
        lo = max(0, P - W + 1)
        sp = self.span(i)
        tgt = []
        for d in range(K):
            t = P + 2 + d
            if t > s["L"] - 1:
                tgt.append(None)
                continue
            j = t - s["gen_start"]
            tgt.append({"pos": t, "token": int(tok[t]), "prev": int(tok[t - 1]), "top_ids": sp["top_ids"][j],
                        "top_lp": sp["top_lp"][j], "resid": float(sp["resid"][j])})
        return {"P": P, "window_pos": list(range(lo, P + 1)), "window_taps": self.taps(i, lo, P + 1),
                "anchor": int(tok[P + 1]), "targets": tgt}

    def structure_errors(self) -> list[str]:
        """Cross-checks of the stored offsets (DATASET.md sections 3, 6.1); empty = clean."""
        err: list[str] = []
        tok_off = row_off = span_off = 0
        rp = self.t["row_pos"]
        for i in range(len(self)):
            s = self.seq(i)
            want = (tok_off, row_off, span_off)
            got = (s["tok_off"], s["row_off"], s["span_off"])
            if got != want:
                err.append(f"seq {i}: offsets {got}, want {want}")
            if s["r0"] != r0_of(s["gen_start"]) or s["n_rows"] != s["L"] - s["r0"] or \
                    s["n_span"] != s["L"] - s["gen_start"]:
                err.append(f"seq {i}: r0/n_rows/n_span inconsistent with L {s['L']} gen_start {s['gen_start']}")
            if not torch.equal(rp[s["row_off"]: s["row_off"] + s["n_rows"]],
                               torch.arange(s["r0"], s["L"], dtype=torch.int32)):
                err.append(f"seq {i}: row_pos is not r0..L-1")
            if tokens_sha256(self.tokens(i)) != s.get("tokens_sha256"):
                err.append(f"seq {i}: tokens_sha256 mismatch")
            tok_off += s["L"]
            row_off += s["n_rows"]
            span_off += s["n_span"]
        if (tok_off, row_off, span_off) != (self.t["tokens"].shape[0], rp.shape[0], self.t["tok_lp"].shape[0]):
            err.append("totals of seq_index differ from the tensor lengths")
        return err

    def record(self, i: int) -> SeqRecord:
        """Sequence i as a SeqRecord (taps read in full; used by re-checks and tests)."""
        s = self.seq(i)
        sp = self.span(i)
        meta = {k: v for k, v in self.meta[i].items() if k not in ("index", "L", "gen_start", "n_gen", "tokens_sha256")}
        return SeqRecord(meta, self.tokens(i).clone(), s["gen_start"], self.taps(i, s["r0"], s["L"]).clone(),
                         sp["top_ids"].clone(), sp["top_lp"].clone(), sp["tok_lp"].clone(), sp["tok_rank"].clone(),
                         sp["resid"].clone())


def shard_paths(root: str, split: str) -> list[str]:
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


# ------------------------------------------------------------------------------------------- V1 CaptureSet adapter
@dataclasses.dataclass
class Row:
    """V1 captures.Row: tok, round (= position here: any window row is older than its sample), mx, taps."""
    tok: int
    round: int
    mx: torch.Tensor | None
    taps: torch.Tensor | None


@dataclasses.dataclass
class Sample:
    """V1 captures.Sample fields. round = P and b = the sequence's index in the set, so `key` (r%06d_b%d) is unique
    and V1 analyze's dump glob (r*_b*.pt) finds it. engine_drafts holds caller-supplied drafts (e.g. the K2 worker's,
    g1.py), else NOISE placeholders (engine_drafts_placeholder True)."""
    round: int
    b: int
    req: str
    B: int
    num_tokens: int
    P: int
    anchor: int
    engine_drafts: list[int]
    engine_conf: list[float] | None
    temperature: float
    realized: list[int | None] = dataclasses.field(default_factory=list)
    engine_scheduled: int | None = None
    engine_accepted: int | None = None
    engine_drafts_placeholder: bool = False

    @property
    def key(self) -> str:
        return f"r{self.round:06d}_b{self.b}"


_ZERO_MX = torch.zeros(H, dtype=torch.bfloat16)


class _SeqRows(collections.abc.Mapping):
    """position -> Row for one sequence, built on access from the shard (taps read by slice)."""

    def __init__(self, shard: Shard, i: int):
        self.shard, self.i = shard, i
        s = shard.seq(i)
        self.r0, self.L = s["r0"], s["L"]
        self.tok = shard.tokens(i)

    def __getitem__(self, p: int) -> Row:
        if not self.r0 <= p < self.L:
            raise KeyError(p)
        # D2 has no engine main_x; V1 prepare_window only logs engine-vs-native main_x from it (a zero row reads as
        # relative error 1.0 in run_ref's mainx_engine_vs_native.json, which is meaningless for D2 sets)
        return Row(int(self.tok[p]), p, _ZERO_MX, self.shard.taps(self.i, p, p + 1)[0].clone())

    def __contains__(self, p: object) -> bool:
        return isinstance(p, int) and self.r0 <= p < self.L

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.r0, self.L))

    def __len__(self) -> int:
        return self.L - self.r0


class ShardCaptureSet:
    """V1 CaptureSet interface over D2 shards. Every block of the selected sequences is a sample (stride/limit thin
    them evenly per sequence). Structural checks fill `plumbing` (a violation is never skipped quietly)."""

    WINDOW = W

    def __init__(self, paths: list[str], seq_ids: set[str] | None = None, stride: int = 1,
                 drafts: dict[str, list[int]] | None = None, verify_sha256: bool = False,
                 drafts_by_seq: dict[str, dict[int, list[int]]] | None = None):
        if not paths:
            raise FileNotFoundError("no shard paths given")
        self.shards = [Shard(p, verify_sha256=verify_sha256) for p in paths]
        self.rows: dict[str, _SeqRows] = {}
        self.samples: list[Sample] = []
        self.plumbing: list[str] = []
        self.unverified_tail: list[str] = []
        self.seq_meta: dict[str, dict] = {}
        n = 0
        for sh in self.shards:
            self.plumbing.extend(f"{os.path.basename(sh.path)} {e}" for e in sh.structure_errors())
            for i in range(len(sh)):
                s = sh.seq(i)
                sid = s["seq_id"]
                if seq_ids is not None and sid not in seq_ids:
                    continue
                if sid in self.rows:
                    self.plumbing.append(f"{sid}: present in two shards")
                    continue
                rows = _SeqRows(sh, i)
                self.rows[sid] = rows
                self.seq_meta[sid] = s
                tok = rows.tok
                sp = sh.span(i)
                hit = sp["top_ids"].long() == tok[s["gen_start"]:].long()[:, None]
                if not torch.equal(hit.any(-1), sp["tok_rank"] <= TOPK):
                    self.plumbing.append(f"{sid}: tok_rank disagrees with top_ids")
                for P in sh.blocks(i)[::stride]:
                    key_b = n
                    realized = [int(tok[t]) if t < s["L"] else None for t in range(P + 2, P + 2 + K)]
                    smp = Sample(P, key_b, sid, 1, 1, P, int(tok[P + 1]), [], None, 1.0, realized)
                    d = None if drafts is None else drafts.get(smp.key)
                    if d is None and drafts_by_seq is not None:            # g1.py k2 output: seq_id -> {P: drafts}
                        d = drafts_by_seq.get(sid, {}).get(P)
                    if d is None:
                        smp.engine_drafts = [NOISE] * K
                        smp.engine_drafts_placeholder = True
                    else:
                        if len(d) != K:
                            self.plumbing.append(f"{smp.key}: {len(d)} supplied drafts")
                        smp.engine_drafts = [int(x) for x in d]
                    self.samples.append(smp)
                n += 1

    def window(self, s: Sample) -> list[int]:
        return list(range(max(0, s.P - W + 1), s.P + 1))

    def window_status(self, s: Sample) -> dict:
        rows = self.rows[s.req]
        pos = self.window(s)
        missing = [p for p in pos if p not in rows]
        return {"n": len(pos), "missing": missing, "no_taps": []}

    def arm_inputs(self, s: Sample) -> dict:
        st = self.window_status(s)
        if st["missing"]:
            raise RuntimeError(f"{s.key}: window positions without a row: {st['missing'][:5]}")
        sh, i = self.rows[s.req].shard, self.rows[s.req].i
        lo = max(0, s.P - W + 1)
        taps = sh.taps(i, lo, s.P + 1)
        win = [(p, "taps", taps[j]) for j, p in enumerate(range(lo, s.P + 1))]
        return {"key": s.key, "req": s.req, "P": s.P, "anchor": s.anchor, "window": win,
                "block_positions": list(range(s.P + 1, s.P + 1 + K)), "noise_token": NOISE}


# ------------------------------------------------------------------------------------------------- synthetic data
def synth_record(seq_id: str, split: str, L: int, gen_start: int, seed: int, family: str = "synthetic") -> SeqRecord:
    """A schema-valid random sequence (tests; dataset.py synth). Tokens are random ids, taps N(0,1) bf16, and the
    top-20 a random distribution whose committed token sits inside the top-20 about 90 % of the time."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (L,), generator=g, dtype=torch.int32)
    r0 = r0_of(gen_start)
    taps = torch.randn(L - r0, TAPW, generator=g).to(torch.bfloat16)
    n = L - gen_start
    top_ids = torch.empty(n, TOPK, dtype=torch.int32)
    top_lp = torch.empty(n, TOPK, dtype=torch.float32)
    tok_lp = torch.empty(n, dtype=torch.float32)
    tok_rank = torch.empty(n, dtype=torch.int32)
    for j in range(n):
        t = int(tokens[gen_start + j])
        logits = torch.randn(TOPK + 1, generator=g, dtype=torch.float64) * 2.0
        lp = torch.log_softmax(logits, -1)          # 21 masses: 20 kept + the tail
        lp_top, _ = lp[:TOPK].sort(descending=True)
        cand = torch.randperm(VOCAB, generator=g)[: TOPK + 1].to(torch.int32)
        ids = cand[cand != t][:TOPK].clone()                 # 20 distinct ids, none of them the committed token
        if float(torch.rand(1, generator=g)) < 0.9:
            k = int(torch.randint(0, TOPK, (1,), generator=g))
            ids[k] = t
            tok_lp[j], tok_rank[j] = float(lp_top[k]), k + 1
        else:
            tok_lp[j], tok_rank[j] = float(lp_top[-1]) - 1.0 - float(torch.rand(1, generator=g)), TOPK + 1
        top_ids[j], top_lp[j] = ids, lp_top.float()
    resid = (1 - top_lp.double().exp().sum(-1)).clamp(0, 1).float()
    meta = {"seq_id": seq_id, "split": split, "family": family, "source": "synthetic", "source_docs": [],
            "prompt_sha256": hashlib.sha256(seq_id.encode()).hexdigest(), "finish_reason": "stop", "segments": 1,
            "replay_request_id": f"d2r.{seq_id}.{r0}.{L}"}
    return SeqRecord(meta, tokens, gen_start, taps, top_ids, top_lp, tok_lp, tok_rank, resid)
