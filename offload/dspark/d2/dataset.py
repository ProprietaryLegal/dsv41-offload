"""D2 dataset builder (x86host): joins token ids (generation state / acap export) + top-20 (replay logprobs) + taps (replay
overlay rows copied from the Spark) into d2.shard.v1 shards (DATASET.md), with every integrity check of section 8.

  build   python3 -m d2.dataset build --replay /opt/dsv41/x86host-data/d2/replay --out /opt/dsv41/x86host-data/d2/shards \
              --prompt-manifest /opt/dsv41/x86host-data/d2/prompts/manifest.json --equiv-gate <equiv gate.json>
          (refuses prefill_replay taps without a passed equivalence gate, and publishes only after G1 plumbing --
          dataset verify + V1 CaptureSet load -- passed on the new shards; overrides are explicit and recorded)
  verify  python3 -m d2.dataset verify --out /opt/dsv41/x86host-data/d2/shards      (sha256 + every record re-checked)
  synth   python3 -m d2.dataset synth --out <dir> [--scale 1]   (schema-valid synthetic shards for D2-TRAIN's tests)

Proprietary output: shards stay on x86host (mode 700 dirs, 600 files), never committed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import shards as S
from .split_audit import audit_plan
from .sequence_audit import sequence_audit, validate_context
from .equiv import GATE_SCHEMA
from .replay import ROW_BYTES, load_seq, read_plan

OVERLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "capture", "speculator_replay.py")


def sha256_path(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def private_dir(path: str) -> str:
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def build_record(e: dict, replay: str, pman: dict, context: dict | None = None) -> S.SeqRecord:
    """One plan entry -> SeqRecord, after DATASET.md section 8 rules 1-4 (rule 5 runs in check_record)."""
    if e.get("request_id", "").startswith("d2c."):
        from .decode_dataset import build_record as build_decode_record
        return build_decode_record(e, replay, pman, context)
    sid = e["seq_id"]
    s = load_seq(e["src"])
    toks = s["tokens"]
    if S.tokens_sha256(toks) != e["tokens_sha256"] or len(toks) != e["L"] or s["gen_start"] != e["gen_start"]:
        raise ValueError(f"{sid}: token ids / L / gen_start differ from the replay plan (rule 1)")
    L, g, frm = e["L"], e["gen_start"], e["from"]
    r0 = S.r0_of(g)
    if frm > r0:
        raise ValueError(f"{sid}: replay saved rows from {frm}, the shard needs them from {r0}")
    with open(os.path.join(replay, "logprobs", sid + ".json")) as fh:
        lm = json.load(fh)
    if (lm["tokens_sha256"], lm["L"], lm["gen_start"]) != (e["tokens_sha256"], L, g):
        raise ValueError(f"{sid}: logprobs metadata disagrees with the plan")
    z = np.load(os.path.join(replay, "logprobs", sid + ".npz"))
    td = os.path.join(replay, "taps", sid)
    with open(os.path.join(td, "DONE.json")) as fh:
        done = json.load(fh)
    rows = L - frm
    if (done["from"], done["end"], done["rows"], done["rows_missing"]) != (frm, L, rows, 0):
        raise ValueError(f"{sid}: DONE.json {done} is not a complete {frm}..{L} replay (rule 3)")
    if done.get("zero_rows") != 0:
        raise ValueError(f"{sid}: DONE.json zero_rows {done.get('zero_rows')}: zero taps rows (rule 3)")
    tp, ip = os.path.join(td, "taps.bin"), os.path.join(td, "ids.bin")
    if os.path.getsize(tp) != rows * ROW_BYTES or os.path.getsize(ip) != rows * 4:
        raise ValueError(f"{sid}: taps.bin / ids.bin sizes do not match {rows} rows")
    if sha256_path(tp) != done["sha256_taps"] or sha256_path(ip) != done["sha256_ids"]:
        raise ValueError(f"{sid}: taps.bin / ids.bin sha256 differ from DONE.json (rule 6)")
    ids = np.fromfile(ip, dtype="<i4")
    if ids.tolist() != toks[frm:L]:
        raise ValueError(f"{sid}: the overlay's row ids differ from the sequence (rule 2)")
    raw = np.fromfile(tp, dtype="<i2", offset=(r0 - frm) * ROW_BYTES)
    taps = torch.from_numpy(raw).view(torch.bfloat16).view(L - r0, S.TAPW)
    pm = pman.get(e["prompt_sha256"], {})
    meta = {"seq_id": sid, "split": e["split"], "family": e["family"],
            "source": pm.get("source", "a_capture" if e["kind"] == "acap" else None),
            "source_docs": pm.get("source_docs", []), "prompt_sha256": e["prompt_sha256"],
            "finish_reason": e["finish_reason"], "segments": e["segments"], "replay_request_id": e["request_id"],
            "replay_utc": lm.get("utc"), "ties_at_edge": lm.get("ties_at_edge"),
            "dup_rows": done.get("dup_rows", 0), "rows_in_order": done.get("in_order"),
            "duplicate_replay": bool(done.get("duplicate_replay"))}
    if context is not None:
        with open(e["src"]) as fh:
            raw_state = json.load(fh)
        meta["sequence_audit"] = sequence_audit(toks, g, context, raw_state, e["kind"])
    else:
        meta["sequence_audit"] = {"schema": "d2.sequence-audit.synthetic.v1", "not_served_data": True}
    return S.SeqRecord(meta, torch.tensor(toks, dtype=torch.int32), g, taps,
                       torch.from_numpy(z["top_ids"]), torch.from_numpy(z["top_lp"]), torch.from_numpy(z["tok_lp"]),
                       torch.from_numpy(z["tok_rank"]), torch.from_numpy(z["resid"]))


class SplitWriter:
    """Streams one split's records into shards of <= MAX_SHARD_TAP_BYTES of taps (records are dropped once written)."""

    def __init__(self, out: str, split: str, taps_source: str, extra: dict):
        self.d = private_dir(os.path.join(out, split))
        self.split, self.taps_source, self.extra = split, taps_source, extra
        self.files: list[dict] = []
        self.cur: list[S.SeqRecord] = []
        self.cur_bytes = 0
        self.families: collections.Counter = collections.Counter()

    def add(self, r: S.SeqRecord) -> None:
        b = (r.L - r.r0) * S.TAPW * 2
        if self.cur and self.cur_bytes + b > S.MAX_SHARD_TAP_BYTES:
            self.flush()
        self.cur.append(r)
        self.cur_bytes += b
        self.families[r.meta["family"]] += 1

    def flush(self) -> None:
        if self.cur:
            base = os.path.join(self.d, f"{self.split}-{len(self.files):05d}")
            self.files.append(S.write_shard(base, self.split, self.cur, self.taps_source, self.extra))
            self.cur, self.cur_bytes = [], 0

    def summary(self) -> dict:
        self.flush()
        return {"n_seq": sum(f["n_seq"] for f in self.files), "n_span": sum(f["n_span"] for f in self.files),
                "n_rows": sum(f["n_rows"] for f in self.files), "families": dict(self.families), "files": self.files}


def finish_manifest(out: str, man: dict) -> None:
    files = []
    for sp in man["splits"].values():
        for f in sp["files"]:
            for k in ("file", "sidecar"):
                if os.path.isabs(f[k]):
                    f[k] = os.path.relpath(f[k], out)
            files += [(f["sha256"], f["file"]), (f["sidecar_sha256"], f["sidecar"])]
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    os.chmod(os.path.join(out, "MANIFEST.json"), 0o600)
    files.append((sha256_path(os.path.join(out, "MANIFEST.json")), "MANIFEST.json"))
    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        for h, f in files:
            fh.write(f"{h}  {f}\n")
    os.chmod(os.path.join(out, "SHA256SUMS"), 0o600)


def cmd_build(a) -> None:
    """Streams the plan split by split into <out>.partial, renamed to <out> only when every sequence passed (or with
    --allow-skips, which lists the failures in MANIFEST.json)."""
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; build into a new directory")
    held = tree_taps_source(a.replay)
    if a.taps_source != held:
        raise SystemExit(f"--taps-source {a.taps_source} but {a.replay} holds {held} taps (S16 F10-b); refused: the "
                         "label must match the tree, and replay taps never skip the equivalence gate")
    gate = require_gate(a)
    tmp = private_dir(a.out + ".partial")
    if os.listdir(tmp):
        raise SystemExit(f"{tmp} is not empty (an interrupted build?): inspect and remove it first")
    if not a.prompt_manifest:
        raise SystemExit("--prompt-manifest is required: exact IDs and disjoint source families must be checked")
    with open(a.prompt_manifest) as fh:
        prompt_manifest = json.load(fh)
    plan = read_plan(a.replay)
    try:
        split_audit = audit_plan(plan, prompt_manifest)
        for entry in plan:
            source = load_seq(entry["src"])
            if not source or any(source[k] != entry[k] for k in
                                 ("seq_id", "split", "family", "prompt_sha256", "kind")):
                raise ValueError("source state identity differs from replay plan")
    except ValueError as err:
        raise SystemExit(f"split integrity refused: {err}") from err
    pman = {p["prompt_sha256"]: p for p in prompt_manifest["prompts"]}
    context_path = getattr(a, "sequence_context", "")
    synthetic = (prompt_manifest.get("schema") == "d2.prompts.synthetic.v1"
                 and all(p.get("source") == "synthetic" for p in prompt_manifest["prompts"]) and bool(pman))
    contexts = {}
    context_paths = None
    if held == 'decode_capture':
        from .decode_dataset import assembly_paths, decode_identity
        context_paths = assembly_paths(decode_identity(Path(a.replay)))
    if not synthetic:
        if held == "decode_capture" and context_path:
            from .post_replay import context_ready
            from .decode_dataset import assembly_paths, decode_identity
            try:
                context_ready(Path(context_path), Path(a.prompt_manifest),
                              Path(a.replay) / "plan.jsonl",
                              paths=assembly_paths(decode_identity(Path(a.replay))), mode="decode_capture")
            except (ValueError, OSError) as err:
                raise SystemExit(f"sequence provenance refused: {err}") from err
        if not context_path:
            raise SystemExit("--sequence-context is required for served data: target/encoder/quantization receipt")
        if held == "decode_capture":
            from .decode_dataset import decode_identity, assembly_paths
            if assembly_paths(decode_identity(Path(a.replay))) is None:
                raise SystemExit("served decode assembly requires --assembly-root and COPY_RECEIPT.json")
        with open(context_path) as fh:
            context = json.load(fh)
        try:
            for entry in plan:
                ctx = context.get("per_sequence", {}).get(entry["seq_id"], context)
                if entry["kind"] == "acap" and entry["seq_id"] not in context.get("per_sequence", {}):
                    raise ValueError("acap requires its own captured sampling/format context receipt")
                validate_context(ctx, prompt_manifest, require_chat_match=entry["kind"] != "acap",
                                 paths=context_paths, mode=held)
                contexts[entry["seq_id"]] = ctx
        except (ValueError, OSError) as err:
            raise SystemExit(f"sequence provenance refused: {err}") from err
    with open(OVERLAY, "rb") as fh:
        overlay_sha = hashlib.sha256(fh.read()).hexdigest()
    extra = {"replay_overlay_sha256": overlay_sha}
    want = set(a.splits.split(",")) if a.splits else None
    writers: dict[str, SplitWriter] = {}
    errors: list[str] = []
    n = 0
    for e in plan:
        if want and e["split"] not in want:
            continue
        try:
            rec = build_record(e, a.replay, pman, contexts.get(e["seq_id"]))
        except (ValueError, FileNotFoundError) as err:
            errors.append(f"{e['seq_id']}: {err}")
            continue
        errs = S.check_record(rec)
        if errs:
            errors.extend(errs)
            continue
        if e["split"] not in writers:
            writers[e["split"]] = SplitWriter(tmp, e["split"], a.taps_source, extra)
        writers[e["split"]].add(rec)
        n += 1
        if n % 25 == 0:
            print(f"{time.time() - t0:.0f} s: {n} sequences written or buffered, {len(errors)} failed", flush=True)
    if errors and not a.allow_skips:
        raise SystemExit(f"{len(errors)} sequences failed the integrity rules; {tmp} left for inspection, no dataset "
                         f"published. First: {errors[:5]}")
    man = {"schema": S.MANIFEST_SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "taps_source": a.taps_source,
           "prompt_manifest_sha256": sha256_path(a.prompt_manifest) if a.prompt_manifest else None,
           "generation": {"settings_source": "per-sequence sequence_audit and historical segment metadata",
                          "sequence_context_sha256": sha256_path(context_path) if context_path else None,
                          "synthetic": synthetic},
           "replay": {"config": "offload/launch/configs/D-decode-capture.env" if held == "decode_capture" else
                                "offload/launch/configs/A-replay.env", "overlay_sha256": overlay_sha,
                      "plan_sha256": sha256_path(os.path.join(a.replay, "plan.jsonl"))},
           "splits": {sp: w.summary() for sp, w in sorted(writers.items())},
           "checks": {"sequences": n, "skipped_sequences": len(errors), "skipped_first": errors[:20],
                      "equivalence_gate": gate, "split_audit": split_audit}}
    if held == "decode_capture":
        from .decode_dataset import decode_identity
        decode = decode_identity(Path(a.replay))
        man['replay']['decode_sha256'] = sha256_path(os.path.join(a.replay, 'DECODE.json'))
        man['replay']['copy_receipt_sha256'] = decode.get('copy_receipt_sha256')
        man['checks']['named_error_quarantine'] = decode.get('quarantine')
    finish_manifest(tmp, man)
    ver, plb = verify_dir(tmp), plumbing(tmp)
    plb_errors = sum(v["plumbing_errors"] for v in plb.values())
    override_plumbing = getattr(a, "override_plumbing", None)
    if (ver["errors"] or plb_errors) and not override_plumbing:
        raise SystemExit(f"G1 plumbing failed on {tmp} (verify {ver['errors']} errors, CaptureSet {plb_errors}); not "
                         f"published. First: {ver['first'][:3]} {[v['first'] for v in plb.values() if v['first']][:3]}")
    man["checks"]["g1_plumbing"] = {"verify": ver, "capture_set": plb, "override": override_plumbing or None}
    finish_manifest(tmp, man)                     # MANIFEST + SHA256SUMS again, now with the plumbing result
    os.rename(tmp, a.out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in man["splits"].items()},
                     indent=1))
    print(f"dataset {a.out}: {n} sequences in {time.time() - t0:.0f} s", flush=True)


def verify_dir(out: str) -> dict:
    """Every sha256, the manifest schema, every shard's structure and every record (DATASET.md section 8)."""
    bad = []
    with open(os.path.join(out, "SHA256SUMS")) as fh:
        for line in fh:
            h, f = line.split()
            if sha256_path(os.path.join(out, f)) != h:
                bad.append(f"sha256 {f}")
    with open(os.path.join(out, "MANIFEST.json")) as fh:
        man = json.load(fh)
    if man.get("schema") != S.MANIFEST_SCHEMA:
        bad.append(f"manifest schema {man.get('schema')}")
    n = rows = 0
    for split, sp in man["splits"].items():
        for f in sp["files"]:
            sh = S.Shard(os.path.join(out, f["file"]), verify_sha256=False)
            if sh.split != split:
                bad.append(f"{f['file']}: split {sh.split}")
            bad += [f"{f['file']}: {e}" for e in sh.structure_errors()]
            for i in range(len(sh)):
                bad += [f"{f['file']}: {e}" for e in S.check_record(sh.record(i))]
                n += 1
            rows += int(sh.t["row_pos"].shape[0])
    return {"sequences": n, "rows": rows, "errors": len(bad), "first": bad[:10]}


def plumbing(out: str, stride: int = 16) -> dict:
    """G1(a): every split loads through the V1 CaptureSet adapter with 0 plumbing errors (blocks thinned by stride)."""
    with open(os.path.join(out, "MANIFEST.json")) as fh:
        man = json.load(fh)
    rep = {}
    for split, sp in man["splits"].items():
        cs = S.ShardCaptureSet([os.path.join(out, f["file"]) for f in sp["files"]], stride=stride)
        rep[split] = {"sequences": len(cs.rows), "samples": len(cs.samples), "plumbing_errors": len(cs.plumbing),
                      "first": cs.plumbing[:5]}
    return rep


def cmd_verify(a) -> None:
    rep = verify_dir(a.out)
    print(json.dumps(rep, indent=1))
    if rep["errors"]:
        raise SystemExit(1)


def tree_taps_source(replay: str) -> str:
    """Read tap mode from plan IDs AND overlay receipts, never the CLI label.

    Replay d2r.* trees require equivalence; d2c.* trees require DECODE.json and
    per-sequence mode headers before the full ledger/row verification.
    """
    ids = [e.get("request_id", "") for e in read_plan(replay)]
    if not ids:
        raise SystemExit(f"{replay}: empty plan")
    if all(i.startswith("d2c.") for i in ids):
        from pathlib import Path
        from .decode_dataset import decode_identity, obj, SCHEMA
        meta = decode_identity(Path(replay), full_copy=True)
        for e in read_plan(replay):
            head = obj(Path(meta["captures"]) / e["seq_id"] / "CAPTURE.json")
            if head.get("schema") != SCHEMA or head.get("taps_source") != "decode_capture":
                raise SystemExit("decode capture requires the overlay's mode receipt")
        return "decode_capture"
    rep = sum(1 for i in ids if i.startswith("d2r."))
    if rep == len(ids):
        return "prefill_replay"
    raise SystemExit(f"{replay}: cannot tell which taps this tree holds ({rep} of {len(ids)} plan entries are replay "
                     "requests d2r.*); refused")


GATE_VERDICTS = ("pass", "pass_bf16_noise", "fail")


def check_gate_file(g: dict, replay: str) -> list[str]:
    """S16 F10-a: the gate file must be a d2.equiv report (schema, keys, a verdict consistent with pass, a compared
    population) of THIS replay: its replay_binding's sha256s equal the hash fields of <replay>/taps/<seq>/DONE.json,
    and every acap sequence of the replay plan is bound. Returns the problems (empty = valid)."""
    bad = []
    gate = g.get("gate")
    if g.get("schema") != GATE_SCHEMA:
        bad.append(f"schema {g.get('schema')!r}, want {GATE_SCHEMA!r} (a d2.equiv report)")
    if not isinstance(gate, dict):
        return bad + ["no gate object"]
    for k in ("population", "verdict", "pass", "threshold", "rule2_bf16_noise"):
        if k not in gate:
            bad.append(f"gate.{k} missing")
    if gate.get("verdict") not in GATE_VERDICTS:
        bad.append(f"gate.verdict {gate.get('verdict')!r} not in {GATE_VERDICTS}")
    elif gate.get("pass") is not (gate["verdict"] != "fail"):
        bad.append(f"gate.pass {gate.get('pass')!r} contradicts verdict {gate['verdict']!r}")
    nd = g.get("blocks_inputs_differ")
    if not isinstance(nd, int) or nd <= 0:
        bad.append(f"blocks_inputs_differ {nd!r}: nothing was compared")
    binding = g.get("replay_binding")
    if not isinstance(binding, dict) or not binding:
        return bad + ["no replay_binding: not tied to a replay"]
    for seq, h in sorted(binding.items()):
        p = os.path.join(replay, "taps", seq, "DONE.json")
        if not os.path.exists(p):
            bad.append(f"bound sequence {seq} has no {p} in this replay")
            continue
        with open(p) as fh:
            done = json.load(fh)
        for k in ("sha256_taps", "sha256_ids"):
            if not isinstance(h, dict) or done.get(k) != h.get(k):
                bad.append(f"{seq}: {k} of this replay differs from the gate's (another replay's gate?)")
    acap = sorted(e["seq_id"] for e in read_plan(replay) if e.get("kind") == "acap")
    unbound = [s for s in acap if s not in binding]
    if unbound:
        bad.append(f"acap sequences of this replay not in the gate's binding: {unbound}")
    return bad


def require_gate(a) -> dict:
    """S14 F10: prefill_replay taps only after the equivalence gate passed (DATASET.md section 3; rule 9), or under an
    explicit, recorded override. A caller that names neither (an older namespace without the fields) is refused.
    S16 F10-a: the gate file must be a d2.equiv report of this replay (check_gate_file); the override covers a fail
    verdict only, never a malformed or foreign gate file."""
    if a.taps_source != "prefill_replay":
        return {"required": False, "taps_source": a.taps_source}
    gate_file = getattr(a, "equiv_gate", None)
    override = getattr(a, "override_equiv_gate", None)
    if gate_file:
        with open(gate_file) as fh:
            g = json.load(fh)
        bad = check_gate_file(g, a.replay)
        if bad:
            raise SystemExit(f"{gate_file} is not a valid equivalence report of {a.replay}: {bad}; refused (an "
                             "override does not cover this; without a gate use --override-equiv-gate alone)")
        gate = g["gate"]
        if gate["pass"] is not True:
            if not override:
                raise SystemExit(f"equivalence gate {gate_file} did not pass (verdict {gate.get('verdict')}); "
                                 "prefill_replay taps are not valid training data without it")
        return {"required": True, "file": gate_file, "sha256": sha256_path(gate_file),
                "verdict": gate.get("verdict"), "pass": gate.get("pass"),
                "block_equal_frac": gate.get("block_equal_frac"), "replay_binding": g["replay_binding"],
                "override": override or None}
    if not override:
        raise SystemExit("--equiv-gate <gate.json> (d2.equiv report) is required for --taps-source prefill_replay; "
                         "--override-equiv-gate '<reason>' builds without it and records the reason")
    return {"required": True, "file": None, "verdict": None, "pass": None, "override": override}


def cmd_synth(a) -> None:
    """Schema-valid random shards in every split (sizes scaled by --scale) + MANIFEST + SHA256SUMS."""
    out = private_dir(a.out)
    plan = {"train": (6, "holding_summary"), "dev": (2, "issue_spotting"), "heldout_legal": (3, "chronology"),
            "heldout_general": (2, "general_book"), "acap": (1, "acap")}
    man = {"schema": S.MANIFEST_SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "taps_source": "synthetic", "splits": {}, "checks": {"synthetic": True}}
    k = 0
    for split, (n, fam) in plan.items():
        recs = []
        for _ in range(int(n * a.scale)):
            g = 100 + (k * 977) % 3000
            recs.append(S.synth_record(f"syn{k:04d}", split, g + 40 + (k * 131) % 400, g, seed=k, family=fam))
            k += 1
        w = SplitWriter(out, split, "synthetic", {"replay_overlay_sha256": None})
        for r in recs:
            w.add(r)
        man["splits"][split] = w.summary()
    finish_manifest(out, man)
    print(f"synthetic dataset: {k} sequences in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--replay", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--prompt-manifest", required=True)
    b.add_argument("--sequence-context", default="", help="identity/sampling/format receipt; required for served data")
    b.add_argument("--taps-source", "--taps", default="prefill_replay", choices=S.TAPS_SOURCES)
    b.add_argument("--splits", default="")
    b.add_argument("--allow-skips", action="store_true", help="write the clean sequences even if some failed "
                   "(the failures are listed in MANIFEST.json checks)")
    b.add_argument("--equiv-gate", default="", help="gate.json of `d2.equiv report`; required for prefill_replay")
    b.add_argument("--override-equiv-gate", default="", metavar="REASON",
                   help="build prefill_replay shards without a passed gate; the reason is recorded in MANIFEST.json")
    b.add_argument("--override-plumbing", default="", metavar="REASON",
                   help="publish despite G1 plumbing errors; the reason is recorded in MANIFEST.json")
    v = sp.add_parser("verify")
    v.add_argument("--out", required=True)
    s = sp.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    {"build": cmd_build, "verify": cmd_verify, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
