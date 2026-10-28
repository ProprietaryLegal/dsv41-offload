"""The D2 replay overlay on CPU with fakes: image prefix intact, inert without the env var, rows keyed by (sequence,
position) through chunked prefill rounds that mix tagged and untagged requests, in-order hashes, DONE.json, errors
recorded instead of raised. Also py_compile under Python 3.12 (the container's version) when one is available."""

import glob
import hashlib
import inspect
import json
import logging
import os
import subprocess
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.join(HERE, "..", "..", "capture", "speculator_replay.py")
MARK = b"# --- DSV41-OFFLOAD D2 replay overlay"
IMAGE_SHA = "f319e29d7b5492f8504d141474ef0bb58d39971249a24749fa8d464a808e8708"
ROW = 15360


def _section() -> str:
    s = open(OVERLAY, "rb").read()
    return s[s.index(MARK):].decode()


def test_image_prefix_is_byte_identical():
    s = open(OVERLAY, "rb").read()
    img = s[: s.index(MARK)].rstrip(b"\n") + b"\n"
    assert hashlib.sha256(img).hexdigest() == IMAGE_SHA
    with open(os.path.join(HERE, "..", "..", "capture", "speculator.py.image-sha256")) as fh:
        assert fh.read().split()[0] == IMAGE_SHA


def test_py_compile_312(tmp_path):
    cands = sorted(glob.glob(os.path.expanduser("~/.local/share/uv/python/cpython-3.12*/bin/python3.12"))) + \
        sorted(glob.glob(os.path.expanduser("~/dsv41-offload/*/venv/bin/python3.12")))
    if not cands:
        pytest.skip("no Python 3.12 on this host")
    r = subprocess.run([cands[0], "-c", f"import py_compile, sys; py_compile.compile({OVERLAY!r}, doraise=True, "
                                        f"cfile={str(tmp_path / 'o.pyc')!r}); print(sys.version_info[:2])"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "(3, 12)"


class FakeSpec:
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
        return "drafts"


def _load(tmp_path, env: bool):
    ns = {"DSparkSpeculator": type("DSparkSpeculator", (FakeSpec,), {}), "torch": torch,
          "logger": logging.getLogger("fake")}
    old = os.environ.get("VLLM_DSPARK_REPLAY_DIR")
    if env:
        os.environ["VLLM_DSPARK_REPLAY_DIR"] = str(tmp_path)
    else:
        os.environ.pop("VLLM_DSPARK_REPLAY_DIR", None)
    try:
        exec(compile(_section(), "speculator_replay_section", "exec"), ns)
    finally:
        if old is None:
            os.environ.pop("VLLM_DSPARK_REPLAY_DIR", None)
        else:
            os.environ["VLLM_DSPARK_REPLAY_DIR"] = old
    return ns


def _tap(pos: int, layer: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(pos * 7 + layer)
    return torch.randn(5120, generator=g).to(torch.bfloat16)


def _tok(seq: int, pos: int) -> int:
    return (seq * 1000 + pos * 13) % 129000


def _round(reqs):
    """reqs: list of (req_id, seq_index, first_pos, n_rows) -> (InputBatch-like, aux list)."""
    ids, pos, qsl = [], [], [0]
    for _, si, p0, n in reqs:
        ids += [_tok(si, p) for p in range(p0, p0 + n)]
        pos += list(range(p0, p0 + n))
        qsl.append(len(ids))
    pad = 3                                              # padded rows after num_tokens, as the engine has
    aux = [torch.stack([_tap(p, L) for p in pos] + [torch.zeros(5120, dtype=torch.bfloat16)] * pad)
           for L in range(3)]
    ib = types.SimpleNamespace(req_ids=[r[0] for r in reqs], num_reqs=len(reqs), num_tokens=len(ids),
                               query_start_loc_np=np.array(qsl, dtype=np.int32),
                               input_ids=torch.tensor(ids + [0] * pad, dtype=torch.int32),
                               positions=torch.tensor(pos + [0] * pad, dtype=torch.int64))
    return ib, aux


def _call(spec, ib, aux):
    return spec.propose(ib, None, None, None, aux, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                        torch.zeros(1), torch.zeros(1))


def test_inert_without_env(tmp_path):
    ns = _load(tmp_path, env=False)
    assert ns["DSparkSpeculator"].propose is FakeSpec.propose
    assert "_d2r_replay_propose" not in ns


def test_rows_keyed_by_sequence_and_position(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    assert inspect.signature(ns["DSparkSpeculator"].propose) is not None
    # seq A: positions 0..999 prefilled in chunks of 300; rows saved from 600 (= gen_start 728 - W 128) to 1000
    # seq B: positions 0..449, saved from 0; an untagged request rides along in some rounds
    A, B = "cmpl-d2r.pA.600.1000-0-1a2b3c4d", "cmpl-d2r.pB.0.450-0-99887766"
    rounds = [[(A, 1, 0, 300), ("cmpl-other-0-x", 9, 0, 50)],
              [(A, 1, 300, 300), (B, 2, 0, 200)],
              [(A, 1, 600, 300), (B, 2, 200, 250)],
              [(A, 1, 900, 100), ("cmpl-other-0-x", 9, 50, 1)]]
    for r in rounds:
        ib, aux = _round(r)
        assert _call(spec, ib, aux) == "drafts"
    for seq, si, frm, end in (("pA", 1, 600, 1000), ("pB", 2, 0, 450)):
        d = tmp_path / seq
        done = json.load(open(d / "DONE.json"))
        assert done["rows"] == end - frm and done["rows_missing"] == 0 and done["dup_rows"] == 0 and done["in_order"]
        taps = open(d / "taps.bin", "rb").read()
        ids = np.frombuffer(open(d / "ids.bin", "rb").read(), dtype="<i4")
        assert len(taps) == (end - frm) * ROW * 2 and len(ids) == end - frm
        assert hashlib.sha256(taps).hexdigest() == done["sha256_taps"]
        assert hashlib.sha256(ids.tobytes()).hexdigest() == done["sha256_ids"]
        rows = torch.frombuffer(bytearray(taps), dtype=torch.bfloat16).view(end - frm, 3, 5120)
        for p in (frm, frm + 1, (frm + end) // 2, end - 1):
            assert int(ids[p - frm]) == _tok(si, p)
            for L in range(3):
                assert torch.equal(rows[p - frm, L], _tap(p, L))
        led = [json.loads(x) for x in open(d / "ledger.jsonl")]
        assert sum(x["rows"] for x in led) == end - frm
    assert not (tmp_path / "ERRORS.jsonl").exists()
    assert not (tmp_path / "other").exists()


def test_errors_are_recorded_not_raised(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    ib, aux = _round([("cmpl-d2r.pC.0.10-0-aa", 3, 0, 10)])
    ib.positions = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 0, 0, 0])   # a gap: not contiguous
    assert _call(spec, ib, aux) == "drafts"
    err = [json.loads(x) for x in open(tmp_path / "ERRORS.jsonl")]
    assert err and "contiguous" in err[0]["error"] and err[0]["seq"] == "pC"
    assert not (tmp_path / "pC" / "DONE.json").exists()


def test_dummy_and_profile_runs_are_ignored(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    ib, aux = _round([("cmpl-d2r.pD.0.5-0-aa", 4, 0, 5)])
    spec.propose(ib, None, None, None, aux, None, None, None, None, None, None, dummy_run=True)
    spec.propose(ib, None, None, None, aux, None, None, None, None, None, None, is_profile=True)
    assert not (tmp_path / "pD").exists()
