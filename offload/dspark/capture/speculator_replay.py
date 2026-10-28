# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.depth_policy import depth_policy_requested
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False
        # The model reads cached Markov bias rows by token id when it has them.
        self._markov_by_ids: bool = False
        # Greedy drafts reduced per vocabulary shard (VLLM_DSPARK_LOCAL_ARGMAX); set in
        # load_draft_model once the model has its local bias rows.
        self._local_argmax: bool = False

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        # The per-round depth policy (depth_policy.py) reads the confidences too.
        self.use_confidence_head = (
            self.enable_adaptive_verification or depth_policy_requested()
        ) and model.model.confidence_head is not None
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        self._markov_by_ids = hasattr(model, "markov_bias_for_ids")
        # The local argmax never materializes full draft logits: greedy drafting only, and
        # no acceptance estimator (it reads the full logits; the confidence head replaces it).
        self._local_argmax = (
            getattr(model, "supports_local_argmax", lambda: False)()
            and self.draft_logits is None
            and not self.use_acceptance_estimator
            and self._draft_topk is None
        )
        if self._local_argmax:
            logger.info("DSpark: greedy drafts argmax per vocabulary shard (local argmax)")
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampler = (
            gumbel_sample
            if self.draft_watermarker is None
            else self.draft_watermarker.sample
        )
        return sampler(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        if self._local_argmax:
            # this rank's vocabulary shard of the base logits, never gathered
            base_logits = self.model.compute_draft_logits_local(sample_hidden)
        else:
            # Draft-vocab logits; sampled ids are remapped to target vocab below.
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            if self._local_argmax:
                draft_sampled_i = self.model.draft_argmax_local(
                    base_logits[:, i], prev, markov_embed
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            if self._markov_by_ids:
                bias = self.model.markov_bias_for_ids(prev, markov_embed)
            else:
                bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)


# --- DSV41-OFFLOAD D2 replay overlay (inert unless VLLM_DSPARK_REPLAY_DIR is set) ----------------------------------------
# Bind-mounted over the image's spec_decode/dspark/speculator.py for the D2 replay window only (configs/A-replay.env).
# Everything above this line is the image file byte for byte (sha256 f319e29d..., the same image part as the capture
# overlay; d2/tests/test_replay_overlay.py checks it). On rank 0, for every request whose id carries the tag
# d2r.<seq>.<from>.<end> (lane D2-DATA's replay.py sets request_id), it saves the drafter taps (aux_hidden_states of
# target layers 37/38/39, BF16) and the input token id of every row whose position p satisfies from <= p < end:
#   <dir>/<seq>/taps.bin   row p at byte offset (p - from) * 30720: 37 | 38 | 39, 5120 BF16 each
#   <dir>/<seq>/ids.bin    int32 little-endian token id of row p at (p - from) * 4
#   <dir>/<seq>/ledger.jsonl  one line per round that wrote rows (round, first position, rows, batch tokens)
#   <dir>/<seq>/DONE.json  when row end-1 has been written: rows, in-order sha256 of both files, duplicates, gaps,
#                          zero_rows (rows with any layer's 5120 taps all zero: a trimmed eager-prefill row, see below)
#   <dir>/ERRORS.jsonl     anything that went wrong (the save path never raises into the engine; replay.py and
#                          dataset.py refuse a sequence without a clean DONE.json); kind "already_complete" = a
#                          sequence replayed again after its DONE.json: the first replay is kept, nothing is rewritten
# Prefill rows only: replay requests carry prompt_logprobs (no prefix-cache reads) and max_tokens 1. Eager prefill steps
# run layers >= 20 only on each request's last min(q, 128) rows and leave the others' taps ZERO (patch 0034, CONTRACT.md
# "tap validity"); model_state.py keeps every row of a request with prompt_logprobs, so replay rows are all computed. Any
# zero row is still counted (DONE.json zero_rows, ERRORS.jsonl kind zero_taps once per sequence), and replay.py /
# dataset.py refuse a sequence with zero_rows != 0. Files are written
# with pwrite, fdatasync'ed and dropped from the page cache (POSIX_FADV_DONTNEED) every VLLM_DSPARK_REPLAY_SYNC_MB
# (default 256): GB10 CUDA cannot reclaim page cache. Created files get the owner of <dir> (the host user).
import hashlib as _d2r_hashlib
import inspect as _d2r_inspect
import json as _d2r_json
import os as _d2r_os
import re as _d2r_re
import time as _d2r_time

_D2R_DIR = _d2r_os.environ.get("VLLM_DSPARK_REPLAY_DIR")
if _D2R_DIR:
    _D2R_TAG = _d2r_re.compile(r"d2r\.([A-Za-z0-9_-]+)\.(\d+)\.(\d+)")
    _D2R_ROW_BYTES = 3 * 5120 * 2
    _D2R_SUB = 512                                    # rows copied to host per step (bounds host memory per copy)
    _D2R_SYNC = int(_d2r_os.environ.get("VLLM_DSPARK_REPLAY_SYNC_MB", "256")) * 1024 * 1024
    _d2r_st = _d2r_os.stat(_D2R_DIR)
    _D2R_UID, _D2R_GID = _d2r_st.st_uid, _d2r_st.st_gid
    if _D2R_UID == 0:
        logger.error("DSV41-OFFLOAD D2 replay: %s is root-owned (docker made the mount point); saved files will be "
                     "unreadable to the host user. mkdir it as the user before start.", _D2R_DIR)
    _d2r_orig_propose = DSparkSpeculator.propose
    _d2r_sig = _d2r_inspect.signature(_d2r_orig_propose)
    _d2r_seqs: dict = {}
    _d2r_round = {"n": 0}

    _d2r_dupes: set = set()

    def _d2r_err(msg: str, kind: str = "error", **kw) -> None:
        try:
            p = _d2r_os.path.join(_D2R_DIR, "ERRORS.jsonl")
            with open(p, "a") as fh:
                fh.write(_d2r_json.dumps({"t": _d2r_time.time(), "kind": kind, "error": msg, **kw}) + "\n")
            _d2r_os.chown(p, _D2R_UID, _D2R_GID)
        except OSError:
            pass
        logger.error("DSV41-OFFLOAD D2 replay: %s %s", msg, kw)

    def _d2r_owned(path: str) -> None:
        _d2r_os.chown(path, _D2R_UID, _D2R_GID)

    def _d2r_open(seq: str, frm: int, end: int) -> dict:
        st = _d2r_seqs.get(seq)
        if st is not None:
            if (st["from"], st["end"]) != (frm, end):
                raise RuntimeError(f"tag range {frm}..{end} differs from the open range {st['from']}..{st['end']}")
            return st
        d = _d2r_os.path.join(_D2R_DIR, seq)
        if _d2r_os.path.exists(_d2r_os.path.join(d, "DONE.json")):
            raise RuntimeError("sequence already complete in this directory (replayed twice?)")
        _d2r_os.makedirs(d, mode=0o700, exist_ok=True)
        _d2r_owned(d)
        flags = _d2r_os.O_WRONLY | _d2r_os.O_CREAT | _d2r_os.O_TRUNC
        st = {"from": frm, "end": end, "dir": d, "next": frm, "written": 0, "dup": 0, "gaps": 0, "since_sync": 0,
              "h_taps": _d2r_hashlib.sha256(), "h_ids": _d2r_hashlib.sha256(), "in_order": True,
              "seen": bytearray(end - frm), "zero_rows": 0, "first_zero": None, "first_round": _d2r_round["n"], "t0": _d2r_time.time(),
              "fd_taps": _d2r_os.open(_d2r_os.path.join(d, "taps.bin"), flags, 0o600),
              "fd_ids": _d2r_os.open(_d2r_os.path.join(d, "ids.bin"), flags, 0o600),
              "ledger": open(_d2r_os.path.join(d, "ledger.jsonl"), "w")}
        for name in ("taps.bin", "ids.bin", "ledger.jsonl"):
            _d2r_owned(_d2r_os.path.join(d, name))
        _d2r_seqs[seq] = st
        return st

    def _d2r_sync(st: dict) -> None:
        for fd in (st["fd_taps"], st["fd_ids"]):
            _d2r_os.fdatasync(fd)
            _d2r_os.posix_fadvise(fd, 0, 0, _d2r_os.POSIX_FADV_DONTNEED)
        st["since_sync"] = 0

    def _d2r_close(seq: str, st: dict) -> None:
        _d2r_sync(st)
        _d2r_os.close(st["fd_taps"])
        _d2r_os.close(st["fd_ids"])
        st["ledger"].close()
        n = st["end"] - st["from"]
        missing = n - sum(st["seen"])
        done = {"seq": seq, "from": st["from"], "end": st["end"], "rows": n, "rows_missing": missing,
                "rows_written": st["written"], "dup_rows": st["dup"], "gaps": st["gaps"], "in_order": st["in_order"],
                "zero_rows": st["zero_rows"], "first_zero_row": st["first_zero"],
                "sha256_taps": st["h_taps"].hexdigest() if st["in_order"] else None,
                "sha256_ids": st["h_ids"].hexdigest() if st["in_order"] else None,
                "row_bytes": _D2R_ROW_BYTES, "first_round": st["first_round"], "last_round": _d2r_round["n"],
                "seconds": round(_d2r_time.time() - st["t0"], 3)}
        p = _d2r_os.path.join(st["dir"], "DONE.json")
        with open(p + ".tmp", "w") as fh:
            _d2r_json.dump(done, fh)
        _d2r_os.replace(p + ".tmp", p)
        _d2r_owned(p)
        del _d2r_seqs[seq]

    def _d2r_save(self, a: dict) -> None:
        ib = a["input_batch"]
        aux = a["aux_hidden_states"]
        nr, nt = ib.num_reqs, ib.num_tokens
        qsl = ib.query_start_loc_np
        for b in range(nr):
            m = _D2R_TAG.search(ib.req_ids[b])
            if m is None:
                continue
            seq, frm, end = m.group(1), int(m.group(2)), int(m.group(3))
            if seq not in _d2r_seqs and _d2r_os.path.exists(_d2r_os.path.join(_D2R_DIR, seq, "DONE.json")):
                # replayed twice (S14 F11): the first complete replay stays; recorded once, never blocks the copy
                if seq not in _d2r_dupes:
                    _d2r_dupes.add(seq)
                    _d2r_err("sequence already complete; the first replay is kept", kind="already_complete",
                             seq=seq, round=_d2r_round["n"])
                continue
            try:
                if not aux:
                    raise RuntimeError("no aux_hidden_states in this round")
                st = _d2r_open(seq, frm, end)
                s0, s1 = int(qsl[b]), min(int(qsl[b + 1]), nt)
                pos = ib.positions[s0:s1].to("cpu", torch.int64)
                if pos.numel() == 0:
                    continue
                p0 = int(pos[0])
                if not torch.equal(pos, torch.arange(p0, p0 + pos.numel(), dtype=torch.int64)):
                    raise RuntimeError(f"round rows of the request are not contiguous positions from {p0}")
                lo, hi = max(p0, frm), min(p0 + pos.numel(), end)
                if lo >= hi:
                    continue
                r0 = s0 + (lo - p0)
                ids = ib.input_ids[r0: r0 + (hi - lo)].to("cpu", torch.int32).contiguous()
                for j in range(0, hi - lo, _D2R_SUB):
                    k = min(_D2R_SUB, hi - lo - j)
                    rows = torch.cat([t[r0 + j: r0 + j + k].to("cpu") for t in aux], dim=-1).contiguous()
                    if rows.dtype != torch.bfloat16 or rows.shape[-1] * 2 != _D2R_ROW_BYTES:
                        raise RuntimeError(f"taps rows {tuple(rows.shape)} {rows.dtype}, want [*, 15360] bfloat16")
                    zr = (rows.view(k, 3, 5120) == 0).all(-1).any(-1)
                    if bool(zr.any()):
                        if st["zero_rows"] == 0:
                            st["first_zero"] = lo + j + int(zr.nonzero()[0])
                            _d2r_err("taps row with a layer all zero (a trimmed eager-prefill row?)", kind="zero_taps",
                                     seq=seq, position=st["first_zero"], round=_d2r_round["n"])
                        st["zero_rows"] += int(zr.sum())
                    tb = rows.view(torch.int16).numpy().tobytes()
                    ib_ = ids[j: j + k].numpy().astype("<i4").tobytes()
                    p = lo + j
                    _d2r_os.pwrite(st["fd_taps"], tb, (p - frm) * _D2R_ROW_BYTES)
                    _d2r_os.pwrite(st["fd_ids"], ib_, (p - frm) * 4)
                    seen = st["seen"]
                    for q in range(p - frm, p - frm + k):
                        if seen[q]:
                            st["dup"] += 1
                        seen[q] = 1
                    if p == st["next"] and st["in_order"]:
                        st["h_taps"].update(tb)
                        st["h_ids"].update(ib_)
                        st["next"] = p + k
                    else:
                        st["in_order"] = False
                        st["gaps"] += 1
                    st["written"] += k
                    st["since_sync"] += len(tb)
                st["ledger"].write(_d2r_json.dumps({"round": _d2r_round["n"], "first": lo, "rows": hi - lo,
                                                    "batch_tokens": nt, "batch_reqs": nr}) + "\n")
                st["ledger"].flush()
                if st["since_sync"] >= _D2R_SYNC:
                    _d2r_sync(st)
                if hi == end:
                    _d2r_close(seq, st)
            except Exception as e:  # noqa: BLE001 -- recorded in ERRORS.jsonl; never raised into the engine
                _d2r_err(f"{type(e).__name__}: {e}", seq=seq, round=_d2r_round["n"])

    def _d2r_replay_propose(self, *args, **kwargs):
        out = _d2r_orig_propose(self, *args, **kwargs)
        b = _d2r_sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        if a["dummy_run"] or a["is_profile"]:
            return out
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return out
        _d2r_save(self, a)
        _d2r_round["n"] += 1
        return out

    DSparkSpeculator.propose = _d2r_replay_propose
    logger.info("DSV41-OFFLOAD: D2 replay taps on, rank 0, requests tagged d2r.<seq>.<from>.<end> into %s", _D2R_DIR)

# --- D2 decode capture: also installed on remote capacity/shadow entry points.
# No import, thread, allocation or class mutation when the variable is unset.
if _d2r_os.environ.get("VLLM_DSPARK_DECODE_CAPTURE_DIR"):
    if _D2R_DIR:
        raise RuntimeError("D2 replay and decode capture modes are mutually exclusive")
    from .decode_capture import Capture as _D2Capture, wrapper as _d2c_wrapper

    _d2c_capture = _D2Capture(_d2r_os.environ["VLLM_DSPARK_DECODE_CAPTURE_DIR"])

    def _d2c_wrap(cls):
        cls.propose = _d2c_wrapper(_d2c_capture, cls.propose)

    _d2c_wrap(DSparkSpeculator)
    logger.info("DSV41-OFFLOAD: D2 decode capture on, tagged requests only")
