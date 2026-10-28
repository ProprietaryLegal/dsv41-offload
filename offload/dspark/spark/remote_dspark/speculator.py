# SPDX-License-Identifier: Apache-2.0
"""vLLM speculators for the remote DSpark drafter (DSV41-OFFLOAD lane O3).

Built by the overlaid ``init_speculator`` only when VLLM_DSPARK_REMOTE is set.

capacity    RemoteDSparkCapacitySpeculator(BaseSpeculator): no local drafter at all - no
            draft weights, Markov cache, draft CUDA graphs, draft workspace, and no draft
            attention layers (so no draft KV-cache groups). Failed rounds give zero drafts
            with confidence 0 (the "conf" depth policy then verifies nothing), or raise
            (VLLM_DSPARK_REMOTE_ON_FAIL=raise).
resilience  RemoteDSparkLocalSpeculator(DSparkSpeculator): the local drafter stays loaded;
            remote drafts are used, the local draft context KV is updated every round
            (eager precompute of the new rows), and a failed round (or failed requests)
            runs the local drafter. Nothing is freed.
shadow      same class: local propose every round plus the remote round; drafts compared
            and logged (parity); local drafts are used unless
            VLLM_DSPARK_REMOTE_SHADOW_USE=remote.
"""

from __future__ import annotations

import os
import socket
import weakref
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.depth_policy import depth_policy_requested
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from . import wire
from .config import RemoteConfig, apply_cost_overrides, n_out_table
from .round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound, make_remote_round
from .stats import Stats

logger = init_logger(__name__)


# ---------------------------------------------------------------------------- setup


def _validate(vllm_config: VllmConfig) -> None:
    spec = vllm_config.speculative_config
    pc = vllm_config.parallel_config
    hf = spec.draft_model_config.hf_config
    problems = []
    if spec.method != "dspark":
        problems.append(f"method {spec.method!r} (only dspark)")
    if spec.draft_sample_method != "greedy":
        problems.append(f"draft_sample_method {spec.draft_sample_method!r} (greedy only: "
                        "probabilistic drafting needs full draft distributions on the wire)")
    if getattr(hf, "dspark_draft_topk", None) is not None:
        problems.append("dspark_draft_topk is set (needs draft distributions)")
    if not getattr(hf, "sample_from_anchor", True):
        problems.append("sample_from_anchor=False (contract assumes anchor-as-first)")
    if getattr(vllm_config, "watermark_config", None) is not None:
        problems.append("watermarking is configured")
    if spec.enable_adaptive_verification:
        problems.append("enable_adaptive_verification (needs local draft logits/estimator)")
    if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
        problems.append("pipeline or data parallelism")
    if getattr(pc, "prefill_context_parallel_size", 1) != 1 or getattr(
            pc, "decode_context_parallel_size", 1) != 1:
        problems.append("context parallelism")
    if vllm_config.model_config.dtype != torch.bfloat16:
        problems.append(f"model dtype {vllm_config.model_config.dtype} (bf16 taps expected)")
    if problems:
        raise ValueError("remote DSpark (VLLM_DSPARK_REMOTE) cannot run with: "
                         + "; ".join(problems))


def _hello(vllm_config: VllmConfig, cfg: RemoteConfig, tp_rank: int) -> wire.Hello:
    spec = vllm_config.speculative_config
    hf = spec.draft_model_config.hf_config
    hidden = int(hf.hidden_size)
    taps = tuple(int(x) for x in hf.dspark_target_layer_ids)
    window = int(hf.sliding_window)
    max_slots = int(vllm_config.scheduler_config.max_num_seqs)
    return wire.Hello(
        num_spec=int(spec.num_speculative_tokens),
        window=window,
        hidden=hidden,
        n_taps=len(taps),
        tap_layer_ids=taps,
        payload_kind=wire.PAYLOAD_TAPS_BF16,
        row_bytes=len(taps) * hidden * 2,
        noise_token_id=int(hf.dspark_noise_token_id),
        vocab_size=int(vllm_config.model_config.get_vocab_size()),
        max_model_len=int(vllm_config.model_config.max_model_len),
        max_slots=max_slots,
        max_rows_per_step=max_slots * window,
        sample_mode=wire.SAMPLE_GREEDY,
        tp_size=int(vllm_config.parallel_config.tensor_parallel_size),
        drafter_identity=cfg.drafter_identity,
        markov_identity=cfg.markov_identity,
        client_name=f"{socket.gethostname()} tp{tp_rank}",
    )


def _n_out_table(vllm_config: VllmConfig, cfg: RemoteConfig) -> tuple[list[int] | None, str]:
    """ReqEntry.n_out by request count (CONTRACT 6.3): the window cap the model runner hands
    DSparkDepthPolicy (model_runner.py:700-722), when the depth policy masks past it."""
    spec = vllm_config.speculative_config
    num_spec = int(spec.num_speculative_tokens)
    dense = None
    uses_dynamic = getattr(spec, "uses_dynamic_speculative_decoding", None)
    if uses_dynamic is not None and uses_dynamic():
        from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

        dense = build_dynamic_sd_schedule_lookup(
            spec.num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=int(vllm_config.scheduler_config.max_num_seqs),
            vllm_num_speculative_tokens=num_spec,
        )
    try:
        from vllm.v1.worker.gpu.spec_decode.depth_policy import _parse_policy
    except ImportError:
        return None, "depth policy parser not found in this image"
    mode = _parse_policy(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))[0]
    return n_out_table(cfg, num_spec, mode, os.environ.get("VLLM_DSPARK_DEPTH_FILE") or None,
                       dense)


def _make_round(vllm_config: VllmConfig, device: torch.device, cfg: RemoteConfig,
                stats: Stats) -> RemoteRound:
    tp = get_tp_group()
    rank = tp.rank_in_group
    hello = _hello(vllm_config, cfg, rank)
    n_out, n_out_why = _n_out_table(vllm_config, cfg)

    def bcast(t: torch.Tensor) -> None:
        tp.broadcast(t, src=0)

    strict = cfg.strict_start or cfg.on_fail == "raise"
    sync = _make_stream_sync(cfg, device, stats, hello) if rank == 0 else None
    # a process proxy that does not start: synchronous rounds unless strict (make_remote_round)
    rnd = make_remote_round(cfg, hello, device=device, tp_rank=rank,
                            broadcast_fn=bcast if tp.world_size > 1 else None, stats=stats,
                            background=not strict, stream_sync=sync, n_out_by_reqs=n_out)
    if rnd.proxy is not None:
        # the proxy process owns the client; a strict start already connected there (or raised)
        stats.set("n_out_reason", n_out_why)
        rnd.proxy.publish_engine_stats(stats.snapshot())
        logger.info("remote DSpark n_out: %s (%s)", n_out if n_out else "all N", n_out_why)
        logger.info("remote DSpark (%s policy) rank 0 -> %s through proxy process pid %d; rank-0 "
                    "buffers: %.1f MiB device, %.1f MiB shared region", cfg.policy, cfg.endpoint,
                    rnd.proxy.pid, rnd.gatherer.device_bytes / 2**20,
                    rnd.region.layout.size / 2**20)
    elif rnd.client is not None:
        stats.set("n_out_reason", n_out_why)
        logger.info("remote DSpark n_out: %s (%s)", n_out if n_out else "all N", n_out_why)
        if strict:
            if not rnd.client.connect_now():
                raise RuntimeError(
                    f"remote DSpark: connect/HELLO to {cfg.endpoint} failed at startup "
                    f"({rnd.client.disabled_reason or rnd.client.last_error})")
            rnd.client.start_background()
        logger.info("remote DSpark (%s policy) rank 0 -> %s; rank-0 buffers: %.1f MiB device, "
                    "%.1f MiB pinned host", cfg.policy, cfg.endpoint,
                    rnd.gatherer.device_bytes / 2**20, rnd.gatherer.host_bytes / 2**20)
    return rnd


def _make_stream_sync(cfg: RemoteConfig, device: torch.device, stats: Stats, hello: wire.Hello):
    """Async mode on rank 0: a self-tested HostFlagSync, or None (synchronous rounds). With the
    process proxy its page lies in a shared region registered here (shmregion.py)."""
    if not cfg.async_mode:
        return None
    if torch.device(device).type != "cuda":
        logger.warning("remote DSpark async mode needs CUDA; synchronous rounds")
        stats.set("async_fallback_reason", "not cuda")
        return None
    from .streamsync import (HostFlagSync, RegisteredRegion, StreamStuckError, StreamSyncError,
                             release_hand_off)

    sync = None
    region = None
    shared = None
    try:
        if cfg.async_proxy == "process":
            from .shmregion import RegionLayout, SharedRegion

            shared = SharedRegion.create(RegionLayout.for_hello(hello))
            region = RegisteredRegion(shared, torch.device(device))
        sync = HostFlagSync(torch.device(device), lib_path=cfg.hostflag_lib,
                            spin_us=cfg.async_spin_us, nap_us=cfg.async_nap_us, region=region)
        ok, why = sync.self_test()
    except StreamStuckError:
        # the stream is parked for good: falling back would hang the engine later (S9 F4); the region stays
        # mapped and registered (the stream may still touch it)
        raise
    except (StreamSyncError, RuntimeError, OSError, ValueError) as e:
        ok, why = False, str(e)
    if not ok:
        logger.warning("remote DSpark async mode off (host-flag hand-off: %s); synchronous rounds", why)
        stats.set("async_fallback_reason", why)
        # every refusal releases the watchdog, the registration and the region (S11 P-2: ~30 MiB mlocked)
        leak = release_hand_off(sync, region, shared)
        if leak:
            logger.error("remote DSpark: after the refused hand-off the %s", leak)
        return None
    if cfg.switch_interval_us > 0 and cfg.async_proxy == "thread":
        import sys

        before = sys.getswitchinterval()
        sys.setswitchinterval(min(before, cfg.switch_interval_us / 1e6))
        logger.info("remote DSpark async: GIL switch interval %.0f -> %.0f us", before * 1e6,
                    sys.getswitchinterval() * 1e6)
    logger.info("remote DSpark async mode on (host-flag hand-off self-test passed; %s)", sync.describe())
    return sync


def _close_round(rnd: RemoteRound) -> None:
    """The speculator is gone (the model runner's shutdown drops it after a device synchronize) or the interpreter
    exits: stop the round's proxy (process: stats flushed, stop word, bounded wait; thread: joined), fail every
    pending round far ahead and close the worker connection (S11 P-6). A finalizer must not raise."""
    try:
        rnd.close()
    except Exception as e:  # noqa: BLE001 -- the hand-off's own atexit fail-ahead still covers the stream
        logger.warning("remote DSpark: closing the remote round at shutdown failed: %s", e)


def init_remote_dspark_speculator(vllm_config: VllmConfig, device: torch.device):
    cfg = RemoteConfig.from_env()
    _validate(vllm_config)
    from vllm.v1.worker.gpu.spec_decode.depth_policy import _parse_policy

    changed = apply_cost_overrides(
        cfg, depth_policy_on=depth_policy_requested(),
        depth_mode=_parse_policy(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))[0],
        num_spec=int(vllm_config.speculative_config.num_speculative_tokens),
        depth_file=os.environ.get("VLLM_DSPARK_DEPTH_FILE") or None)
    if changed:
        logger.info("remote DSpark: depth-policy costs for the remote round: %s", changed)
    if cfg.policy == "capacity":
        return RemoteDSparkCapacitySpeculator(vllm_config, device, cfg)
    return RemoteDSparkLocalSpeculator(vllm_config, device, cfg)


# ---------------------------------------------------------------------------- capacity


class RemoteDSparkCapacitySpeculator(BaseSpeculator):
    """No local drafter (CONTRACT 10, capacity)."""

    _speculator_name = "RemoteDSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device, cfg: RemoteConfig):
        spec = vllm_config.speculative_config
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = spec
        self.method = spec.method
        self.cfg = cfg
        self.num_speculative_steps = int(spec.num_speculative_tokens)
        # anchor-as-first: N query tokens per request (as DSparkSpeculator)
        self.num_query_per_req = self.num_speculative_steps
        self.num_extra_query_per_req = 0
        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.draft_model_config = spec.draft_model_config
        self.draft_tokens = torch.zeros(self.max_num_reqs, self.num_speculative_steps,
                                        dtype=torch.int64, device=device)
        self.draft_token_confidence_probs = torch.zeros(
            self.max_num_reqs, self.num_speculative_steps, dtype=torch.float32, device=device)
        self.draft_logits = None
        self.supports_mm_inputs = False
        # the worker returns the confidence head's outputs (CONTRACT 6.3)
        self.use_confidence_head = True
        self.enable_adaptive_verification = False
        self.pcp_manager = None
        self.stats = Stats()
        self.remote = _make_round(vllm_config, device, cfg, self.stats)
        self._close = weakref.finalize(self, _close_round, self.remote)

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        return None

    def capture(self) -> None:
        return None

    def prepare_watermarking(self, sampler, idx_mapping) -> None:
        return None

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata: dict[str, Any], slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled, num_rejected,
                last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run: bool = False, skip_attn_for_dummy_run: bool = False,
                mm_inputs=None, is_profile: bool = False) -> torch.Tensor:
        n = input_batch.num_reqs
        if dummy_run or n == 0:
            # profiling / capture / warmup: no network, same on every rank
            self.draft_tokens[:n].zero_()
            self.draft_token_confidence_probs[:n].zero_()
            return self.draft_tokens[:n]
        if not aux_hidden_states:
            raise RuntimeError("remote DSpark needs the target's aux hidden states (taps)")
        res = self.remote.run(input_batch, aux_hidden_states, num_sampled, num_rejected,
                              last_sampled, next_prefill_tokens)
        # failed requests carry zero drafts / zero confidence in the broadcast already
        self.draft_tokens[:n].copy_(res.drafts)
        self.draft_token_confidence_probs[:n].copy_(res.conf)
        if self.cfg.on_fail == "raise" and res.read_status() != STATUS_OK:
            raise RuntimeError(
                "remote DSpark round failed and VLLM_DSPARK_REMOTE_ON_FAIL=raise "
                f"(status {res.read_status()}; see {self.cfg.stats_file} on rank 0)")
        return self.draft_tokens[:n]


# ---------------------------------------------------------------------------- local-backed


class RemoteDSparkLocalSpeculator(DSparkSpeculator):
    """Local drafter retained (resilience / shadow; CONTRACT 10)."""

    _speculator_name = "DSpark+remote"

    def __init__(self, vllm_config: VllmConfig, device: torch.device, cfg: RemoteConfig):
        super().__init__(vllm_config, device)
        self.cfg = cfg
        self.stats = Stats()
        self.remote = _make_round(vllm_config, device, cfg, self.stats)
        self._close = weakref.finalize(self, _close_round, self.remote)
        self._shadow = ShadowCompare(self.num_speculative_steps, self.stats) \
            if cfg.policy == "shadow" else None

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if not self.use_confidence_head:
            logger.warning("remote DSpark: the local drafter has no confidence head in use; "
                           "remote confidences are still written")
        return model

    def _local_context_sync(self, input_batch, aux_hidden_states, num_sampled,
                            num_rejected, last_sampled, next_prefill_tokens, temperature,
                            seeds) -> None:
        """Write this round's context rows into the local draft KV without drafting
        (the first half of DFlashSpeculator.propose, eager), so that a later round can
        fall back to the local drafter with an exact window."""
        num_target_tokens = input_batch.num_tokens
        hidden = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        self.hidden_states[:num_target_tokens].copy_(hidden[:num_target_tokens])
        self.prepare_context_anchor(input_batch, num_rejected)
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.block_tables.cp_rank,
                self.dcp_size,
                self.block_tables.cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )
        self._precompute_context_kv(0, num_target_tokens)

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run: bool = False, skip_attn_for_dummy_run: bool = False,
                mm_inputs=None, is_profile: bool = False) -> torch.Tensor:
        n = input_batch.num_reqs
        args = (input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds)
        kw = dict(dp_sync=dp_sync, dummy_run=dummy_run,
                  skip_attn_for_dummy_run=skip_attn_for_dummy_run, mm_inputs=mm_inputs,
                  is_profile=is_profile)
        if dummy_run or n == 0 or not aux_hidden_states:
            return super().propose(*args, **kw)
        res = self.remote.run(input_batch, aux_hidden_states, num_sampled, num_rejected,
                              last_sampled, next_prefill_tokens)
        status = res.read_status()  # host value on every rank: the branch below must match
        if self.cfg.policy == "resilience":
            if status == STATUS_OK:
                self._local_context_sync(input_batch, aux_hidden_states, num_sampled,
                                         num_rejected, last_sampled, next_prefill_tokens,
                                         temperature, seeds)
                self.draft_tokens[:n].copy_(res.drafts)
                self.draft_token_confidence_probs[:n].copy_(res.conf)
                return self.draft_tokens[:n]
            super().propose(*args, **kw)   # local drafts for every request
            self.stats.inc("local_fallback_rounds")
            if status != STATUS_ROUND_FAILED:
                self._merge_remote(res, n)  # keep remote drafts where they are usable
            return self.draft_tokens[:n]
        # shadow: time the local drafter too (rank 0, CUDA events), so the coordinator gets
        # both costs for the remote-mode depth-policy tables
        timing = self._shadow is not None and self.remote.is_sender and \
            torch.device(self.device).type == "cuda"
        if timing:
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
        super().propose(*args, **kw)
        if timing:
            ev1.record()
        if self._shadow is not None and res.reply is not None:
            self._shadow.compare(self.draft_tokens[:n], self.draft_token_confidence_probs[:n],
                                 res)
        if timing:
            ev1.synchronize()
            self.stats.observe("local_propose_us", ev0.elapsed_time(ev1) * 1000.0)
        if self.cfg.shadow_use == "remote" and status != STATUS_ROUND_FAILED:
            self._merge_remote(res, n)
        return self.draft_tokens[:n]

    def _merge_remote(self, res, n: int) -> None:
        ok = res.ok_mask.to(torch.bool).view(n, 1)
        self.draft_tokens[:n] = torch.where(ok, res.drafts.to(torch.int64),
                                            self.draft_tokens[:n])
        self.draft_token_confidence_probs[:n] = torch.where(
            ok, res.conf, self.draft_token_confidence_probs[:n])


class ShadowCompare:
    """Rank-0 parity statistics: local vs remote drafts (CONTRACT 10, shadow)."""

    def __init__(self, num_spec: int, stats: Stats, keep_examples: int = 32):
        self.N = num_spec
        self.stats = stats
        self.keep = keep_examples
        self.agree_pos = np.zeros(num_spec, dtype=np.int64)
        self.first_mismatch = np.zeros(num_spec + 1, dtype=np.int64)  # index N = none
        self.compared = 0
        self.examples: list[dict] = []
        self.max_conf_diff = 0.0

    def compare(self, local_ids: torch.Tensor, local_conf: torch.Tensor, res) -> None:
        reply = res.reply
        need = res.need_draft
        ids_l = local_ids.to("cpu").numpy()
        conf_l = local_conf.to("cpu").numpy()
        for i in range(ids_l.shape[0]):
            if need is not None and not need[i]:
                continue
            if int(reply.status[i]) not in wire.USABLE_REQ_STATUS:
                continue
            r = reply.draft_ids[i].astype(np.int64)
            eq = ids_l[i] == r
            self.agree_pos += eq
            mism = np.flatnonzero(~eq)
            first = int(mism[0]) if mism.size else self.N
            self.first_mismatch[first] += 1
            self.compared += 1
            d = float(np.abs(conf_l[i] - reply.confidence[i]).max())
            self.max_conf_diff = max(self.max_conf_diff, d)
            if first < self.N and len(self.examples) < self.keep:
                ex = {"local": ids_l[i].tolist(), "remote": r.tolist(),
                      "first_mismatch": first,
                      "window_rows": int(reply.window_rows[i]),
                      "status": int(reply.status[i])}
                if reply.margin is not None:
                    ex["remote_margin"] = [round(float(x), 4) for x in reply.margin[i]]
                self.examples.append(ex)
        self.stats.set("shadow", {
            "compared_requests": self.compared,
            "agree_rate_by_position": [round(a / self.compared, 4) if self.compared else None
                                       for a in self.agree_pos.tolist()],
            "first_mismatch_hist": self.first_mismatch.tolist(),
            "full_block_agree_rate": (round(self.first_mismatch[self.N] / self.compared, 4)
                                      if self.compared else None),
            "max_conf_abs_diff": round(self.max_conf_diff, 5),
            "examples": self.examples,
        })

# Capacity mode does not call DSparkSpeculator.propose. Decode capture must wrap
# these entry points explicitly; ordinary production imports remain unchanged.
if os.environ.get("VLLM_DSPARK_DECODE_CAPTURE_DIR"):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import _d2c_wrap

    _d2c_wrap(RemoteDSparkCapacitySpeculator)
    _d2c_wrap(RemoteDSparkLocalSpeculator)
