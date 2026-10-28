# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-ins for the vLLM modules remote_dspark.speculator imports, so the
speculator classes can be exercised on Controller without vLLM. Only what is touched."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from types import SimpleNamespace

import torch

CALLS: dict[str, int] = {}


def _count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


class BaseSpeculator:
    num_query_per_req = 1
    num_extra_query_per_req = 0


class _FakeDraftModel:
    def combine_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        _count("combine_hidden_states")
        return x[:, :4]


class DSparkSpeculator(BaseSpeculator):
    """Records what the remote layer calls on the local drafter."""

    def __init__(self, vllm_config, device):
        spec = vllm_config.speculative_config
        self.vllm_config = vllm_config
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self.num_query_per_req = self.num_speculative_steps
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = 4096
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_tokens = torch.zeros(self.max_num_reqs, self.num_speculative_steps,
                                        dtype=torch.int64)
        self.draft_token_confidence_probs = torch.zeros(self.max_num_reqs,
                                                        self.num_speculative_steps)
        self.use_confidence_head = True
        self.draft_logits = None
        self.model = _FakeDraftModel()
        self.hidden_states = torch.zeros(4096, 4, dtype=torch.bfloat16)
        self.draft_kv_cache_group_ids = [0]
        self.block_tables = SimpleNamespace(slot_mappings=[None], input_block_tables=[None],
                                            kernel_block_sizes=[64], cp_rank=0,
                                            cp_interleave=1)
        self.input_buffers = self.context_positions = None
        self._context_slot_mappings = [None]
        self.sample_indices = self.sample_pos = self.sample_idx_mapping = None
        self.temperature = self.seeds = None
        self.dcp_size = 1
        self.parallel_drafting_token_id = 128799
        self.sample_from_anchor = True
        self.next_local: torch.Tensor | None = None   # drafts the "local drafter" returns
        self.next_local_conf: torch.Tensor | None = None

    def load_draft_model(self, target_model, names):
        return self.model

    def prepare_context_anchor(self, input_batch, num_rejected):
        _count("prepare_context_anchor")

    def _precompute_context_kv(self, start, end, dummy_run=False):
        _count("precompute_context_kv")
        self.last_precompute = (start, end)

    def propose(self, input_batch, *args, **kwargs):
        _count("local_propose")
        n = input_batch.num_reqs
        if self.next_local is not None:
            self.draft_tokens[:n] = self.next_local[:n]
            self.draft_token_confidence_probs[:n] = self.next_local_conf[:n]
        else:
            self.draft_tokens[:n] = 7
            self.draft_token_confidence_probs[:n] = 0.5
        return self.draft_tokens[:n]


def prepare_dflash_inputs(*args, **kwargs):
    _count("prepare_dflash_inputs")


class TP:
    def __init__(self, rank=0, world=1):
        self.rank_in_group, self.world_size = rank, world
        self.sent: list[torch.Tensor] = []

    def broadcast(self, t, src=0):
        self.sent.append(t.clone())
        return t


TP_GROUP = TP()
DEPTH_POLICY_ON = {"v": False}


def _parse_policy(text):
    """The image's depth_policy._parse_policy (modes only)."""
    p = (text or "").strip().lower()
    p = p.split()[0] if p else ""
    if p in ("", "0", "off", "none"):
        return "off", 0
    for mode in ("confb", "confr", "conf"):
        if p == mode or p.startswith(mode + ":"):
            return mode, 0
    if p.startswith("fixed:"):
        return "fixed", int(p.split(":", 1)[1])
    raise ValueError(text)


def build_dynamic_sd_schedule_lookup(schedule, vllm_max_batch_size, vllm_num_speculative_tokens):
    """The image's dense lookup (index = batch size; gaps carry the previous K)."""
    dense = [0] * (vllm_max_batch_size + 1)
    nxt, last = 1, None
    for a, b, k in sorted(tuple(r) for r in schedule):
        for bs in range(nxt, min(a, vllm_max_batch_size + 1)):
            dense[bs] = min(vllm_num_speculative_tokens, last if last is not None else k)
        for bs in range(a, min(b, vllm_max_batch_size) + 1):
            dense[bs] = min(vllm_num_speculative_tokens, k)
        nxt, last = b + 1, k
    for bs in range(nxt, vllm_max_batch_size + 1):
        dense[bs] = min(vllm_num_speculative_tokens, last)
    return dense


def install() -> None:
    """Register the stub modules (idempotent) and reload remote_dspark.speculator."""
    def mod(name, **attrs):
        m = sys.modules.get(name) or types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("vllm")
    mod("vllm.config", VllmConfig=object)
    mod("vllm.distributed", get_tp_group=lambda: TP_GROUP)
    mod("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    for pkg in ("vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                "vllm.v1.worker.gpu.spec_decode", "vllm.v1.worker.gpu.spec_decode.dflash",
                "vllm.v1.worker.gpu.spec_decode.dspark"):
        mod(pkg)
    mod("vllm.v1.worker.gpu.spec_decode.depth_policy",
        depth_policy_requested=lambda: DEPTH_POLICY_ON["v"], _parse_policy=_parse_policy)
    mod("vllm.v1.spec_decode")
    mod("vllm.v1.spec_decode.dynamic")
    mod("vllm.v1.spec_decode.dynamic.utils",
        build_dynamic_sd_schedule_lookup=build_dynamic_sd_schedule_lookup)
    mod("vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        prepare_dflash_inputs=prepare_dflash_inputs)
    mod("vllm.v1.worker.gpu.spec_decode.dspark.speculator", DSparkSpeculator=DSparkSpeculator)
    mod("vllm.v1.worker.gpu.spec_decode.speculator", BaseSpeculator=BaseSpeculator)
    if "remote_dspark.speculator" in sys.modules:
        importlib.reload(sys.modules["remote_dspark.speculator"])


def vllm_config(*, hidden=4, max_seqs=4, method="dspark", draft_sample="greedy", schedule=None,
                **over):
    hf = SimpleNamespace(hidden_size=hidden, dspark_target_layer_ids=[37, 38, 39],
                         sliding_window=128, dspark_noise_token_id=128799)
    spec = SimpleNamespace(method=method, draft_sample_method=draft_sample,
                           enable_adaptive_verification=False, num_speculative_tokens=5,
                           draft_model_config=SimpleNamespace(hf_config=hf),
                           num_speculative_tokens_per_batch_size=schedule)
    spec.uses_dynamic_speculative_decoding = (
        lambda: spec.num_speculative_tokens_per_batch_size is not None)
    cfg = SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1,
                                        prefill_context_parallel_size=1,
                                        decode_context_parallel_size=1,
                                        tensor_parallel_size=2),
        model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=327680,
                                     get_vocab_size=lambda: 129280),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
        watermark_config=None,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg
