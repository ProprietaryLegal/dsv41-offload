"""Runs the REAL DeepseekV4DecoderLayer / DeepseekV4Model code (K1 bytes and the R-SP overlay) on a toy decoder.

The two classes are lifted from model.py by AST and executed against CPU stand-ins for what they call: row-wise torch
references of the mHC kernels, an attention that mixes every earlier row and ends in a row-parallel wo_b, a MoE whose
runner honours ``moe_config.skip_final_all_reduce``, an Engram with column-parallel-style wkv and the post-wkv kernel's
semantics, the image's real DecoderReplayLayers, and a fake TP group (fake_tp.FakeTP). Weights are seeded and shared by
every rank, as TP shards of one model.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import itertools
import sys
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from offload.sp_prefill import build_overlay as bo
from offload.sp_prefill.tests.fake_tp import FakeTP

IMAGE = Path('/opt/dsv41/data/src-image')
DTYPE = torch.bfloat16
HIDDEN, HC, VOCAB, HEADS = 16, 4, 97, 4  # HEADS: per-model attention width split over the ranks
N_HASH_COLS, HEAD_DIM = 3, 8
IMAGE_TOKEN = 90  # ids >= this are image-span tokens (Engram gate shut)


def load_helper() -> Any:
    """The overlay's sp_prefill.py as a module (no vLLM needed)."""
    path = bo.HERE / 'overlay' / bo.HELPER
    name = 'vllm_rsp_sp_prefill_under_test'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lift(source: str, names: set[str], scope: dict[str, Any], filename: str) -> dict[str, Any]:
    parsed = ast.parse(source)
    nodes = [n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names, f'{filename}: {names - {n.name for n in nodes}} missing'
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), scope)
    return scope


# ---------------------------------------------------------------- row-wise stand-ins for the mHC kernels


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rowdot(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """[R, K] x [M, K] -> [R, M] as an elementwise product and a last-dim sum: row-independent by construction."""
    return (x.unsqueeze(1) * w.unsqueeze(0)).sum(dim=-1)


@dataclass
class Rec:
    """Row counts each stand-in saw, per rank."""
    rows: dict[str, list[int]] = field(default_factory=dict)

    def note(self, name: str, n: int) -> None:
        self.rows.setdefault(name, []).append(n)


REC: dict[int, Rec] = {}


def _rank_rec(tp: FakeTP) -> Rec:
    return REC.setdefault(tp.rank, Rec())


def make_mhc(tp: FakeTP) -> dict[str, Any]:
    def mixes(flat: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, eps: float,
              pre_eps: float, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = _rowdot(flat.float(), fn) * _rms(flat.float(), eps)
        post = torch.sigmoid(m[:, :HC] * scale[0] + base[:HC]) * alpha
        pre = torch.sigmoid(m[:, HC:2 * HC] * scale[1] + base[HC:2 * HC]) + pre_eps
        comb = torch.softmax((m[:, 2 * HC:] * scale[2] + base[2 * HC:]).view(-1, HC, HC), dim=-1)
        return post, comb, pre

    def collapse_norm(residual: torch.Tensor, pre: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        y = (residual.float() * pre.unsqueeze(-1)).sum(dim=1)
        return (y * _rms(y, eps) * w.float()).to(DTYPE)

    def post(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, res_mix: torch.Tensor) -> torch.Tensor:
        _rank_rec(tp).note('mhc_post', x.shape[0])
        out = post_mix.unsqueeze(-1) * x.float().unsqueeze(1) + torch.einsum('tij,tjd->tid', res_mix, residual.float())
        return out.to(DTYPE)

    def pre_delayed(residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, rms_eps: float,
                    hc_pre_eps: float, hc_sinkhorn_eps: float, alpha: float, iters: int, x: torch.Tensor | None = None,
                    pre_mix: torch.Tensor | None = None, norm_weight: torch.Tensor | None = None,
                    norm_eps: float = 1e-6) -> tuple[torch.Tensor, ...]:
        _rank_rec(tp).note('mhc_pre', residual.shape[0])
        assert norm_weight is not None
        flat = x if x is not None else residual.flatten(1)
        post_mix, comb, next_pre = mixes(flat, fn, scale, base, rms_eps, hc_pre_eps, alpha)
        if x is not None:
            y = x.float()
            layer_input = (y * _rms(y, norm_eps) * norm_weight.float()).to(DTYPE)
        else:
            layer_input = collapse_norm(residual, pre_mix if pre_mix is not None else next_pre, norm_weight, norm_eps)
        return post_mix, comb, layer_input, next_pre

    def shifted_post_pre(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, res_mix: torch.Tensor,
                         fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
                         hc_sinkhorn_eps: float, alpha: float, iters: int, pre_mix: torch.Tensor | None = None,
                         norm_weight: torch.Tensor | None = None, norm_eps: float = 1e-6, capture_aux: bool = False,
                         stream: object = None, reduce_results: bool = False) -> tuple[torch.Tensor, ...]:
        assert stream is None and not reduce_results
        new_res = post(x, residual, post_mix, res_mix)
        out = pre_delayed(new_res, fn, scale, base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, alpha, iters,
                          pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps)
        aux = new_res.mean(dim=1) if capture_aux else new_res.new_empty(0, new_res.shape[-1])
        return (new_res, *out, aux)

    def collapse(h: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        _rank_rec(tp).note('collapse', h.shape[0])
        return (h.float() * pre_mix.unsqueeze(-1)).sum(dim=1).to(DTYPE)

    return {'mhc_post_tilelang': post, 'mhc_pre_delayed_tilelang': pre_delayed,
            'mhc_shifted_post_pre': shifted_post_pre, 'hc_collapse_triton': collapse}


# ---------------------------------------------------------------- modules


def _w(gen: torch.Generator, *shape: int, scale: float = 0.3) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen) * scale).to(DTYPE)


class RowLinear(nn.Module):
    """RowParallelLinear: this rank's input shard times its weight shard; all-reduce iff reduce_results."""

    def __init__(self, tp: FakeTP, w: torch.Tensor) -> None:
        super().__init__()
        self.tp, self.w, self.reduce_results = tp, w, True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = (z.float() @ self.w.float()).to(DTYPE)
        return self.tp.all_reduce(out) if self.reduce_results else out


class Attn(nn.Module):
    """Mixes every earlier row (causal mean), projects this rank's head shard, ends in wo_b; records KV writes."""

    def __init__(self, tp: FakeTP, gen: torch.Generator, layer: int, kv_log: list[tuple[str, int, torch.Tensor]]) -> None:
        super().__init__()
        self.tp, self.layer, self.kv_log, self.gemm_rs = tp, layer, kv_log, None
        wa, wo = _w(gen, HIDDEN, HEADS), _w(gen, HEADS, HIDDEN)
        self.wa_full, self.wo_full = wa, wo
        self.wo_b = RowLinear(tp, wo)  # sharded on first use (rank known only inside the thread)

    def _shard(self) -> tuple[torch.Tensor, torch.Tensor]:
        n, r = self.tp.world_size, self.tp.rank
        width = HEADS // n
        cols = slice(r * width, (r + 1) * width)
        return self.wa_full[:, cols], self.wo_full[cols]

    def forward(self, positions: torch.Tensor, x: torch.Tensor, _: object) -> torch.Tensor:
        assert x.shape[0] == positions.shape[0], 'attention must see every row'
        self.kv_log.append(('fwd', self.layer, x.clone()))
        wa, wo = self._shard()
        self.wo_b.w = wo
        h = x.float().cumsum(dim=0) / (positions.float().unsqueeze(-1) + 1)
        z = (h @ wa.float()).to(DTYPE)
        out: torch.Tensor = self.wo_b(z)
        return out

    def forward_kv(self, positions: torch.Tensor, x: torch.Tensor) -> None:
        assert x.shape[0] == positions.shape[0]
        self.kv_log.append(('kv', self.layer, x.clone()))


class Runner(nn.Module):
    """FusedMoE runner stand-in: routed (by token id and hidden) + shared, this rank's intermediate shard; final
    all-reduce unless moe_config.skip_final_all_reduce."""

    def __init__(self, tp: FakeTP, gen: torch.Generator) -> None:
        super().__init__()
        self.tp = tp
        self.moe_config = types.SimpleNamespace(skip_final_all_reduce=False, is_sequence_parallel=False)
        self.w1 = _w(gen, 3, HIDDEN, 8)
        self.w2 = _w(gen, 3, 8, HIDDEN)
        self.ws1, self.ws2 = _w(gen, HIDDEN, 8), _w(gen, 8, HIDDEN)
        self.gate = _w(gen, HIDDEN, 3)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor,
                input_ids: torch.Tensor | None) -> torch.Tensor:
        assert input_ids is not None and input_ids.shape[0] == hidden_states.shape[0], 'MoE needs every row'
        n, r = self.tp.world_size, self.tp.rank
        cols = slice(r * 8 // n, (r + 1) * 8 // n)
        x = hidden_states.float()
        logits = x @ self.gate.float()
        expert = (logits.argmax(dim=-1) + input_ids) % 3
        weight = torch.sigmoid(logits.max(dim=-1).values).unsqueeze(-1)
        routed = torch.zeros_like(x)
        for e in range(3):
            sel = expert == e
            h = torch.nn.functional.silu(x[sel] @ self.w1[e][:, cols].float())
            routed[sel] = h @ self.w2[e][cols].float()
        shared = torch.nn.functional.silu(x @ self.ws1[:, cols].float()) @ self.ws2[cols].float()
        out = (routed * weight + shared).to(DTYPE)
        if self.moe_config.skip_final_all_reduce:
            return out
        return self.tp.all_reduce(out)


class MoE(nn.Module):
    """DeepseekV4MoE stand-in: forward == validation + the runner call (the EXL3 fused path takes <= 8 rows)."""

    def __init__(self, tp: FakeTP, gen: torch.Generator) -> None:
        super().__init__()
        self.experts = Runner(tp, gen)
        self.gate = types.SimpleNamespace(tid2eid=torch.zeros(1), bias_vl=None)
        self.use_native_mega_moe = False
        self.use_sequence_parallel = False

    def defers_finalize(self, num_tokens: int) -> bool:
        return False

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None, mega: object = None) -> torch.Tensor:
        assert input_ids is not None
        out: torch.Tensor = self.experts(hidden_states=x, router_logits=x, input_ids=input_ids)
        return out.view(x.shape)


def engram_post_ref(engram: Any, hidden_states: torch.Tensor, kv: torch.Tensor, token_mask: torch.Tensor | None,
                    num_kv_tokens: int) -> torch.Tensor:
    """_fused_engram_post_wkv_kernel in torch: one (token, hc) program per row; key/value zero past num_kv_tokens."""
    num_tokens, hc, dim = hidden_states.shape
    hidden = hidden_states.float()
    full = torch.zeros(num_tokens, (hc + 1) * dim)
    valid = min(num_kv_tokens, num_tokens)
    full[:valid] = kv[:valid].float()
    key = full[:, : hc * dim].view(num_tokens, hc, dim)
    value = full[:, hc * dim :].unsqueeze(1)
    q, k = engram.q_weight.float(), engram.k_weight.float()
    hidden_rms = torch.rsqrt((hidden * hidden).sum(-1) / dim + engram.eps)
    key_rms = torch.rsqrt((key * key).sum(-1) / dim + engram.eps)
    dot = (hidden * q * k * key).sum(-1) * hidden_rms * key_rms * dim ** -0.5
    gate_in = torch.sqrt(torch.clamp(dot.abs(), min=engram.clamp_value))
    gate = torch.sigmoid(torch.where(dot < 0, -gate_in, gate_in))
    if token_mask is not None:
        active = torch.zeros(num_tokens, dtype=torch.bool)
        active[:valid] = token_mask[:valid]
        gate = torch.where(active.unsqueeze(-1), gate, torch.zeros_like(gate))
    return (hidden + gate.unsqueeze(-1) * value).to(DTYPE)


class Engram(nn.Module):
    def __init__(self, gen: torch.Generator, layer_hash_index: int) -> None:
        super().__init__()
        self.use_sequence_parallel = False
        self.layer_hash_index = layer_hash_index
        self.table = _w(gen, 31, N_HASH_COLS, HEAD_DIM)
        self.w_kv = _w(gen, N_HASH_COLS * HEAD_DIM, (HC + 1) * HIDDEN)
        self.q_weight, self.k_weight = _w(gen, HC, HIDDEN, scale=1.0), _w(gen, HC, HIDDEN, scale=1.0)
        self.hc_mult, self.dim, self.eps, self.clamp_value = HC, HIDDEN, 1e-6, 1e-6
        self.prepared: list[int] = []

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        self.prepared.append(hash_ids.shape[0])

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        return self.table[hash_ids % 31, torch.arange(N_HASH_COLS)]

    def wkv(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.w_kv.float()).to(DTYPE)

    def forward(self, hidden_states: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        assert token_mask is None or token_mask.shape == (hash_ids.shape[0],)
        return engram_post_ref(self, hidden_states, kv, token_mask, hash_ids.shape[0])


class Embedding(nn.Module):
    """VocabParallelEmbedding: this rank's vocabulary half, zeros elsewhere, all-reduced."""

    def __init__(self, tp: FakeTP, gen: torch.Generator) -> None:
        super().__init__()
        self.tp, self.w = tp, _w(gen, VOCAB, HIDDEN, scale=1.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        half = -(-VOCAB // self.tp.world_size)
        own = (ids // half) == self.tp.rank
        out = torch.where(own.unsqueeze(-1), self.w[ids], torch.zeros(1, dtype=DTYPE))
        return self.tp.all_reduce(out)


class RMSNorm(nn.Module):
    def __init__(self, gen: torch.Generator) -> None:
        super().__init__()
        self.weight, self.variance_epsilon = _w(gen, HIDDEN, scale=1.0), 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        return (y * _rms(y, self.variance_epsilon) * self.weight.float()).to(DTYPE)


class Hasher:
    use_slot_cache, lookback_depth = True, 2

    def ensure_cache(self) -> bool:
        return True

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor, *rest: object) -> torch.Tensor:
        prev = torch.cat([input_ids[:1] * 0, input_ids[:-1]])
        return torch.stack([(input_ids * (7 + c) + prev * (3 + 2 * c) + positions) for c in range(N_HASH_COLS)],
                           dim=-1).unsqueeze(1).expand(-1, 2, -1).contiguous()


# ---------------------------------------------------------------- model assembly


@dataclass
class Config:
    layers: int = 6
    engram_layers: tuple[int, ...] = (1,)
    replay_start: int | None = 3  # None: no decoder replay
    aux_layers: tuple[int, ...] = (2, 5, 6)
    mtp_buffer: bool = False
    min_rows: int = 0  # the patched model's prefill_sp_min_rows


class Built:
    def __init__(self, source: str, tp: FakeTP, helper: Any) -> None:
        self.tp, self.helper = tp, helper
        self.kv_logs: dict[int, list[tuple[str, int, torch.Tensor]]] = {}
        self.forward_contexts: list[object] = []
        self.current_context: list[object] = []
        self.attn_metadata: dict[str, Any] = {'swa0': types.SimpleNamespace(
            query_start_loc=None, slot_mapping=None, block_table=None, num_decodes=0, num_prefills=1)}
        scope: dict[str, Any] = {
            'torch': torch, 'nn': nn, 'typing': typing, 'partial': partial, 'islice': itertools.islice,
            'in_piecewise_cudagraph': lambda: False, 'MHC_OVERLAP_MAX_TOKENS': 32,
            'mhc_pre_delayed_overlap': None, 'tensor_model_parallel_all_reduce': tp.all_reduce,
            'sp_all_gather': None, 'sp_reduce_scatter': None, 'sp_shard': None, 'sp_padding_mask': None,
            '_memprobe': types.SimpleNamespace(ENABLED=False), '_NULLCTX': contextlib.nullcontext(),
            'MoEOutput': type('MoEOutput', (), {}), 'EagleModelMixin': type('EagleModelMixin', (), {}),
            'get_pp_group': lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
            'is_forward_context_available': lambda: True,
            'get_forward_context': lambda: types.SimpleNamespace(attn_metadata=self.attn_metadata),
            'dbo_current_ubatch_id': lambda: 0, 'image_sentinel_mask': lambda ids: ids >= IMAGE_TOKEN,
            'gather_engram_hashes': lambda h, dp_shared_memory: h, 'get_engram_dp_size': lambda: 1,
            'get_tensor_model_parallel_world_size': lambda: tp.world_size,
            'get_tensor_model_parallel_rank': lambda: tp.rank,
            'envs': types.SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
            'logger': types.SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None),
            **make_mhc(tp),
        }
        if helper is not None:
            scope['_sp'] = helper
        lift(source, {'DeepseekV4DecoderLayer', 'DeepseekV4Model'}, scope, 'model.py')
        self.Layer, self.Model = scope['DeepseekV4DecoderLayer'], scope['DeepseekV4Model']
        replay_scope: dict[str, Any] = {'torch': torch, 'MoEOutput': scope['MoEOutput'],
                                        'override_forward_context': self._override, 'Callable': object,
                                        'ForwardContext': object}
        lift((IMAGE / 'vllm/models/deepseek_v41/decoder_replay_layers.py').read_text(), {'DecoderReplayLayers'},
             replay_scope, 'decoder_replay_layers.py')
        self.Replay = replay_scope['DecoderReplayLayers']

    @contextlib.contextmanager
    def _override(self, context: object) -> Iterator[None]:
        self.current_context.append(context)
        try:
            yield
        finally:
            self.current_context.pop()

    def model(self, cfg: Config, seed: int = 0) -> Any:
        tp = self.tp
        gen = torch.Generator().manual_seed(seed)
        kv_log = self.kv_logs.setdefault(tp.rank, [])
        layers = []
        for i in range(cfg.layers):
            layer = self.Layer.__new__(self.Layer)
            nn.Module.__init__(layer)
            layer.hidden_size, layer.mhc_stream, layer.fuse_mhc_all_reduce = HIDDEN, None, False
            layer.use_sequence_parallel, layer.layer_idx = False, i
            layer.engram = Engram(gen, cfg.engram_layers.index(i)) if i in cfg.engram_layers else None
            layer.rms_norm_eps = 1e-6
            layer.attn = Attn(tp, gen, i, kv_log)
            layer.ffn = MoE(tp, gen)
            layer.attn_norm, layer.ffn_norm = RMSNorm(gen), RMSNorm(gen)
            layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps, layer.hc_post_alpha = HC, 2, 1e-6, 2.0
            mix = (2 + HC) * HC
            layer.hc_attn_fn = (torch.randn(mix, HC * HIDDEN, generator=gen) * 0.1)
            layer.hc_ffn_fn = (torch.randn(mix, HC * HIDDEN, generator=gen) * 0.1)
            layer.hc_attn_fn_broadcast = (torch.randn(mix, HIDDEN, generator=gen) * 0.1) if i == 0 else None
            layer.hc_attn_base, layer.hc_ffn_base = torch.randn(mix, generator=gen), torch.randn(mix, generator=gen)
            layer.hc_attn_scale, layer.hc_ffn_scale = torch.rand(3, generator=gen), torch.rand(3, generator=gen)
            layers.append(layer)
        m = self.Model.__new__(self.Model)
        nn.Module.__init__(m)
        m.layers = nn.ModuleList(layers)
        m.start_layer, m.end_layer = 0, cfg.layers
        m.use_native_mega_moe = m.use_sequence_parallel = m.fuse_mhc_all_reduce = False
        m.hc_mult, m.hc_dim = HC, HC * HIDDEN
        m.embed_tokens = Embedding(tp, gen)
        m.norm = RMSNorm(gen)
        m.engram_hash = Hasher()
        m.engram_swa_prefix = 'swa0'
        m.engram_dp_shared_memory = False
        m.aux_hidden_state_layers = cfg.aux_layers
        m._mtp_hidden_buffer = torch.zeros(256, HC * HIDDEN, dtype=DTYPE) if cfg.mtp_buffer else None
        m.prefill_sp_min_rows = cfg.min_rows
        if cfg.replay_start is None:
            m.decoder_replay_layers, m.decoder_replay_start = None, cfg.layers
        else:
            cut = m.layers[cfg.replay_start]
            m.decoder_replay_start = cfg.replay_start
            m.decoder_replay_layers = self.Replay(4, m._run_replay_layers, row_buffers=[],
                                                  write_batch_kv=cut.write_kv, first_swa_prefix='swa0')
        return m


def batch(lengths: tuple[int, ...], seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (sum(lengths),), generator=gen)
    positions = torch.cat([torch.arange(n) for n in lengths])
    return ids, positions


def replay_rows(lengths: tuple[int, ...], window: int = 4) -> torch.Tensor:
    """Each request's last ``window`` rows (what the model state hands DecoderReplayLayers)."""
    starts = list(itertools.accumulate((0, *lengths[:-1])))
    return torch.cat([torch.arange(s + max(0, n - window), s + n) for s, n in zip(starts, lengths)])
