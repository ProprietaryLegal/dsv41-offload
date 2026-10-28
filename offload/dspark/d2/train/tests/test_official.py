"""Independent forward oracle: pinned, unmodified DeepSeek model.py via V1.

Set D2_V1_ROOT and D2_OFFICIAL to relocate these read-only dependencies.
These are required integration tests, never silently skipped.
"""
from __future__ import annotations

import os
import importlib
import sys
from typing import Any

import torch
import pytest

from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, inputs, tiny_config
from offload.dspark.native.formats import decode_fp8_block


@pytest.mark.parametrize("position", (0, 12))
def test_pinned_official_forward(position: int) -> None:
    sys.path.insert(0, os.environ.get("D2_V1_ROOT", "/opt/dsv41/repos/.worktrees/dsv41-v1-verify/offload/dspark"))
    rk = importlib.import_module("verify.ref_kernels")
    load_official_model = importlib.import_module("verify.official").load_official_model

    torch.set_num_threads(2)
    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    rk.CFG.acc, rk.CFG.online_bf16_p, rk.CFG.cache_weights = torch.float64, False, False
    M = load_official_model(os.environ.get("D2_OFFICIAL", "/opt/dsv41/data/ref-ds41/inference"))
    c = tiny_config()
    src = SyntheticSource(c)
    a = M.ModelArgs(dim=c.dim, moe_inter_dim=c.inter, n_layers=1, n_mtp_layers=c.stages,
                    vocab_size=c.vocab, n_heads=c.heads, head_dim=c.hd, rope_head_dim=c.rd,
                    q_lora_rank=c.qr, o_groups=c.groups, o_lora_rank=c.outr, window_size=c.window,
                    dspark_n_routed_experts=c.experts, dspark_n_activated_experts=c.topk,
                    dspark_block_size=c.block, dspark_noise_token_id=c.noise, dspark_markov_rank=c.markov,
                    dspark_target_layer_ids=(0, 0, 0), compress_ratios=(0, 0, 0, 0), route_scale=c.route_scale,
                    swiglu_limit=c.limit, max_batch_size=2, max_seq_len=80)
    stages = [M.DSparkBlock(1+i, a) for i in range(c.stages)]
    for i, st in enumerate(stages):
        state = {}
        for n, target in st.state_dict().items():
            name = f"mtp.{i}.{n}"
            raw = src.get(name)
            if n == "attn.wo_a.weight":
                raw = decode_fp8_block(raw, src.get(name.replace("weight", "scale")))
            elif target.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e4m3fn):
                raw = raw.view(target.dtype)
            state[n] = raw.to(target.dtype)
        st.load_state_dict(state, assign=True)
        st.attn.wo_a.weight = torch.nn.Parameter(st.attn.wo_a.weight.float(), requires_grad=False)
        for m in st.modules():
            if getattr(m, "scale", None) is not None:
                m.weight.scale = m.scale
    d = Drafter(src, c, acc=torch.float64).eval()
    inp = inputs(c, P=position)
    with torch.no_grad():
        mx = d.conditioning(inp["taps"], inp["engine_mx"], inp["has_taps"])
        main = stages[0].main_norm(stages[0].main_proj(inp["taps"].float()))
        torch.testing.assert_close(mx, main, rtol=0, atol=0)
        ids = torch.full((2, c.block), c.noise, dtype=torch.long)
        ids[:, 0] = inp["anchor"]
        h = torch.nn.functional.embedding(ids, src.get("embed.weight").float()).unsqueeze(2).repeat(1, 1, c.hc, 1)
        pre = M.make_identity_pre_mix(h, c.hc)
        trace: dict[str, Any] = {}
        hd, hx = d.hidden_from(inp, trace=trace)
        for i, st in enumerate(stages):
            for j, pos in enumerate(inp["positions"][0].tolist()):
                kv = st.attn.kv_norm(st.attn.wkv(main[:, j:j+1]))
                M.apply_rotary_emb(kv[..., -c.rd:], st.attn.freqs_cis[pos:pos+1])
                M.act_quant(kv, 32, "ue8m0", M.scale_dtype, True)
                st.attn.window_kv_cache[:2, pos % c.window] = kv[:, 0]
            h, pre = st(h, int(inp["P"][0]), pre, main[:, -1:])
            torch.testing.assert_close(trace[f"s{i}"]["h_out"], h, rtol=2e-6, atol=2e-7)
        ref_x = stages[-1].hc_pre(h, pre)
        ref_h = stages[-1].norm(ref_x)
        torch.testing.assert_close(hx, ref_x, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(hd, ref_h, rtol=2e-6, atol=2e-7)
        prev = torch.tensor([[3, 9, 8, 7, 6], [4, 5, 6, 7, 8]])
        ref_logits = torch.nn.functional.linear(ref_h.float(), src.get("head.weight").float())
        ref_logits += stages[-1].markov_head(prev)[0]
        torch.testing.assert_close(d.logit_chunk(hd, prev, 0, c.vocab), ref_logits, rtol=2e-6, atol=2e-7)
    torch.set_default_dtype(old_default)
