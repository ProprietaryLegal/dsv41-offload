"""D3a sub-steps: stage 0 of the fast path and of reference.py fed IDENTICAL inputs, one step at a time.

Every EXL3 matrix and the routed experts of the fast path are swapped for exact fp32 math first (parity_ref
make_precise), so any difference left is plumbing (mixes, collapse, norms, RoPE, KV record, attention, wo_a, MoE
routing, post) or rounding order. Usage: python -m worker.tools.parity_substeps ROOT OUT.json
"""

import json
import os
import sys

import torch
from tokenizers import Tokenizer

from ..checkpoint import Checkpoint
from ..model import DSparkDrafter
from ..ops import block_attention, route
from ..reference import RefDrafter
from .parity_ref import _parity_text, make_precise, make_taps, rel


def _fmt(*a):
    return " ".join(str(x) for x in a)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "d3a_substeps.json"
    out = []
    dev = torch.device("cuda", 0)
    ck = Checkpoint(root); m = DSparkDrafter(ck, dev); ref = RefDrafter(ck, dev); make_precise(m, ref)
    ids = Tokenizer.from_file(os.path.join(root, "tokenizer.json")).encode(_parity_text()).ids
    ids = (ids * 3)[:400]; toks = torch.tensor(ids, device=dev)
    taps = make_taps(m, toks, torch.Generator(device=dev).manual_seed(1017))
    lp = 140; W = 128
    rows = torch.arange(lp - W + 1, lp + 1, device=dev)
    m.reset_slots(torch.arange(1, device=dev)); m.insert(torch.zeros_like(rows), rows, m.main_x(taps[rows]))
    slots = torch.arange(1, device=dev); anchors = toks[lp+1:lp+2]; last = torch.tensor([lp], device=dev)
    idsq, pos, win = m.block_inputs(slots, anchors, last, 5, gather=True)
    ctx_kv, valid = win.kv, win.valid
    rp = m.ring_pos[0]; sel = (rp >= lp - W + 1) & (rp <= lp)
    ctx = [m.ring_kv[0, i][sel] for i in range(3)]
    s = m.s; T = 5; st = m.stages[0]; rst = ref.stages[0]
    x = m.embed[idsq.reshape(T)]; stream = x.unsqueeze(1).expand(T, 4, s.hidden).contiguous()
    pf = m._mixes(stream, *st.hc_attn); pr = ref.hc_mixes_(stream, rst["hc_attn_fn"], rst["hc_attn_scale"], rst["hc_attn_base"])
    out.append(_fmt("attn mixes pre/post/comb rel", [rel(a, b) for a, b in zip(pf, pr)]))
    xin_f = m._collapse_norm(stream, None, st.attn_norm)
    pre1 = torch.zeros(T, 4, device=dev); pre1[:, 0] = 1
    xin_r = ref.norm_(ref.hc_pre_(stream, pre1), rst["attn_norm"])
    out.append(_fmt("attn input rel", rel(xin_f, xin_r), "identical", torch.equal(xin_f, xin_r)))
    pos_f = pos.reshape(T)
    a_f = m._attention(st, 0, xin_f, pos_f, win, 1, 5)
    a_r = ref.attention_(rst, xin_f, pos_f, ctx[0])
    out.append(_fmt("attention out rel (same input)", rel(a_f, a_r)))
    # pieces of attention
    qr_f = m._norm(st.wq_a(xin_f), st.q_norm); qr_r = ref.norm_(rst["wq_a"](xin_f).to(torch.bfloat16), rst["q_norm"])
    out.append(_fmt(" qr rel", rel(qr_f, qr_r)))
    q_f = m._rope(st.wq_b(qr_f).to(torch.bfloat16).view(T, 64, 512), pos_f)
    q_r = ref.rope_(rst["wq_b"](qr_f).to(torch.bfloat16).unflatten(-1, (64, 512)), pos_f)
    out.append(_fmt(" q (roped) rel", rel(q_f, q_r), "nope identical", torch.equal(q_f[..., :448], q_r[..., :448]), "rope rel", rel(q_f[..., 448:], q_r[..., 448:])))
    kv_f = m._kv(m._norm(st.wkv(xin_f), st.kv_norm), pos_f); kv_r = ref.kv_quant_(ref.rope_(ref.norm_(rst["wkv"](xin_f).to(torch.bfloat16), rst["kv_norm"]), pos_f))
    out.append(_fmt(" block kv rel", rel(kv_f, kv_r)))
    o_f = block_attention(q_f.view(1, 5, 64, 512), ctx_kv[:, 0], valid, kv_f.view(1, 5, 512), st.attn_sink, m.scale)
    keys = torch.cat([ctx[0], kv_f], 0).float()
    o_r = torch.empty(5, 64, 512, device=dev)
    for t in range(5):
        sc = (q_f[t].float() @ keys.t()) * (512 ** -0.5); sc = torch.cat([sc, rst["attn_sink"].unsqueeze(1)], 1); o_r[t] = sc.softmax(-1)[:, :-1] @ keys
    out.append(_fmt(" attention core rel", rel(o_f.view(5, 64, 512), o_r), "valid ctx rows", int(valid.sum()), "ref ctx rows", ctx[0].shape[0]))
    o2f = m._rope(o_f.view(T, 64, 512), pos_f, inverse=True); o2r = ref.rope_(o_r.to(torch.bfloat16), pos_f, inverse=True)
    out.append(_fmt(" inv-roped o rel", rel(o2f, o2r)))
    z_f = torch.bmm(o2f.view(T, 8, -1).transpose(0, 1), m.wo_a_bf16(0)).transpose(0, 1).reshape(T, -1)
    z_r = torch.einsum("sgd,grd->sgr", o2f.view(T, 8, -1).float(), rst["wo_a"]).to(torch.bfloat16).flatten(1)
    out.append(_fmt(" wo_a out rel", rel(z_f, z_r)))
    s1_f = m._post(a_f, stream, pf[1], pf[2]); s1_r = ref.hc_post_(a_f, stream, pf[1], pf[2])
    out.append(_fmt("post stream rel", rel(s1_f, s1_r)))
    ff = m._mixes(s1_f, *st.hc_ffn); fr = ref.hc_mixes_(s1_f, rst["hc_ffn_fn"], rst["hc_ffn_scale"], rst["hc_ffn_base"])
    out.append(_fmt("ffn mixes rel", [rel(a, b) for a, b in zip(ff, fr)]))
    fin_f = m._collapse_norm(s1_f, pf[0], st.ffn_norm); fin_r = ref.norm_(ref.hc_pre_(s1_f, pf[0]), rst["ffn_norm"])
    out.append(_fmt("ffn input rel", rel(fin_f, fin_r), "identical", torch.equal(fin_f, fin_r)))
    y_f = m._moe(st, fin_f); y_r, idx_r = ref.moe_(rst, fin_f)
    out.append(_fmt("moe out rel (same input)", rel(y_f, y_r)))
    idf, wf = route(fin_f, st.gate_w, st.gate_bias, 3, 1.5); out.append(_fmt(" experts fast", idf.tolist(), "ref", idx_r.tolist()))
    # routed alone vs shared alone
    lim = 10.0
    sh_f = st.sh_w2((torch.nn.functional.silu(st.sh_w1(fin_f).clamp(max=lim)) * st.sh_w3(fin_f).clamp(-lim, lim)).to(torch.bfloat16))
    w1, w3, w2 = rst["sh"]
    sh_r = w2((torch.nn.functional.silu(w1(fin_f).clamp(max=lim)) * w3(fin_f).clamp(-lim, lim)).to(torch.bfloat16))
    out.append(_fmt(" shared expert rel", rel(sh_f, sh_r)))
    s2_f = m._post(y_f, s1_f, ff[1], ff[2]); s2_r = ref.hc_post_(y_r, s1_f, ff[1], ff[2])
    out.append(_fmt("stage0 out stream rel", rel(s2_f, s2_r)))
    out.append(_fmt("magnitudes: stream", float(stream.float().abs().max()), "attn_out", float(a_f.float().abs().max()), "ffn_out", float(y_f.float().abs().max())))
    for line in out:
        print(line)
    with open(out_path, "w") as f:
        json.dump({"steps": out}, f, indent=1)


if __name__ == "__main__":
    main()
