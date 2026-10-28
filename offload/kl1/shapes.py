"""Bytes a kernel must move, ESTIMATED from legal-v2a's real shapes and EXL3 bit widths (TP2, one rank).

Sources (all read-only, on Controller):
  * shapes: offload/spark_leads/metadata/official-1017.json text_config (hidden 5120, moe_intermediate 2304, 384 routed
    experts, top 6, 1 shared, 40 layers, q_lora 1280, 64 heads x 512, o_groups 8, o_lora 1024, vocab 129280);
  * TP cuts: src-image vllm_exl3_moe/dense.py (wq_a+wkv replicated, wq_b / wo_a column-parallel, wo_b row-parallel,
    shared expert like a routed expert) and smoe.py (each rank owns half of every expert's intermediate dim);
  * bit widths: the startup log line `EXL3 MoE plan /srv/dsv41/legal-v2a/exl3_moe_plan.json: 40 layers, 2.750 bpw,
    groups {(2.0, 2.0): 6095, (3.0, 3.0): 7075, (4.0, 4.0): 2125, (5.0, 5.0): 65}` (UP2-screen-300K-20261025T131508Z/
    0186-sparkhead.log; the plan file is not on Controller), the kernels' own template K (mrow / reconstruct K5, LM head K6:
    `lm_head: EXL3 K6 (5120 x 64640 per rank)` in the same log) and smoe.cu's "~11 MB" shared-expert stream (= K5).
Formula: an EXL3 in x out matrix at K bits is in*out*K/8 trellis bytes (16x16 tiles of 32*K bytes) plus fp16 suh [in]
and svh [out]. Activations, run tables and partial sums are ignored (kilobytes against megabytes, mostly L2).
"""

from __future__ import annotations

from dataclasses import dataclass

H = 5120
I_EXPERT = 2304
TP = 2
I_LOCAL = I_EXPERT // TP            # 1152 intermediate columns of every expert on each rank
E = 384
TOP_K = 6
LAYERS = 40
Q_LORA = 1280
HEADS = 64
HEAD_DIM = 512
O_GROUPS = 8
O_LORA = 1024
VOCAB_LOCAL = 129280 // TP          # 64640 LM-head rows per rank
KV_DIM = 512                        # wkv out (one 512-wide latent)
MHC_W = (24, 20480)                 # fp32 hc mix weight read by sm120_tf32_hc_prenorm (l2pf.py: 24 x 20480)
SHARED_K = 5
SWIGLU_LIMIT = 10.0                 # text_config swiglu_limit (smoe act_limit)
ROUTED_SCALE = 1.5                  # text_config routed_scaling_factor
DENSE_K = 5
HEAD_K = 6
# legal-v2a expert K histogram (k_gu == k_d for every expert), 40 x 384 experts
K_MIX: dict[int, int] = {2: 6095, 3: 7075, 4: 2125, 5: 65}
GB10_SPEC_GBPS = 273.0             # UPSTREAM CLAIM: NVIDIA's DGX Spark memory bandwidth
# UPSTREAM CLAIM (image source comments, the kernel author's microbenchmarks, not campaign records):
SOURCE_READ_GBPS = (230.0, 240.0)   # exl3_smoe.cu:28 / exl3_mrow.cu:32 block-contiguous k-row reads
SOURCE_DECODE_GBPS = (225.0, 242.0)  # tf_experts.py:107 fused expert decode incl. trellis decode, 1-16 rows


def exl3_bytes(n_in: int, n_out: int, k: float) -> float:
    if n_in % 16 or n_out % 16 or k <= 0:
        raise ValueError(f'EXL3 shape {n_in}x{n_out} K{k}')
    return n_in * n_out * k / 8 + 2 * (n_in + n_out)


def mean_k() -> float:
    total = sum(K_MIX.values())
    if total != LAYERS * E:
        raise ValueError('K_MIX does not cover 40 x 384 experts')
    return sum(k * n for k, n in K_MIX.items()) / total


def expert_gu_bytes(k: float) -> float:
    """Gate + up of one expert on one rank (smoe_gu streams both, [w1 | w3])."""
    return 2 * exl3_bytes(H, I_LOCAL, k)


def expert_down_bytes(k: float) -> float:
    return exl3_bytes(I_LOCAL, H, k)


def expert_bytes(k: float) -> float:
    return expert_gu_bytes(k) + expert_down_bytes(k)


def unique_uniform(picks_rows: float, experts: int = E, top_k: int = TOP_K) -> float:
    """Expected distinct experts when `picks_rows` rows each pick top_k of `experts` independently and uniformly.
    An upper bound for real routing at a fixed row count when consecutive tokens share experts (no anti-correlation)."""
    if picks_rows < 0:
        raise ValueError('rows < 0')
    return float(experts * (1 - (1 - top_k / experts) ** picks_rows))


def shared_runs(live_rows: int, rx: int = 8) -> int:
    """smoe runs of the shared expert: every live row takes it, RX = 8 rows a run (exl3_smoe.cu RX)."""
    return -(-live_rows // rx) if live_rows > 0 else 0


@dataclass(frozen=True)
class Linear:
    name: str
    n_in: int
    n_out: int
    k: float
    count: int = 1                 # matrices of this shape in the module (wo_a: 4 head groups a rank)

    @property
    def bytes(self) -> float:
        return self.count * exl3_bytes(self.n_in, self.n_out, self.k)


# The four exl3_mrow launches of a decode layer, in forward order (attention.py:364-396, dense.py TP cuts).
DECODE_DENSE: tuple[Linear, ...] = (
    Linear('wq_a+wkv', H, Q_LORA + KV_DIM, DENSE_K),                      # replicated (in 5120, out 1280 + 512)
    Linear('wq_b', Q_LORA, HEADS * HEAD_DIM // TP, DENSE_K),               # column-parallel
    Linear('wo_a', HEADS * HEAD_DIM // O_GROUPS, O_LORA, DENSE_K, O_GROUPS // TP),   # 4 head groups a rank
    Linear('wo_b', O_GROUPS * O_LORA // TP, H, DENSE_K),                   # row-parallel
)
LM_HEAD = Linear('lm_head', H, VOCAB_LOCAL, HEAD_K)
ROUTER_BYTES = E * H * 2            # bf16 gate weight, replicated
MHC_BYTES = MHC_W[0] * MHC_W[1] * 4


def reconstruct_shape(grid: tuple[int, int, int]) -> tuple[int, int]:
    """reconstruct_had_kernel's grid is (out/128, in/128): (128, 10) = wq_b 1280 -> 16384, (40, 32) = wo_b 4096 -> 5120,
    (8, 32) = one wo_a group, (9, 40) / (40, 9) = shared gate|up / down, (10, 40) = wq_a, (4, 40) = wkv."""
    return grid[1] * 128, grid[0] * 128


def reconstruct_bytes(grid: tuple[int, int, int], k: int) -> float:
    """Read the K-bit trellis and scales, write the fp16 matrix (the GEMM after it reads that matrix again)."""
    n_in, n_out = reconstruct_shape(grid)
    return exl3_bytes(n_in, n_out, k) + 2 * n_in * n_out


def gbps(nbytes: float, us: float) -> float:
    if us <= 0:
        raise ValueError('non-positive duration')
    return nbytes / us / 1e3
