"""Run explicitly under run_locked.sh, never through ordinary CPU pytest."""
from __future__ import annotations

import json
from typing import Callable, cast

import torch

from offload.dspark.d2.train.ops import frozen_linear
from offload.dspark.d2.train.safety import assert_gpu_lease
from offload.dspark.native.formats import act_qdq, decode_fp4


def main() -> None:
    assert_gpu_lease()
    g = torch.Generator(device="cuda").manual_seed(17)
    rows = []
    for M, N, K in ((1, 32, 64), (5, 64, 128), (3, 128, 512), (1, 2304, 5120), (8, 5120, 2304)):
        w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, generator=g, device="cuda")
        s = torch.randint(111, 128, (N, K//32), dtype=torch.uint8, generator=g, device="cuda")
        x = act_qdq(torch.randn(M, K, generator=g, device="cuda"))
        x.requires_grad_(True)
        got = frozen_linear(x, w, s, torch.float64)
        decoded = decode_fp4(w, s)
        expected = (x.double() @ decoded.double().T).float()
        assert torch.equal(got, expected), f"packed FP64 sum differs at {(M, N, K)}"
        dy = torch.randn(M, N, generator=g, device="cuda")
        cast(Callable[..., None], got.backward)(dy)
        dx = dy @ decoded
        assert x.grad is not None
        assert torch.equal(x.grad, dx), f"packed backward differs at {(M, N, K)}"
        rows.append({"M": M, "N": N, "K": K, "forward_exact": True, "dx_exact_fp32": True})
    print(json.dumps({"pass": True, "cases": rows}), flush=True)


if __name__ == "__main__":
    main()
