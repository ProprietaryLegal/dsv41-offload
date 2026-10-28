"""One-step gradient audit; aggregate parameter metadata only, never tensors."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict, cast, Callable

import torch

from .data import Block, D2Blocks, SyntheticBlocks
from .loss import distillation_loss
from .model import Drafter, load
from .ops import Dense
from .perf import Targets
from .pilot import write_json
from .synthetic import SyntheticSource, tiny_config


class GradientAudit(TypedDict):
    schema: str
    loss: float
    nonzero_gradient_parameters: list[str]
    zero_gradient_parameters: list[str]
    missing_gradient_parameters: list[str]
    excluded_parameters: list[str]
    lora_nonzero_A_zero_B: list[str]
    steps: int


def one_step(model: Drafter, block: Block, chunk: int = 8192) -> GradientAudit:
    initialized = []
    for name, module in model.named_modules():
        if isinstance(module, Dense) and module.rank:
            if not bool(module.a.detach().count_nonzero()) or bool(module.b.detach().count_nonzero()):
                raise ValueError(f"zero-init LoRA requires nonzero A and zero B: {name}")
            initialized.append(name)
    if not initialized:
        raise ValueError("no LoRA factors to audit")
    model.train()
    excluded = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("confidence_head" not in name)
        if not parameter.requires_grad:
            excluded.append(name)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    optimizer.zero_grad(set_to_none=True)
    device = model.activation_device
    inp = {k: t.to(device) for k, t in block["inputs"].items()}
    targets = cast(Targets, {k: t.to(device) for k, t in block["targets"].items()})
    h, _ = model.hidden_from(inp)
    loss, _ = distillation_loss(model, h, chunk=chunk, **targets)
    cast(Callable[[], None], loss.backward)()
    nonzero, zero, missing = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"nonfinite gradient: {name}")
        elif bool(parameter.grad.count_nonzero()):
            nonzero.append(name)
        else:
            zero.append(name)
    if not any(name.endswith(".b") for name in nonzero):
        raise ValueError("zero-init LoRA has no nonzero B gradient")
    optimizer.step()
    return {"schema": "dsv41.d2.gradient-audit.v1", "steps": 1, "loss": float(loss.detach()),
            "nonzero_gradient_parameters": nonzero, "zero_gradient_parameters": zero,
            "missing_gradient_parameters": missing, "excluded_parameters": excluded,
            "lora_nonzero_A_zero_B": initialized}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ckpt")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    torch.manual_seed(23)
    torch.set_num_threads(2)
    if a.synthetic:
        config = tiny_config()
        model = Drafter(SyntheticSource(config), config, checkpoint_stages=False)
        block = SyntheticBlocks(config, 1)[0]
    else:
        from .safety import assert_gpu_lease
        assert_gpu_lease()
        if not a.ckpt or not a.manifest:
            parser.error("real audit needs --ckpt and --manifest")
        model = load(a.ckpt, "cuda", reference_math=True)
        block = D2Blocks(a.manifest, "train")[0]
    write_json(a.out, one_step(model, block))


if __name__ == "__main__":
    main()
