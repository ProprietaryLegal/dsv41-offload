"""Fixed fast/reference gates. Private captures receive no optimizer updates.

Reference and fast arms run in separate processes to fit a single 3090. Both
consume identical materialized inputs, initialization, and synthetic targets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch import Tensor

from .loss import distillation_loss
from .model import load
from .perf import synthetic_targets
from .data import sha256
from .replay import capture_inputs, load_capture_set, v1_import
from .safety import assert_gpu_lease
from .fast_policy import BACKEND, tensor_groups


def prepare(a: argparse.Namespace) -> None:
    a.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows: list[dict[str, Any]] = []
    for p in sorted(a.fixtures.glob("n1-*.pt")):
        fixture = torch.load(p, map_location="cpu", weights_only=False)
        rows.append({"key": p.stem, "population": "n1", "inputs": fixture["inputs"]})
    if len(rows) != 4:
        raise ValueError("require four N1 windows")
    v1_import(a.v1_root)
    cs = load_capture_set(a.captures)
    by_key = {s.key: s for s in cs.samples}
    parity = json.loads((a.reference_dumps / "parity.json").read_text())
    if parity["pass"] is not True or parity["samples"] != 50:
        raise ValueError("require the qualified 50-round reference population")
    for row in parity["rows"]:
        key = row["key"]
        inp, _ = capture_inputs(cs, by_key[key], "cpu")
        rows.append({"key": key, "population": "captures", "inputs": inp})
    for row in rows:
        torch.save(row, a.out / (row["key"] + ".pt"))
    manifest = {"schema": "dsv41.d2.fast-inputs.v1", "samples": len(rows),
                "files": {p.name: sha256(p) for p in sorted(a.out.glob("*.pt"))},
                "reference_population_sha256": sha256(a.reference_dumps / "parity.json")}
    (a.out / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


def arm(a: argparse.Namespace) -> None:
    assert_gpu_lease()
    torch.set_num_threads(8)
    torch.manual_seed(23)
    model = load(a.ckpt, "cuda", rank=a.rank, reference_math=a.command == "reference")
    if a.command == "fast":
        model.enable_fast_training()
    model.train()
    manifest = json.loads((a.inputs / "inputs.json").read_text())
    a.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    for population in ("n1", "captures"):
        model.zero_grad(set_to_none=True)
        members = []
        for name, digest in manifest["files"].items():
            p = a.inputs / name
            if sha256(p) != digest:
                raise ValueError(f"input identity changed: {name}")
            data = torch.load(p, map_location="cpu", weights_only=False)
            if data["population"] == population:
                members.append(data)
        for row in members:
            inp = {k: v.to(model.activation_device) for k, v in row["inputs"].items()}
            h, hx = model.hidden_from(inp)
            target = synthetic_targets(model, inp)
            loss, _ = distillation_loss(model, h, chunk=a.chunk, **target)
            # Also exercise the separately trained confidence tensor. This
            # explicit auxiliary objective is used only for gradient comparison;
            # the loss-value gate below uses unmodified distillation_loss.
            conf = (model.confidence(hx.detach(), target["previous"]).sigmoid() - .5).square().mean()
            cast(Callable[[], None], ((loss + .1 * conf) / len(members)).backward)()
            with torch.no_grad():
                picks, _ = model.greedy(h, hx, inp["anchor"])
                previous = torch.cat((inp["anchor"][:, None], picks[:, :-1]), -1)
                if a.command == "fast":
                    ref_row = torch.load(a.reference / (row["key"] + ".pt"), map_location="cpu", weights_only=False)
                    if ref_row["inputs_sha256"] != sha256(a.inputs / (row["key"] + ".pt")):
                        raise ValueError("reference sample differs from fast input")
                    previous = ref_row["previous"].to(model.activation_device)
                logits = torch.cat([model.logit_chunk(h, previous, lo, min(lo + 4096, model.c.vocab)).cpu()
                                    for lo in range(0, model.c.vocab, 4096)], -1)
            torch.save({"drafts": picks.cpu(), "hidden": h.detach().cpu(), "previous": previous.cpu(),
                        "logits": logits, "loss": float(loss.detach()), "population": population,
                        "inputs_sha256": sha256(a.inputs / (row["key"] + ".pt"))},
                       a.out / (row["key"] + ".pt"))
            print(json.dumps({"arm": a.command, "population": population, "sample": row["key"],
                              "loss": float(loss.detach())}), flush=True)
        grads = {n: None if p.grad is None else p.grad.detach().cpu() for n, p in model.named_parameters()}
        torch.save(grads, a.out / (population + ".gradients.pt"))
    summary = {"schema": "dsv41.d2.fast-arm.v1", "arm": a.command, "vocab_chunk": a.chunk, "lora_rank": a.rank,
               "backend": BACKEND if model.fast_training else "qualified-reference",
               "frozen_identity": model.frozen_identity, "inputs_sha256": sha256(a.inputs / "inputs.json"),
               "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "gradient_objective": "mean synthetic top20+tail distillation + 0.1 confidence MSE(target=0.5); no updates",
               "zero_lora_a_gradients": "Initial LoRA B is zero; zero/zero gradients are reported explicitly.",
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (a.out / "arm.json").write_text(json.dumps(summary, indent=2) + "\n")


def cosine(a: Tensor | None, b: Tensor | None) -> dict[str, float | bool]:
    if a is None or b is None:
        return {"pass": False, "cosine": 0., "missing_gradient": True}
    x, y = a.double().flatten(), b.double().flatten()
    nx, ny = x.norm(), y.norm()
    if nx == 0 or ny == 0:
        equal = bool(nx == 0 and ny == 0)
        return {"pass": equal, "cosine": float(equal), "both_zero": equal}
    c = float(x.dot(y) / (nx * ny))
    return {"pass": c >= .999, "cosine": c, "reference_norm": float(nx), "fast_norm": float(ny)}


def population_gates(rows: list[dict[str, Any]], gradients: dict[str, dict[str, float | bool]]) -> dict[str, Any]:
    if not rows or not gradients:
        raise ValueError("empty agreement population or gradient groups")
    kl = torch.tensor([r["kl"] for r in rows], dtype=torch.float64)
    medians = kl.quantile(.5, dim=0).tolist()
    p99 = kl.quantile(.99, dim=0).tolist()
    pos1 = sum(r["position1"] for r in rows) / len(rows)
    exact = sum(r["exact"] for r in rows) / len(rows)
    loss_error = max(r["loss_relative_error"] for r in rows)
    gates = {"drafts": pos1 >= .99 and exact >= .95,
             "kl": max(medians) <= 1e-3 and max(p99) <= 1e-2,
             "loss": loss_error <= .005,
             "gradients": all(g["pass"] and not g.get("both_zero", False) for g in gradients.values())}
    return {"pass": all(gates.values()), "gates": gates, "samples": len(rows),
            "position1_agreement": pos1, "exact_blocks": exact, "kl_median_by_position": medians,
            "kl_p99_by_position": p99, "max_loss_relative_error": loss_error,
            "gradient_groups": gradients, "rows": rows}


def compare(a: argparse.Namespace) -> None:
    torch.set_num_threads(8)
    reference = json.loads((a.reference / "arm.json").read_text())
    fast = json.loads((a.fast / "arm.json").read_text())
    for field in ("frozen_identity", "inputs_sha256", "gradient_objective", "vocab_chunk"):
        if reference[field] != fast[field]:
            raise ValueError(f"agreement arm identity differs: {field}")
    by_pop: dict[str, list[dict[str, Any]]] = {"n1": [], "captures": []}
    for p in sorted(a.reference.glob("*.pt")):
        if p.name.endswith(".gradients.pt"):
            continue
        ref = torch.load(p, map_location="cpu", weights_only=False)
        arm_data = torch.load(a.fast / p.name, map_location="cpu", weights_only=False)
        if ref["inputs_sha256"] != arm_data["inputs_sha256"]:
            raise ValueError("sample identity differs")
        if not torch.equal(ref["previous"], arm_data["previous"]):
            raise ValueError("logit KL conditionals differ")
        logp = ref["logits"].double().log_softmax(-1)
        logq = arm_data["logits"].double().log_softmax(-1)
        kl = (logp.exp() * (logp - logq)).sum(-1).squeeze(0)
        row = {"key": p.stem, "position1": bool(ref["drafts"][0, 0] == arm_data["drafts"][0, 0]),
               "exact": bool(torch.equal(ref["drafts"], arm_data["drafts"])), "kl": kl.tolist(),
               "loss_relative_error": abs(arm_data["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-30)}
        by_pop[ref["population"]].append(row)
    populations = {}
    inferred_rank: int | None = None
    for population, rows in by_pop.items():
        rg = torch.load(a.reference / (population + ".gradients.pt"), map_location="cpu", weights_only=False)
        fg = torch.load(a.fast / (population + ".gradients.pt"), map_location="cpu", weights_only=False)
        if rg.keys() != fg.keys():
            raise ValueError("gradient tensor names differ")
        ranks = {rg[n].shape[0] for n in rg if n.endswith(".a")}
        if len(ranks) != 1 or any(rg[n].shape != fg[n].shape for n in rg if rg[n] is not None and fg[n] is not None):
            raise ValueError("gradient tensor shapes/ranks differ")
        rank = next(iter(ranks))
        if inferred_rank is not None and inferred_rank != rank:
            raise ValueError("gradient populations have different LoRA ranks")
        inferred_rank = rank
        if any(arm_meta.get("lora_rank", rank) != rank for arm_meta in (reference, fast)):
            raise ValueError("declared LoRA rank differs from measured gradients")
        groups = tensor_groups(set(rg))
        gradients = {}
        for name, members in groups.items():
            if any(rg[n] is None or fg[n] is None for n in members):
                gradients[name] = cosine(None, None)
            else:
                gradients[name] = cosine(torch.cat([rg[n].flatten() for n in members]),
                                         torch.cat([fg[n].flatten() for n in members]))
        populations[population] = population_gates(rows, gradients)
        populations[population]["gradient_tensors"] = {n: cosine(rg[n], fg[n]) for n in rg}
        populations[population]["gradient_group_members"] = groups
        individual_pass = all(g["pass"] for g in populations[population]["gradient_tensors"].values())
        populations[population]["gates"]["gradients"] &= individual_pass
        populations[population]["pass"] = all(populations[population]["gates"].values())
        del rg, fg
    report = {"schema": "dsv41.d2.fast-agreement.v1", "backend": BACKEND,
              "frozen_identity": reference["frozen_identity"], "inputs_sha256": reference["inputs_sha256"],
              "source_sha256": fast["source_sha256"], "populations": populations,
              "pass": all(p["pass"] for p in populations.values()),
              "reference_arm_sha256": sha256(a.reference / "arm.json"),
              "fast_arm_sha256": sha256(a.fast / "arm.json"), "batch_size": 1,
              "vocab_chunk": reference["vocab_chunk"],
              "lora_rank": inferred_rank,
              "no_optimizer_updates": True, "gradient_objective": reference["gradient_objective"]}
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "populations": {n: {k: v for k, v in p.items()
          if k not in ("rows", "gradient_groups", "gradient_tensors", "gradient_group_members")}
          for n, p in populations.items()}}), flush=True)


def main() -> None:
    os.umask(0o077)
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--fixtures", type=Path, required=True)
    prep.add_argument("--reference-dumps", type=Path, required=True)
    prep.add_argument("--v1-root", required=True)
    prep.add_argument("--captures", required=True)
    prep.add_argument("--out", type=Path, required=True)
    for command in ("reference", "fast"):
        run = sub.add_parser(command)
        run.add_argument("--ckpt", required=True)
        run.add_argument("--inputs", type=Path, required=True)
        run.add_argument("--out", type=Path, required=True)
        run.add_argument("--chunk", type=int, default=4096)
        run.add_argument("--rank", type=int, choices=(32, 64), default=32)
        if command == "fast":
            run.add_argument("--reference", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("--fast", type=Path, required=True)
    cmp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.command == "prepare":
        prepare(a)
    elif a.command == "compare":
        compare(a)
    else:
        arm(a)


if __name__ == "__main__":
    main()
