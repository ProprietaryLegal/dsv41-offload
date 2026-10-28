"""0064 actual installed kernels on every loaded target expert bank, before TP.

Call run(worker) on BOTH idle ranks through the qualification bootstrap. Uses
fixed public random activations and the worker's resident weights/IDs/permutation,
never loads another bank, saves tensors, changes capacity, or substitutes BF16
tolerance. A separate teacher-forced full-model/task review remains mandatory.
"""
from __future__ import annotations

import ast
import importlib
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch

from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3.engine_common import Cases, attr, finite, invoke, passed, replaced, require, runner, source

ROWS = (33, 42, 43, 63)


def equal_bytes(a: torch.Tensor, b: torch.Tensor, label: str) -> int:
    require(a.shape == b.shape and a.dtype == b.dtype, label + ': shape/dtype mismatch')
    require(bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))),
            label + ': output bytes differ')
    return a.numel() * a.element_size()


def unchanged_call(candidate: bytes, stock: bytes) -> None:
    """Compare the actual installed oracle body to the reviewed unmodified one."""
    def body(data: bytes) -> str:
        tree = ast.parse(data)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmoeLayer')
        call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
        return ast.dump(call, include_attributes=False)
    require(body(candidate) == body(stock), '0064 oracle __call__ changed')


def check_layer(method: object, forward: object) -> Cases:
    bank, smoe = attr(method, 'bank'), attr(method, 'smoe')
    require(smoe is not None, 'loaded layer has no smoe')
    require(invoke(attr(smoe, 'max_rows')) == 42 and attr(smoe, 'top_k') == 6, 'actual capacity must stay 42')
    perm = cast(torch.Tensor, attr(method, 'perm'))
    hidden = int(cast(int, attr(bank, 'hidden')))
    experts = int(cast(int, attr(bank, 'num_experts')))
    limit = cast(float, attr(method, 'act_limit'))
    dev = perm.device
    result: Cases = {}
    def full(x_rows: torch.Tensor, ids_rows: torch.Tensor, w_rows: torch.Tensor,
             mask: torch.Tensor | None) -> torch.Tensor:
        return cast(torch.Tensor, invoke(attr(forward, 'routed_moe_forward'), x_rows, w_rows, ids_rows, perm,
            attr(method, 'groups'), bank, attr(method, 'coop'), attr(method, 'fused'), limit,
            attr(method, 'mcg'), attr(method, 'mul1'),
            decode_kernel=attr(attr(method, 'settings'), 'decode_kernel'), par=attr(method, 'par'),
            smoe=smoe, pad=mask))
    # Preserve the serving RNG; no dependence on the proprietary prompt stream.
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(640042)
        x = torch.randn(64, hidden, dtype=torch.bfloat16, device=dev)
        ids = (torch.arange(64 * 6, device=dev).reshape(64, 6) % experts).to(torch.int32)
        weights = torch.rand(64, 6, dtype=torch.float32, device=dev)
        weights /= weights.sum(dim=1, keepdim=True)
        for m in ROWS:
            compared = 0
            for padding in (False, True):
                mask = torch.arange(m, device=dev) % 7 == 0 if padding else None
                row_ids = ids[:m].clone()
                if padding:
                    row_ids[::5, -1] = -1  # unrouted picks as well as padded rows
                args = (x[:m], row_ids, weights[:m], perm, limit, mask)
                require(invoke(attr(forward, 'smoe_split_ok'), m, x[:m],
                    attr(attr(method, 'settings'), 'decode_kernel'), attr(method, 'coop'), smoe) is True,
                    f'm{m}: real dispatch not eligible')
                actual = full(x[:m], row_ids, weights[:m], mask)
                if m <= 42:
                    ref = cast(torch.Tensor, invoke(smoe, *args))
                else:
                    ref = torch.cat([cast(torch.Tensor, invoke(smoe, x[t:t+1], row_ids[t:t+1],
                        weights[t:t+1], perm, limit, None if mask is None else mask[t:t+1])) for t in range(m)])
                compared += equal_bytes(actual, ref, f'm{m}')
            result[f'm{m}'] = passed(2, rows=m, capacity=42, oracle='single-smoe' if m <= 42 else 'per-row-decode',
                comparison='every-real-output-byte-before-TP', matched_inputs_weights_ids_perm_pad_limit=True,
                bytes_compared=compared)
        for m in (32, 64):
            args_full = (x[:m], weights[:m], ids[:m], perm, attr(method, 'groups'), bank,
                         attr(method, 'coop'), attr(method, 'fused'), limit,
                         attr(method, 'mcg'), attr(method, 'mul1'))
            opts = {'decode_kernel': attr(attr(method, 'settings'), 'decode_kernel'),
                    'par': attr(method, 'par'), 'smoe': smoe}
            with replaced(forward, '_SPLIT_MAX', 0):
                control = cast(torch.Tensor, invoke(attr(forward, 'routed_moe_forward'), *args_full, **opts)).clone()
            with replaced(forward, '_SPLIT_MAX', 63):
                require(invoke(attr(forward, 'smoe_split_ok'), m, x[:m], opts['decode_kernel'],
                    attr(method, 'coop'), smoe) is False, f'dispatch{m} entered split')
                candidate = cast(torch.Tensor, invoke(attr(forward, 'routed_moe_forward'), *args_full, **opts))
            result[f'dispatch{m}'] = passed(1, unchanged=True,
                bytes_compared=equal_bytes(control, candidate, f'dispatch{m}'))
    torch.cuda.synchronize(dev)
    return result


def logits_sha256(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> str:
    """Bind human task review to the exact C/C/A numerical probe, without saving tensors."""
    h = hashlib.sha256()
    for value in logits:
        header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True).encode()
        h.update(len(header).to_bytes(8, 'little'))
        h.update(header)
        h.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def rounding(logits_c1: torch.Tensor, logits_c2: torch.Tensor, logits_arm: torch.Tensor,
             review: dict[str, Json], *, run: str) -> Json:
    """Full-vocabulary teacher-forced probability check; review must bind the actual probes.

    Caller supplies outputs from repeated identical-prefix, identical-batch forwards
    (C,C,A). Task assessment is a separate human review, never inferred from KL.
    """
    require(logits_c1.shape == logits_c2.shape == logits_arm.shape and logits_c1.ndim == 2,
            'teacher-forced logits shape differs')
    require(review.get('run') == run and review.get('task_review') == 'pass'
            and review.get('same_prefix_and_batch_shape') is True, 'missing bound task/prefix review')
    for key in ('prefix_sha256', 'batch_sha256', 'probe_sha256'):
        value = str(review.get(key, ''))
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'missing ' + key)
    require(all(bool(torch.isfinite(t).all()) for t in (logits_c1, logits_c2, logits_arm)), 'nonfinite logits')
    require(review['probe_sha256'] == logits_sha256((logits_c1, logits_c2, logits_arm)),
            'task review is not bound to these exact C/C/A logits')
    lc, lr, la = (torch.log_softmax(t.double(), dim=-1) for t in (logits_c1, logits_c2, logits_arm))
    pc = lc.exp()
    cc = finite(float((pc * (lc-lr)).sum(dim=-1).clamp_min(0).max().item()), 'C/C KL')
    kl = finite(float((pc * (lc-la)).sum(dim=-1).clamp_min(0).max().item()), 'C/A KL')
    require(kl <= max(.005, 2*cc), 'teacher-forced KL exceeds C/C floor')
    return passed(logits_c1.shape[0], same_prefix_and_batch_shape=True, captured_mass=1.,
                  kl=kl, cc_kl=cc, task_review='pass', review=review)


def run(worker: object, *, rounding_evidence: Json) -> Cases:
    r = runner(worker)
    forward = importlib.import_module('vllm_exl3_moe.forward')
    smoe_module = importlib.import_module('vllm_exl3_moe.smoe')
    # Pinned stock body is shipped as source, not imported in place of the engine.
    stock = Path(__file__).with_name('stock_smoe.py.txt').read_bytes()
    unchanged_call(Path(cast(str, smoe_module.__file__)).read_bytes(), stock)
    require(attr(forward, '_SPLIT_MAX') == 63, 'split arm is not enabled')
    model = attr(r, 'model')
    methods = [attr(module, 'quant_method') for _, module in cast(Iterable[tuple[str, object]],
        invoke(attr(model, 'named_modules'))) if hasattr(module, 'quant_method')
        and hasattr(attr(module, 'quant_method'), 'smoe')]
    require(bool(methods), 'no loaded target expert banks')
    cases: Cases = {}
    for method in methods:
        layer = check_layer(method, forward)
        for name, row in layer.items():
            if name not in cases:
                cases[name] = row
            else:
                existing, other = record(cases[name]), record(row)
                existing['samples'] = number(existing['samples']) + number(other['samples'])
                existing['bytes_compared'] = number(existing['bytes_compared']) + number(other['bytes_compared'])
    cases['teacher-forced-rounding'] = rounding_evidence
    return cases
