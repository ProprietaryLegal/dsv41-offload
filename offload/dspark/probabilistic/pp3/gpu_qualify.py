"""PP3 solo test of the actual prepared image standard-rejection kernels.

No vLLM engine, model, Spark access or service changes. Five predeclared toy
targets, 20000 requests each; chi-square family alpha 1e-5 (df=3). Tests full
target mass outside q, disjoint/equal proposals, no-q invalidation and T0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import sys
import types
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from offload.dspark.worker.sampled import Top32, _finish
from offload.dspark.worker.tools.pp3_probe import UUID, timing


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules(source: Path, overlay: Path) -> Any:
    # Package-only shims avoid importing a GPU engine or its platform plugins.
    # All math functions (Gumbel, Philox, verifier) are the real pinned sources.
    names = ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu',
             'vllm.v1.worker.gpu.sample', 'vllm.v1.worker.gpu.spec_decode',
             'vllm.v1.watermarking', 'vllm.v1.watermarking.prfs')
    for name in names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    shim = types.ModuleType('vllm.triton_utils')
    for key, value in {'triton': triton, 'tl': tl, 'tldevice': libdevice, 'HAS_TRITON': True}.items():
        setattr(shim, key, value)
    sys.modules[shim.__name__] = shim
    for suffix in ('watermarking.prfs.base', 'watermarking.prfs.philox',
                   'worker.gpu.sample.gumbel', 'worker.gpu.sample.watermark'):
        load('vllm.v1.'+suffix, source / ('vllm/v1/'+suffix.replace('.', '/')+'.py'))
    return load('vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils',
                overlay / 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        raise RuntimeError('solo UUID required')
    import fcntl
    fcntl.flock(int(os.environ['PP3_LOCK_FD']), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != UUID.removeprefix('GPU-'):
        raise RuntimeError('CUDA UUID mismatch')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    rs = modules(a.source, a.overlay)
    dev = torch.device('cuda:0')
    n, vocab = 20000, 64
    cases = ('full_overlap', 'sparse_tail', 'disjoint', 'equal', 'invalidated')
    results: list[dict[str, object]] = []
    receipt: dict[str, object] = {'schema': 'pp3.gpu-qualification.v1', 'complete': False,
        'classification': 'MEASURED solo SM86; actual prepared image kernels, engine not loaded',
        'cases': results, 'trials_per_case': n, 'family_alpha': 1e-5,
        'torch': torch.__version__, 'triton': triton.__version__, 'uuid': UUID,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    receipt['dependency_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (a.source, a.overlay) for path in sorted(root.rglob('*.py'))}
    producer_path = importlib.import_module('offload.dspark.worker.sampled').__file__
    if producer_path is None:
        raise RuntimeError('producer source path missing')
    receipt['producer_sha256'] = hashlib.sha256(Path(producer_path).read_bytes()).hexdigest()

    def save() -> None:
        receipt['maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt['gpu_peak_bytes'] = torch.cuda.max_memory_allocated()
        a.out.write_text(json.dumps(receipt, indent=2)+'\n')

    save()
    for case_id, case in enumerate(cases):
        weights = torch.tensor([.125, .25, .25, .375], device=dev)
        pids = torch.tensor([40, 41, 42, 43], device=dev)
        qids = pids if case != 'disjoint' else torch.tensor([0, 1, 2, 3], device=dev)
        qw = weights if case == 'equal' else torch.tensor([.5, .125, .125, .25], device=dev)
        if case == 'sparse_tail':
            qw = torch.tensor([.5, .5, 0., 0.], device=dev)
        logits = torch.full((1, vocab), -torch.inf, device=dev)
        logits[0, qids] = qw.log() * .7
        sampler = Top32(1, vocab, dev)
        keys = torch.tensor([[4103000+case_id, 1, 32768]], dtype=torch.int64, device=dev)
        sampler(logits, keys, torch.tensor([.7], device=dev))
        # Reuse selected candidates, varying the actual producer's Philox key.
        candidates = sampler.candidates.expand(n, -1).contiguous()
        flags = torch.zeros(n, sampler.tiles, dtype=torch.int32, device=dev)
        keys = torch.stack((torch.arange(n, device=dev, dtype=torch.int64)+4103000+case_id*100000,
                            torch.full((n,), 1, device=dev), torch.full((n,), 32768, device=dev)), 1)
        token = torch.empty(n, dtype=torch.int32, device=dev)
        support = torch.empty(n, 32, dtype=torch.int32, device=dev)
        q = torch.empty(n, 32, device=dev)
        error = torch.empty(n, dtype=torch.int32, device=dev)
        _finish[(n,)](candidates, keys, torch.full((n,), .7, device=dev), token, support, q,
                       error, flags, candidates.shape[1], triton.next_power_of_2(candidates.shape[1]), num_warps=4)
        if error.any().item():
            raise AssertionError('producer error')
        dense = torch.zeros(n, 1, vocab, device=dev)
        dense[:, 0].scatter_(1, support.long(), q)
        target = torch.full((n, 2, vocab), -torch.inf, device=dev)
        target[:, :, pids] = weights.log()
        proposal = torch.stack((torch.zeros_like(token), token), 1).flatten()
        if case == 'invalidated':
            proposal[1::2] = -1
            dense.zero_()
        cu = torch.arange(0, n*2+1, 2, dtype=torch.int32, device=dev)
        mapping = torch.arange(n, dtype=torch.int32, device=dev)
        expanded = mapping.repeat_interleave(2)
        localpos = torch.tensor([0, 1], device=dev, dtype=torch.int32).repeat(n)
        positions = (localpos+32768).long()
        seeds = torch.arange(n, device=dev, dtype=torch.int64)+9900000+case_id*100000
        temp = torch.ones(n, device=dev)
        os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '1'
        emitted, lengths = rs.rejection_sample(target.flatten(0, 1), dense, proposal,
            cu, positions, mapping, expanded, localpos, temp, seeds, 1)
        counts = torch.bincount(emitted[:, 0], minlength=vocab)
        outside = int(counts.sum().item()-counts[pids].sum().item())
        if outside:
            raise AssertionError('output escaped target support')
        observed = counts[pids].cpu().tolist()
        expected = [n*x for x in (.125, .25, .25, .375)]
        chi = sum((float(o)-e)**2/e for o, e in zip(observed, expected, strict=True))
        x = math.sqrt(chi/2)
        pvalue = math.erfc(x) + math.sqrt(2*chi/math.pi)*math.exp(-chi/2)
        if pvalue <= 1e-5/len(cases):
            raise AssertionError(f'{case}: chi-square rejected ({chi}, p={pvalue})')
        row: dict[str, object] = {'case': case, 'counts': observed, 'chi_square_df3': chi,
                                  'p_value': pvalue, 'accepted': int((lengths > 1).sum().item())}
        results.append(row)
        save()
        print(json.dumps(row), flush=True)
        # Same target, same greedy requests/seeds: actual image old/new paths
        # must agree byte-for-byte despite q being present in the new path.
        temp.zero_()
        os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '0'
        old, old_lengths = rs.rejection_sample(target.flatten(0, 1), None, proposal,
            cu, positions, mapping, expanded, localpos, temp, seeds, 1)
        os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '1'
        new, new_lengths = rs.rejection_sample(target.flatten(0, 1), dense, proposal,
            cu, positions, mapping, expanded, localpos, temp, seeds, 1)
        torch.testing.assert_close(old_lengths, new_lengths, rtol=0, atol=0)
        # Image allocates sampled with torch.empty; positions >=num_sampled
        # are undefined storage, not emitted tokens. Compare every defined byte.
        valid = torch.arange(old.shape[1], device=dev)[None, :] < old_lengths[:, None]
        torch.testing.assert_close(old[valid], new[valid], rtol=0, atol=0)
    # Deterministic ties, signed zero, bad logits/temperatures, and reordering.
    sampler = Top32(4, 64, dev)
    x = torch.zeros(4, 64, device=dev)
    x[:, ::2] = -0.
    keys = torch.tensor([[4103+j, 12, 32768] for j in range(4)], device=dev)
    tau = torch.full((4,), .7, device=dev)
    tok, support, q = sampler(x, keys, tau)
    first = tok.clone()
    torch.testing.assert_close(support, torch.arange(32,device=dev,dtype=torch.int32).expand(4,-1),rtol=0,atol=0)
    torch.testing.assert_close(q, torch.full((4,32),1/32,device=dev),rtol=0,atol=0)
    tok, _, _ = sampler(x.flip(0), keys.flip(0), tau)
    torch.testing.assert_close(tok, first.flip(0),rtol=0,atol=0)
    x[0] = -torch.inf
    x[1, 50] = torch.nan
    x[2, 50] = torch.inf
    tau[3] = 0.
    tok, _, q = sampler(x, keys, tau)
    assert sampler.error.tolist() == [1]*4 and tok.tolist() == [-1]*4 and not q.any().item()
    receipt['producer_edges'] = 'PASS ties/signed-zero/reordering/NaN/+inf/all-masked/tau0'
    # Full-vocabulary dense-p verification cost. No target model or processors.
    b, depth, v = 1, 5, 129280
    torch.manual_seed(4103)
    target = torch.randn(depth+1, v, device=dev)
    dense = torch.zeros(b, depth, v, device=dev)
    dense[:, :, :32] = 1/32
    proposal = torch.zeros(depth+1, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, depth+1], dtype=torch.int32, device=dev)
    mapping = torch.zeros(1, dtype=torch.int32, device=dev)
    expanded = torch.zeros(depth+1, dtype=torch.int32, device=dev)
    localpos = torch.arange(depth+1, dtype=torch.int32, device=dev)
    positions = localpos.long()+32768
    temp, seeds = torch.ones(1, device=dev), torch.tensor([9900000], device=dev)
    def run(q: torch.Tensor | None = dense) -> object:
        return rs.rejection_sample(target, q, proposal, cu, positions, mapping, expanded,
                                   localpos, temp, seeds, depth)
    os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '1'
    receipt['dense_p_verifier'] = timing(run)
    os.environ['VLLM_DSPARK_SAMPLED_TOP32'] = '0'
    receipt['greedy_q_verifier'] = timing(lambda: run(None))
    # Sparse support IDs + q staged once in each direction; no network claim.
    payload = torch.zeros(320, dtype=torch.int32, device=dev)
    host = torch.empty_like(payload, device='cpu', pin_memory=True)
    receipt['sparse1280_d2h'] = timing(lambda: host.copy_(payload, non_blocking=True))
    receipt['sparse1280_h2d'] = timing(lambda: payload.copy_(host, non_blocking=True))
    dense = torch.empty(1,5,v,device=dev)
    support = torch.arange(32,device=dev).expand(1,5,-1)
    q = torch.full((1,5,32),1/32,device=dev)
    def expand_q() -> None:
        dense.zero_()
        dense.scatter_(2,support,q)
    receipt['dense_q_expansion'] = timing(expand_q)
    receipt['greedy_identity_requests'] = n*len(cases)
    receipt['complete'] = True
    save()


if __name__ == '__main__':
    main()
