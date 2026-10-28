"""Native-file census and isolated real-shape GPU qualification; never downloads.

CPU census: CUDA_VISIBLE_DEVICES='' python -m ...qualify --census --output FILE
GPU run: run_locked.sh -m ...qualify --output FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from collections.abc import Callable

import torch
from safetensors import safe_open

from . import PackedExperts, activation_into, combine_into
from .reference import decode_codes, unpack

ROOT = Path('/opt/dsv41/x86host-data/native-1017')


def tensor(stage: int, expert: int, proj: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = ROOT / f'model-{44+stage:05d}-of-00048.safetensors'
    key = f'mtp.{stage}.ffn.experts.{expert}.w{proj}'
    with safe_open(path, framework='pt', device='cpu') as f:
        return f.get_tensor(key+'.weight').view(torch.uint8), f.get_tensor(key+'.scale').view(torch.uint8)


def census() -> dict[str, object]:
    """Bounded raw byte/scale census; independent of GPU decoding."""
    histogram = torch.zeros(4096, dtype=torch.int64)
    tensors: list[dict[str, object]] = []
    start = time.monotonic()
    for stage in range(3):
        for expert in range(128):
            for proj in (1, 2, 3):
                p, s = tensor(stage, expert, proj)
                # Grouped nibble coordinates; no float expansion, one expert at a time.
                grouped = p.reshape(p.shape[0], -1, 16)
                ss = s.to(torch.int32).unsqueeze(-1)*16
                for nibble in (grouped & 15, grouped >> 4):
                    histogram += torch.bincount((ss+nibble).flatten().long(), minlength=4096)
                tensors.append({'stage': stage, 'expert': expert, 'projection': proj,
                                'packed_shape': list(p.shape), 'scale_shape': list(s.shape)})
        print(f'census stage {stage} complete: {time.monotonic()-start:.1f}s', flush=True)
    codes = torch.arange(16).repeat(256).byte()
    scales = torch.arange(256).repeat_interleave(16).byte()
    values = decode_codes(codes, scales)
    used = histogram != 0
    bf = values.bfloat16().double()
    bad = used & (~torch.isfinite(values) | ~torch.isfinite(bf) | (bf != values))
    subnormal = used & (values != 0) & (values.abs() < torch.finfo(torch.bfloat16).tiny)
    files: dict[str, str] = {}
    for stage in range(3):
        path = ROOT / f'model-{44+stage:05d}-of-00048.safetensors'
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8*1024*1024), b''):
                digest.update(block)
        files[path.name] = digest.hexdigest()
    assert not bad.any(), f'non-BF16 values present: {(bad.nonzero().flatten()).tolist()}'
    assert not subnormal.any(), 'BF16 subnormal operands present; compute not admitted'
    return {'boundary': 'CPU full native routed-expert byte/scale census, all stages/experts/projections',
            'elapsed_s': time.monotonic()-start, 'files_sha256': files,
            'native_value_count': int(histogram.sum()), 'present_code_scale_pairs': int(used.sum()),
            'scale_min': int(scales[used].min()), 'scale_max': int(scales[used].max()),
            'unsupported_pairs': int(bad.sum()), 'subnormal_pairs': int(subnormal.sum()),
            'histogram_scale_major': histogram.tolist(), 'tensors': tensors}


def telemetry() -> str:
    return subprocess.check_output([
        'nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,power.limit,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used',
        '--format=csv,noheader'], text=True).strip()


def measured(fn: Callable[[], None]) -> dict[str, object]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(20):
            fn()
    times = []
    for _ in range(7):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end)*1000/20)
    return {'us_samples': times, 'median_us': statistics.median(times)}


def bench(experts: int, regfed: bool = False) -> dict[str, object]:
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'
            or os.environ.get('KX_MXFP4_LOCKED') != '1'):
        raise RuntimeError('GPU qualification requires run_locked.sh')
    if not 3 <= experts <= 128:
        raise ValueError('3..128 experts required')
    torch.manual_seed(12354)
    torch.cuda.reset_peak_memory_stats()
    dev = torch.device('cuda:0')
    from .regfed import RegisterExperts
    banks: list[PackedExperts | RegisterExperts] = []
    cpu_shards: list[tuple[torch.Tensor, torch.Tensor]] = []
    load_start = time.monotonic()
    for proj in (1, 3, 2):
        pp, ss = [], []
        for e in range(experts):
            p, s = tensor(0, e, proj)
            # TP rank 0 channel shard (rank-1 is checked independently below).
            if proj == 2:
                p, s = p[:, :576], s[:, :36]
            else:
                p, s = p[:1152], s[:1152]
            pp.append(p.contiguous())
            ss.append(s.contiguous())
        p, s = torch.stack(pp), torch.stack(ss)
        banks.append((RegisterExperts if regfed else PackedExperts).from_checkpoint(p, s, dev))
        cpu_shards.append((p, s))
    gate, up, down = banks
    result: dict[str, object] = {
        'boundary': 'isolated stage-0 native TP2 channel-shard linear algebra on one development GPU; NOT full worker/TP2/acceptance',
        'data': 'official native checkpoint weights; seeded BF16 synthetic activations and explicit synthetic routes',
        'experts': experts, 'regfed': regfed, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
        'cuda': torch.version.cuda, 'load_seconds': time.monotonic()-load_start,
        'telemetry_start': telemetry(), 'peak_allocated_after_load': torch.cuda.max_memory_allocated(),
        'packed_resident_bytes': sum(b.codes.numel()+b.scales.numel() for b in banks),
        'measurements': [], 'numerics': []}
    measurements: list[dict[str, object]] = []
    numerics: list[dict[str, object]] = []
    for rows in (1, 5, 20):
        for reuse in ('spread', 'three_reused'):
            ids = (torch.arange(rows*3, dtype=torch.int32, device=dev) % (experts if reuse == 'spread' else 3)).view(rows,3)
            x = torch.randn(rows, 5120, device=dev, dtype=torch.bfloat16)*.1
            g = torch.empty(rows,3,1152, device=dev)
            u = torch.empty_like(g)
            a = torch.empty(rows*3,1152, device=dev, dtype=torch.bfloat16)
            parts = torch.empty(rows,3,5120,device=dev)
            output = torch.empty(rows,5120,device=dev)
            weights = torch.full((rows,3), .5, device=dev)
            ones = torch.ones_like(weights)
            status = torch.zeros(1,dtype=torch.int32,device=dev)
            schedule = torch.empty(rows*3,8,dtype=torch.int32,device=dev)

            def linear(bank: PackedExperts | RegisterExperts, xx: torch.Tensor,
                       yy: torch.Tensor, mode: str, *, up_bank: PackedExperts | RegisterExperts | None = None,
                       up_result: torch.Tensor | None = None, per_slot: bool = False) -> None:
                if isinstance(bank,RegisterExperts):
                    assert up_bank is None or isinstance(up_bank,RegisterExperts)
                    bank.linear_into(xx,ids,yy,status,schedule,up=up_bank,up_out=up_result,per_slot=per_slot)
                else:
                    assert up_bank is None or isinstance(up_bank,PackedExperts)
                    bank.linear_into(xx,ids,yy,status,up=up_bank,up_out=up_result,per_slot=per_slot,mode=mode)

            for mode in (('regfed',) if regfed else ('mma', 'simt')):
                for fused in (False, True):
                    def run() -> None:
                        status.zero_()
                        if fused:
                            linear(gate,x,g,mode,up_bank=up,up_result=u)
                        else:
                            linear(gate,x,g,mode)
                            linear(up,x,u,mode)
                        activation_into(g,u,weights,a,10.)
                        linear(down,a,parts,mode,per_slot=True)
                        combine_into(parts,ones,output)
                    timing = measured(run)
                    assert status.item() == 0
                    assert torch.isfinite(output).all()
                    record = {'rows':rows,'top_k':3,'reuse':reuse,'mode':mode,'fused_gate_up':fused,
                              'telemetry':telemetry(), **timing}
                    measurements.append(record)
                    print(json.dumps(record),flush=True)
                # Full projection CPU FP32 and selected-column FP64 comparisons.
                linear(gate,x,g,mode)
                actual = g[0,0].cpu()
                native = unpack(cpu_shards[0][0][:1],cpu_shards[0][1][:1])[0].float()
                ref32 = x[0].cpu().float() @ native.T
                ref64 = x[0].cpu().double() @ native.double().T
                err = (actual.double()-ref64).abs()
                torch.testing.assert_close(actual,ref32,atol=2e-4,rtol=2e-4)
                numerics.append({'rows':rows,'reuse':reuse,'mode':mode,
                                 'max_abs_vs_fp64':float(err.max()),
                                 'rms_vs_fp64':float(err.square().mean().sqrt()),
                                 'max_abs_vs_cpu_fp32':float((actual-ref32).abs().max()),
                                 'exact_fp32_fraction':float((actual==ref32).float().mean())})
    # Check the actual second channel shard, including group32 input slicing.
    shard_evidence = []
    for stage in range(3):
        for proj in (1,2,3):
            p,s = tensor(stage,0,proj)
            if proj == 2:
                p,s = p[:,576:],s[:,36:]
            else:
                p,s = p[1152:],s[1152:]
            w = (RegisterExperts if regfed else PackedExperts).from_checkpoint(p[None].contiguous(),s[None].contiguous(),dev)
            x = torch.randn(1,w.inputs,device=dev,dtype=torch.bfloat16)*.1
            ids = torch.zeros(1,1,device=dev,dtype=torch.int32)
            out = torch.empty(1,1,w.outputs,device=dev)
            status = torch.zeros(1,device=dev,dtype=torch.int32)
            if isinstance(w,RegisterExperts):
                sched = torch.empty(1,8,device=dev,dtype=torch.int32)
                w.linear_into(x,ids,out,status,sched)
            else:
                w.linear_into(x,ids,out,status)
            ref = x.cpu().float() @ unpack(p[None],s[None])[0].float().T
            torch.testing.assert_close(out[0].cpu(),ref,atol=2e-4,rtol=2e-4)
            shard_evidence.append({'stage':stage,'projection':proj,'rank':1,
                                   'max_abs_vs_cpu_fp32':float((out[0].cpu()-ref).abs().max())})
    result['rank1_shard_checks'] = shard_evidence
    result['measurements'] = measurements
    result['numerics'] = numerics
    result['telemetry_end'] = telemetry()
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--census',action='store_true')
    parser.add_argument('--experts',type=int,default=128)
    parser.add_argument('--regfed',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = census() if args.census else bench(args.experts,args.regfed)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
