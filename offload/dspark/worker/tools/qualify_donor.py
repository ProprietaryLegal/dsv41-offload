"""W1b standalone donor qualification, in decode -> op -> full-step order.

No listeners or service calls. Every tensor is local to x86host. Decode checks all
real shards against the installed decoder and independent CPU corner tiles.
Numerical differences are recorded, never accepted as an exact replacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .profile_support import require_gpu_window

if TYPE_CHECKING:
    import torch


def differences(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int | bool]:
    import torch
    a, b = a.float(), b.float()
    d = a - b
    unequal = a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)
    return {"exact": not bool(unequal.any()), "mismatches": int(unequal.sum()),
            "nonfinite": int((~torch.isfinite(a)).sum()), "max_abs": float(d.abs().max()),
            "rel_fro": float(d.norm() / b.norm().clamp_min(1e-30))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("decode", "op", "full"))
    p.add_argument("donor", choices=("trellis", "cuda-exl3"))
    p.add_argument("root", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--subset", choices=("dense", "small", "head"), default="small")
    p.add_argument("--names", nargs="+", help="bounded repeat of named op controls")
    args = p.parse_args()
    require_gpu_window()
    import torch
    from ..checkpoint import Checkpoint
    from ..exl3 import kernels as K
    from ..model import Exl3Linear, DSparkDrafter, facc_ext
    from .donor_adapter import CudaExl3Dense, TrellisDense
    from .exl3_identity import cpu_decode, linear_ref64
    from .experiment_support import paired_graphs
    torch.set_num_threads(2)
    ck = Checkpoint(str(args.root))
    dev = torch.device("cuda", 0)
    donor = (TrellisDense if args.donor == "trellis" else CudaExl3Dense)(args.source, args.build)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    output = args.out.open("w")

    def record(row: dict[str, object]) -> None:
        output.write(json.dumps(row) + "\n")
        output.flush()
        print(json.dumps(row), flush=True)

    names = sorted(n.removesuffix(".trellis") for n in ck.entries if n.endswith(".trellis"))
    gen = torch.Generator(device=dev).manual_seed(41)

    class Candidate:
        def __init__(self, lin: Exl3Linear):
            self.lin = lin
            self.in_features, self.out_features = lin.in_features, lin.out_features
            self.packed = donor.prepare(lin.trellis, allow_wide=True)[0] if args.donor == "trellis" else None

        def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
            # Match current input conversion; preserve fp32 main_proj (not replaced).
            x = x.to(torch.float16).contiguous()
            if args.donor == "trellis":
                y = torch.empty(x.shape[0], self.out_features, device=dev, dtype=torch.float16)
                donor.linear(x, self.packed, self.lin.suh, self.lin.svh, torch.empty_like(x), y)
            else:
                y = donor.linear(x, self.lin.trellis, self.lin.suh[None], self.lin.svh, [self.out_features])
            y = y.float()
            if out is not None:
                out.copy_(y)
                return out
            return y

    if args.phase == "decode":
        for name in names:
            require_gpu_window()
            tr, suh, svh, bits = ck.exl3(name)
            if args.donor == "trellis" and bits == 2:
                continue  # explicitly reported unsupported coverage below
            td = tr.to(dev)
            ref = torch.empty(td.shape[0]*16, td.shape[1]*16, dtype=torch.float16, device=dev)
            K.ext().reconstruct(ref, td, bits, False, True)
            corners = []
            for rs, cs in ((0, 0), (tr.shape[0]-2, tr.shape[1]-2)):
                cpu = cpu_decode(tr[rs:rs+2, cs:cs+2].contiguous(), bits)
                corners.append(torch.equal(cpu.view(torch.int16), ref[rs*16:(rs+2)*16, cs*16:(cs+2)*16].cpu().contiguous().view(torch.int16)))
            if args.donor == "cuda-exl3":
                results = {}
                for vector in (False, True):
                    got = torch.ops.w1_donor.decode(td, vector)
                    results[str(vector)] = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
                    del got
            else:
                packed, plan = donor.prepare(td, allow_wide=True)
                if not torch.equal(donor.unpack(packed), td):
                    raise RuntimeError(f"lossless repack failed: {name}")
                mismatch = 0
                # Identity rows isolate actual kernel decode, with no Hadamards.
                for r in range(0, ref.shape[0], 32):
                    require_gpu_window()
                    x = torch.zeros(min(32, ref.shape[0]-r), ref.shape[0], dtype=torch.float16, device=dev)
                    x[torch.arange(x.shape[0], device=dev), torch.arange(r, r+x.shape[0], device=dev)] = 1
                    y = torch.empty(x.shape[0], ref.shape[1], dtype=torch.float16, device=dev)
                    donor.rotated(x, packed, y)
                    mismatch += int((y.view(torch.int16) != ref[r:r+x.shape[0]].view(torch.int16)).sum())
                results = {"rotated_identity": mismatch}
                del packed, x, y
            record({"name": name, "bits": bits, "elements": ref.numel(), "mismatches": results,
                    "cpu_corner_tiles_exact": all(corners)})
            if not all(corners) or any(results.values()):
                raise RuntimeError(f"decode identity failed: {name}: {results}")
            del td, ref
        record({"phase": "decode_complete", "supported_shards": sum(ck.exl3(n)[3] != 2 or args.donor != "trellis" for n in names),
                "unsupported_k2_shards": 1152 if args.donor == "trellis" else 0})
    elif args.phase == "op":
        # Every dense shard and representative routed shards at every stage.
        selected = [n for n in names if ".experts." not in n]
        if args.donor == "cuda-exl3":
            selected += [f"mtp.{s}.ffn.experts.{e}.{w}" for s in range(3) for e in (0, 77, 127) for w in ("w1", "w2", "w3")]
        if args.names:
            if any(n not in selected for n in args.names):
                raise ValueError("--names includes a matrix outside this donor's op coverage")
            selected = args.names
        for name in selected:
            require_gpu_window()
            tr, suh, svh, bits = ck.exl3(name)
            main_proj = name == "mtp.0.main_proj"
            current = Exl3Linear(tr, suh, svh, bits, dev, prescale=main_proj,
                                 ext=facc_ext() if main_proj else None,
                                 force_shape=DSparkDrafter.HEAD_SHAPE if name == "head" else 0)
            candidate = Candidate(current)
            # Independent fp64 reference, all columns, both Hadamards. Host RAM only.
            w = cpu_decode(tr, bits)
            for rows in (1, 2, 3, 4):
                require_gpu_window()
                x = (torch.randn(rows, current.in_features, generator=gen, device=dev)*.3).half()
                torch.save(x.cpu(), args.out.with_name(args.out.stem + f"-{name}-rows{rows}-input.pt"))
                now, got = current(x), candidate(x)
                ref = linear_ref64(x.cpu(), w, suh, svh).float().to(dev)
                record({"name": name, "rows": rows, "bits": bits, "donor_vs_current": differences(got, now),
                        "donor_vs_fp64_double_hadamard": differences(got, ref), "current_vs_fp64": differences(now, ref),
                        "timing": paired_graphs(lambda: current(x), lambda: candidate(x))})
            del current, candidate, w, ref, now, got
    else:
        if args.baseline is None:
            p.error("full requires --baseline containing saved synthetic.pt/replies.jsonl")
        m = DSparkDrafter(ck, dev, slots=8)
        saved = torch.load(args.baseline / "synthetic.pt", weights_only=True)
        taps = saved["taps"].to(dev)
        for slot, last in enumerate((300, 500, 700, 900)):
            pos = torch.arange(last-127, last+1, device=dev)
            m.insert(torch.full_like(pos, slot), pos, m.main_x(taps[pos]))
        # main_proj is never replaced: required prescale/fp32 accumulation.
        if args.subset == "head":
            m.head = Candidate(m.head)
        else:
            attrs = ("wq_a", "wkv", "sh_w1", "sh_w2", "sh_w3") if args.subset == "small" else ("wq_a", "wkv", "wq_b", "wo_b", "sh_w1", "sh_w2", "sh_w3")
            for st in m.stages:
                for attr in attrs:
                    setattr(st, attr, Candidate(getattr(st, attr)))
        require_gpu_window()
        # This saved matrix inserts <=24 rows. The bounded donor build contains
        # <=32-row families; never claim its unbuilt 64/128-row dispatch was tested.
        m.capture(batch_sizes=(1,2,3,4), ks=(5,), n_outs=(1,2,3,4,5), insert_rows=(8,16,32), margins=False)
        for line in (args.baseline / "replies.jsonl").read_text().splitlines():
            require_gpu_window()
            row = json.loads(line)
            b, n, rows, cache = row["case"].split("_")
            batch, n, rows = int(b[1:]), int(n[1:]), int(rows[4:])
            lasts = (300,500,700,900)[:batch]
            pos = torch.cat([torch.arange(lp-rows//batch+1,lp+1, device=dev) for lp in lasts])
            sr = torch.arange(batch, device=dev).repeat_interleave(rows//batch)
            m.insert_taps_graphed(sr,pos,taps[pos])
            slots = torch.arange(batch,device=dev)
            anchors = torch.tensor(row["anchors"],device=dev)
            last = torch.tensor(lasts,device=dev)
            target = torch.tensor(row["packed_reply"],device=dev,dtype=torch.float64)
            for mode in ("eager", "graph"):
                ids,conf = (m.draft if mode == "eager" else m.draft_graphed)(slots,anchors,last,5,n_out=n)
                record({"case":row["case"], "subset":args.subset,"mode":mode,
                        "ids":differences(ids,target[:,:5]),"confidence":differences(conf,target[:,5:])})
        record({"phase":"full_complete","peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
    output.close()

if __name__ == "__main__":
    main()
