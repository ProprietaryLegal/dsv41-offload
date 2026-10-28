"""Bit-compare the image's persistent_topk against the fixed dsv41_topk op on SM121.

Run INSIDE the image on one Spark GPU, in a window, after jit_build.py, with production stopped and the GPU idle
(WINDOW.md step T2):

    python3 -m offload.upfix.topk.bitcompare --execute --library <built .so> --out <receipt.json>

persistent_topk picks a path per row by its length L: trivial (L <= k), short (k < L <= 8,192), medium (8,192 < L <=
32,768, #62936's recover_medium) and radix (L > 32,768, the cooperative path of #63475). Cases cover every path at
aligned strides (VEC_SIZE 4, as in production) and target each fix:
- production decode shapes: k 512, widths 163,840 and 327,680 (K1's 327,680 cap // indexer compress ratio 2 and 1),
  1-24 rows with lengths on all three paths; 524,288 (the 524,288 cap) at 1-4 rows, where a group needs more CTAs
  than are resident and the launcher hands off to the image's _C::top_k_per_row_decode;
- #62936: packed rows (> 4,096 candidates in the threshold's 11-bit bin) on the short, medium and radix paths;
- #62467: upstream's masked-row-tail trigger (aligned stride, L = k+1..k+3 or a medium L not a multiple of 4, 0 or
  100 finite values, the rest -inf or -1e30, finite padding past L), and -inf rows on the medium and radix paths;
- #63475: long rows on the radix path, repeated;
- exact ties straddling the k-th value on all three paths, and NaN rows (#63118).
Tie-free rows have one correct top-k SET and are scored exactly. Rows with ties, -inf spans or NaN are scored by
validity: every index in [0, L), min(k, L) distinct indices, the selected values equal the reference top-k values as a
multiset (NaN ranks above +inf, as torch.topk), and every finite position of a masked row selected. NaN cases run on
the fixed op only, after every other case (a pre-#63118 kernel may fault). PASS = zero bad rows for the fixed op;
image-kernel bad rows are reported (they are the bugs being fixed), never hidden. Without --execute it prints the plan.
Lengths are 1-D here; production passes (B, next_n), which the kernel reads as the same flat per-row array.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch

RADIX_THRESHOLD = 32768          # persistent_topk.cuh: radix path above this length
SHORT_THRESHOLD = 8192           # persistent_topk.cuh SHORT_EXACT_THRESHOLD: medium path above this length
WORKSPACE_BYTES = 1024 * 1024    # indexer_topk.RADIX_TOPK_WORKSPACE_SIZE
DECOY = 1.0e30                   # finite padding past each row's length; a kernel that reads it selects it
REQUIRED_ABOVE = -1.0e29         # masked rows: positions above this (the finite ones) must all be selected
# Indexer decode logits are max_model_len // compress_ratio wide (K1 attention.py); the served model's indexers use
# ratio 2 (layers 2/8/14) and ratio 1 (layers 20-36).
PROD_WIDTHS = (327_680 // 2, 327_680)
CAP524_WIDTHS = (524_288 // 2, 524_288)
KS = (512, 1024, 2048)


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    width: int          # row stride of the buffer (view_offset > 0: the view is shifted, stride unchanged)
    k: int
    lengths: str        # 'full' | 'trivial' | 'short' | 'short-hi' | 'medium' | 'radix' | 'mixed' | 'fixed'
    values: str         # 'shuffled' | 'normal' | 'packed' | 'ties' | 'masked' | 'neginf' | 'nan' | 'nan-many'
    view_offset: int    # 0, or 1 for an x[:, 1:] view (VEC_SIZE 1)
    repeats: int
    fix: str
    scoring: str = 'exact'      # 'exact' (tie-free: the unique set) | 'valid' (see module docstring)
    fixed_only: bool = False    # NaN rows: the fixed op only, after every other case
    length: int = 0             # lengths == 'fixed'
    finite: int = 0             # masked rows: finite values inside [0, L)
    fill: str = ''              # masked rows: value of every other position inside [0, L) ('-inf' | '-1e30')


def _aligned(n: int) -> int:
    return (n + 3) // 4 * 4


def cases() -> list[Case]:
    found: list[Case] = []
    for k in KS:
        for rows in (1, 4, 8, 33, 65, 128):
            # Misaligned stride or x[:, 1:] view: VEC_SIZE 1 baseline (cannot reach the float4 tail-lane bug).
            found.append(Case(f'vec1-k{k}-r{rows}', rows, 4099, k, 'mixed', 'shuffled', 1, 1, 'VEC_SIZE 1 baseline'))
            found.append(Case(f'trivial-k{k}-r{rows}', rows, 8192, k, 'trivial', 'normal', 0, 1, 'L <= k baseline'))
        found.append(Case(f'packed-radix-k{k}', 8, 65536, k, 'full', 'packed', 0, 3, '#62936/#59645 radix'))
        for rows in (4, 24):
            found.append(Case(f'packed-short-k{k}-r{rows}', rows, 8192, k, 'short-hi', 'packed', 0, 3, '#62936 short'))
            found.append(Case(f'packed-medium-k{k}-r{rows}', rows, 32768, k, 'medium', 'packed', 0, 3,
                              '#62936 recover_medium'))
        found.append(Case(f'radix-k{k}', 4, 2 * RADIX_THRESHOLD + 516, k, 'radix', 'normal', 0, 50, '#63475'))
        found.append(Case(f'radix-tail-k{k}', 4, 150_017, k, 'radix', 'packed', 1, 20, '#63475 VEC_SIZE 1'))
        for path, width in (('short', 8192), ('medium', 32768), ('radix', 65536)):
            found.append(Case(f'ties-{path}-k{k}', 4, width, k, path, 'ties', 0, 3, 'ties at the k-th value', 'valid'))
        for path, width in (('medium', 32768), ('radix', 65536)):
            found.append(Case(f'neginf-{path}-k{k}', 4, width, k, path, 'neginf', 0, 3, '#62467 -inf rows', 'valid',
                              finite=k // 2, fill='-inf'))
        # Upstream test_persistent_topk_masked_row_tail at an aligned stride, plus a medium length.
        for length in (k + 1, k + 2, k + 3, 20_001):
            for finite in (0, 100):
                for fill_name, fill in (('ninf', '-inf'), ('n1e30', '-1e30')):
                    found.append(Case(f'masked-k{k}-L{length}-f{finite}-{fill_name}', 4, _aligned(length) + 8, k,
                                      'fixed', 'masked', 0, 1, '#62467 masked row tail', 'valid',
                                      length=length, finite=finite, fill=fill))
    # Production decode shapes at the K1 cap: index_topk 512, rows up to 4 requests x 6 (1 + 5 speculative).
    for width in PROD_WIDTHS:
        for path in ('short', 'medium', 'radix'):
            found.append(Case(f'prod-w{width}-r1-{path}', 1, width, 512, path, 'normal', 0, 5, 'production shape'))
        for rows in (4, 8, 24):
            found.append(Case(f'prod-w{width}-r{rows}', rows, width, 512, 'mixed', 'normal', 0, 10, 'production shape'))
    # The 524,288 cap: ratio-2 width stays cooperative; ratio-1 width hands off to _C::top_k_per_row_decode.
    found.append(Case(f'cap524-w{CAP524_WIDTHS[0]}-r4', 4, CAP524_WIDTHS[0], 512, 'mixed', 'normal', 0, 3,
                      '524,288 cap'))
    for rows in (1, 2, 3, 4):
        found.append(Case(f'cap524-w{CAP524_WIDTHS[1]}-r{rows}', rows, CAP524_WIDTHS[1], 512, 'radix', 'normal', 0, 3,
                          '524,288 cap: launcher hand-off'))
    # NaN rows last, fixed op only.
    for k in (512, 2048):
        for path, width in (('short', 8192), ('medium', 32768), ('radix', 65536)):
            for kind in ('nan', 'nan-many'):
                found.append(Case(f'{kind}-{path}-k{k}', 4, width, k, path, kind, 0, 1, '#63118 NaN', 'valid', True))
    return found


def length_range(case: Case, mode: str) -> tuple[int, int]:
    width = case.width - case.view_offset
    ranges = {'trivial': (1, case.k - 1), 'short': (case.k + 21, SHORT_THRESHOLD),
              'short-hi': (max(case.k + 21, 4609), SHORT_THRESHOLD), 'medium': (SHORT_THRESHOLD + 1, RADIX_THRESHOLD),
              'radix': (RADIX_THRESHOLD + 1, width)}
    low, high = ranges[mode]
    high = min(high, width)
    if low > high:
        raise ValueError(f'{case.name}: no {mode} lengths fit width {width}')
    return low, high


def lengths_for(case: Case, gen: 'torch.Generator') -> 'torch.Tensor':
    import torch

    width = case.width - case.view_offset
    if case.lengths == 'full':
        return torch.full((case.rows,), width, dtype=torch.int32)
    if case.lengths == 'fixed':
        return torch.full((case.rows,), case.length, dtype=torch.int32)
    modes = ['short', 'medium', 'radix'] if case.lengths == 'mixed' else [case.lengths]
    if case.lengths == 'mixed' and width <= RADIX_THRESHOLD:
        modes = ['short', 'medium'] if width > SHORT_THRESHOLD else ['short']
    out = torch.empty(case.rows, dtype=torch.int32)
    for r in range(case.rows):
        low, high = length_range(case, modes[r % len(modes)])
        pick = int(torch.randint(low, high + 1, (1,), generator=gen))
        out[r] = pick - 1 if pick % 4 == 0 and pick - 1 >= low and case.lengths != 'trivial' else pick
    return out


def _distinct(sorted_values: 'torch.Tensor') -> 'torch.Tensor':
    """Strictly increasing fp32 from a sorted row: repeats move up by whole representable steps (vectorised)."""
    import torch

    bits = sorted_values.contiguous().view(torch.int32).to(torch.int64)
    magnitude = bits & 0x7FFFFFFF
    key = torch.where(bits < 0, -magnitude, magnitude)          # monotone integer key of the float order
    steps = torch.arange(key.numel(), dtype=torch.int64)
    key = torch.cummax(key - steps, 0).values + steps
    raw = torch.where(key < 0, (-key) | 0x80000000, key)
    raw = torch.where(raw >= 2 ** 31, raw - 2 ** 32, raw)
    return raw.to(torch.int32).view(torch.float32)


def _row(case: Case, length: int, gen: 'torch.Generator') -> 'torch.Tensor':
    import torch

    width = case.width - case.view_offset
    perm = torch.randperm(width, generator=gen)
    if case.values == 'shuffled':
        return perm.to(torch.float32) - width / 2                 # distinct integers, exact in fp32
    if case.values == 'packed':
        return (1.0 + perm.to(torch.float64) * 2.0 ** -23).to(torch.float32)   # adjacent fp32 values, one bin
    row = _distinct(torch.randn(width, generator=gen, dtype=torch.float32).sort().values)[perm]
    if case.values == 'normal':
        return row
    live = row[:length]
    if case.values == 'ties':
        order = torch.argsort(live, descending=True)
        live[order[case.k - 20: case.k + 20]] = float(live[order[case.k - 1]])   # 40 equal values straddling rank k
    elif case.values in ('masked', 'neginf'):
        keep = torch.randperm(length, generator=gen)[:case.finite]
        kept = live[keep].clone()
        live.fill_(float(case.fill))
        live[keep] = kept
    else:                                                           # nan, nan-many
        where = torch.randperm(length, generator=gen)
        if case.values == 'nan-many':
            live[where[:min(4 * case.k, 3 * length // 4)]] = float('nan')
        else:
            n, m = case.k // 4, case.k // 8
            live[where[:n]] = float('nan')
            live[where[n:n + m]] = float('inf')
            live[where[n + m:n + 2 * m]] = torch.finfo(torch.float32).max
            live[where[n + 2 * m:n + 3 * m]] = 1.0e29
    return row


def logits_for(case: Case, lengths: 'torch.Tensor', gen: 'torch.Generator') -> 'torch.Tensor':
    """Rows (stride = case.width) with DECOY past each length; a shifted view when view_offset > 0."""
    import torch

    out = torch.empty((case.rows, case.width), dtype=torch.float32)
    out[:, :case.view_offset] = -DECOY
    view = out[:, case.view_offset:]
    for r in range(case.rows):
        n = int(lengths[r])
        view[r] = _row(case, n, gen)
        view[r, n:] = DECOY
    return view


def tie_free(logits: 'torch.Tensor', lengths: 'torch.Tensor') -> bool:
    import torch

    for r in range(logits.shape[0]):
        live = torch.sort(logits[r, : int(lengths[r])]).values
        if bool((live[1:] == live[:-1]).any()) or bool(torch.isnan(live).any()):
            return False
    return True


def reference(logits: 'torch.Tensor', lengths: 'torch.Tensor', k: int) -> list[frozenset[int]]:
    import torch

    found: list[frozenset[int]] = []
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        found.append(frozenset(int(i) for i in torch.topk(logits[r, :n].double(), min(k, n)).indices.tolist()))
    return found


def same_multiset(a: 'torch.Tensor', b: 'torch.Tensor') -> bool:
    """Equal as multisets, NaN equal to NaN."""
    import torch

    if a.numel() != b.numel():
        return False
    x, y = torch.sort(a).values, torch.sort(b).values
    return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())


def row_valid(live: 'torch.Tensor', chosen: Sequence[int], k: int, require_above: float | None) -> bool:
    """Indices in [0, L), min(k, L) distinct, values = the reference top-k multiset, required positions selected."""
    import torch

    n = live.numel()
    if len(chosen) != min(k, n) or len(set(chosen)) != len(chosen) or any(not 0 <= i < n for i in chosen):
        return False
    picked = live[torch.tensor(chosen, dtype=torch.int64)]
    if not same_multiset(picked, torch.topk(live, len(chosen)).values):
        return False
    if require_above is not None:
        required = set(torch.nonzero(live > require_above).flatten().tolist())
        if not required <= set(chosen):
            return False
    return True


def score(case: Case, logits: 'torch.Tensor', lengths: 'torch.Tensor', expected: list[frozenset[int]] | None,
          output: 'torch.Tensor') -> tuple[int, int]:
    """(bad rows, rows that selected an index at or past L)."""
    bad = decoy = 0
    require = REQUIRED_ABOVE if case.values in ('masked', 'neginf') else None
    for r, row in enumerate(output.cpu().tolist()):
        n = int(lengths[r])
        chosen = [int(i) for i in row if int(i) >= 0]
        decoy += any(i >= n for i in chosen)
        if expected is not None:
            bad += frozenset(chosen) != expected[r] or len(chosen) != len(expected[r])
        else:
            bad += not row_valid(logits[r, :n], chosen, case.k, require)
    return bad, decoy


Kernel = Callable[['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', int, int], None]


def run_case(case: Case, kernels: dict[str, Kernel], device: str, seed: int) -> dict[str, object]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    lengths = lengths_for(case, gen)
    logits = logits_for(case, lengths, gen)
    expected: list[frozenset[int]] | None = None
    if case.scoring == 'exact':
        if not tie_free(logits, lengths):
            raise ValueError(f'{case.name}: generated rows are not tie-free')
        expected = reference(logits, lengths, case.k)
    dev_logits = logits.to(device) if case.view_offset == 0 else _device_view(logits, case, device)
    dev_lengths = lengths.to(device)
    max_seq_len = int(lengths.max())
    result: dict[str, object] = {'case': asdict(case), 'seed': seed, 'max_seq_len': max_seq_len,
                                 'input_sha256': hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()}
    for name, kernel in kernels.items():
        bad_rows = decoy_hits = 0
        for _ in range(case.repeats):
            output = torch.full((case.rows, case.k), -1, dtype=torch.int32, device=device)
            workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            kernel(dev_logits, dev_lengths, output, workspace, case.k, max_seq_len)
            bad, decoy = score(case, logits, lengths, expected, output)
            bad_rows += bad
            decoy_hits += decoy
        result[name] = {'bad_rows': bad_rows, 'decoy_hits': decoy_hits, 'rows_checked': case.rows * case.repeats}
    return result


def _device_view(logits: 'torch.Tensor', case: Case, device: str) -> 'torch.Tensor':
    """Recreate the shifted-base view on the device (same stride as the full buffer)."""
    import torch

    full = torch.empty((case.rows, case.width), dtype=torch.float32)
    full[:, case.view_offset:] = logits
    full[:, : case.view_offset] = -DECOY
    return full.to(device)[:, case.view_offset:]


def bad_rows(results: Sequence[dict[str, object]], kernel: str) -> int:
    return sum(int(cast(dict[str, int], r[kernel])['bad_rows']) for r in results if kernel in r)


def verdict(results: Sequence[dict[str, object]], fixed: str = 'fixed') -> str:
    if any(fixed not in r for r in results):
        raise ValueError(f'every case must run the {fixed!r} kernel')
    return 'PASS' if bad_rows(results, fixed) == 0 else 'FAIL'


def write_receipt(out: Path, receipt: Mapping[str, object]) -> None:
    """Atomic: a crash mid-write leaves the previous receipt, never a truncated one."""
    temp = out.with_name(out.name + '.partial')
    temp.write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    os.replace(temp, out)


def run_all(plan: Sequence[Case], kernels: Mapping[str, Kernel], device: str, seed: int, out: Path,
            header: Mapping[str, object]) -> dict[str, object]:
    """Run every case, rewriting the receipt after each one. Until the last case finishes the receipt says
    complete False and verdict INCOMPLETE, so a crash or a timeout (even in the final NaN cases) keeps every earlier
    result for the report and still fails safe: configs.verify_topk refuses an incomplete receipt."""
    results: list[dict[str, object]] = []
    receipt: dict[str, object] = {**header, 'cases': len(plan), 'cases_done': 0, 'complete': False,
                                  'verdict': 'INCOMPLETE', 'results': results, 'image_kernel_bad_rows': 0}
    write_receipt(out, receipt)
    for i, case in enumerate(plan):
        chosen = {'fixed': kernels['fixed']} if case.fixed_only else dict(kernels)
        try:
            results.append(run_case(case, chosen, device, seed + i))
        except BaseException as error:
            receipt['error'] = f'{case.name}: {type(error).__name__}: {error}'
            write_receipt(out, receipt)
            raise
        receipt.update(cases_done=len(results), image_kernel_bad_rows=bad_rows(results, 'image'))
        write_receipt(out, receipt)
    receipt.update(complete=True, verdict=verdict(results))
    write_receipt(out, receipt)
    return receipt


def execute(library: Path, out: Path, device: str = 'cuda:0', seed: int = 41) -> dict[str, object]:
    import torch

    importlib.import_module('vllm._custom_ops')  # registers torch.ops._C.* (inside the image only)
    cast(Callable[[str], None], torch.ops.load_library)(str(library))
    both: dict[str, Kernel] = {'image': torch.ops._C.persistent_topk, 'fixed': torch.ops.dsv41_topk.persistent_topk}
    header: dict[str, object] = {
        'schema': 'dsv41-upfix-topk-bitcompare-v3', 'library': str(library),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'device': torch.cuda.get_device_name(device), 'capability': list(torch.cuda.get_device_capability(device)),
        'torch': torch.__version__}
    return run_all(cases(), both, device, seed, out, header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--library', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps([asdict(c) for c in cases()], indent=1))
        return
    if args.library is None or args.out is None or args.out.exists():
        parser.error('--library and a fresh --out are required with --execute')
    receipt = execute(args.library, args.out)
    print(receipt['verdict'], 'image kernel bad rows:', receipt['image_kernel_bad_rows'])
    raise SystemExit(0 if receipt['verdict'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
