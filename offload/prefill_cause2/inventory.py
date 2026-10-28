"""PF1-CAUSE2 item 1: what the PF1-CAUSE overlay changes besides timing (CPU, numbers and hashes only).

`diff`: rebuilds the reviewed cause sources from the pinned K1 bases (offload/prefill_cause/build_overlay.py) and lists
every edit at the AST level: the inserted imports, every `_pf.call` span (file, function, label, callee, metadata
calls), and a census of behaviour-relevant calls (streams, events, syncs, host reads, allocations, copies) in the base
vs the overlay outside the span wrappers. Stripping the wrappers must give the base AST (else it refuses).

`lead`: reads saved PF1 trace JSONL files and reports, per arm and rank, the full-chunk event-record count and the
host's lead over the GPU: when the host issues layer 9's and layer 19's FFN all-reduce (ms after `chunk.begin`) and
how much of the chunk's GPU time is still queued at that point (`chunk.total` minus that offset).
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import cast

from offload.prefill_cause import build_overlay as bo

# Calls whose presence changes streams, synchronization, host reads, allocation or copies. Matched on the callee's
# last attribute (`torch.cuda.current_stream().wait_stream` -> `wait_stream`).
CENSUS = ('Stream', 'stream', 'current_stream', 'wait_stream', 'wait_event', 'record_stream', 'Event', 'record',
          'synchronize', 'item', 'tolist', 'cpu', 'empty', 'zeros', 'empty_like', 'zeros_like', 'contiguous', 'clone',
          'copy_', 'to', 'memory_stats', 'empty_cache', 'data_ptr', 'is_current_stream_capturing')
FULL_ROWS = 4000                    # a full prefill chunk: 4,092 new rows (window 2/CW)
AR_ORDINALS = {'embedding': 0, 'layer0_ffn': 2, 'layer9_ffn': 20, 'layer19_ffn': 40, 'last': 80}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tail(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ast.unparse(func)


def census(tree: ast.AST, skip_spans: bool) -> dict[str, int]:
    """Counts of CENSUS calls; with skip_spans the `_pf.call(label, meta, fn, ...)` wrapper and its metadata are not
    counted (the wrapped callee and its arguments are), so base and overlay compare like for like."""
    counts: collections.Counter[str] = collections.Counter()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and ast.unparse(node.func) == '_pf.call' and skip_spans:
            for sub in [node.args[2], *node.args[3:], *(k.value for k in node.keywords)]:
                visit(sub)
            return
        if isinstance(node, ast.Call) and _tail(node) in CENSUS:
            counts[_tail(node)] += 1
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    return dict(sorted(counts.items()))


def diff() -> dict[str, object]:
    pins = cast(dict[str, str], json.loads((bo.HERE / 'BASES.json').read_text()))
    files: dict[str, object] = {}
    for path in bo.FILES:
        base_bytes = (bo.K1 / path).read_bytes()
        if sha(base_bytes) != pins[path]:
            raise ValueError(f'K1 base drifted from offload/prefill_cause/BASES.json: {path}')
        base = base_bytes.decode()
        text = bo.instrument(base, path)
        tree = ast.parse(text)
        had = {ast.unparse(n) for n in ast.parse(base).body if isinstance(n, (ast.Import, ast.ImportFrom))}
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.unparse(n) not in had]
        spans = []
        owners = {id(sub): fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                  for sub in ast.walk(fn) if isinstance(sub, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == '_pf.call':
                meta_calls = sorted({ast.unparse(c.func) for c in ast.walk(node.args[1]) if isinstance(c, ast.Call)})
                spans.append({'line': node.lineno, 'function': owners.get(id(node), '?'),
                              'label': ast.literal_eval(node.args[0]), 'callee': ast.unparse(node.args[2]),
                              'metadata_calls': meta_calls})
        spans.sort(key=lambda s: cast(int, s['line']))
        before, after = census(ast.parse(base), False), census(tree, True)
        if before != after:
            raise ValueError(f'{path}: behaviour-relevant calls differ outside the span wrappers: {before} vs {after}')
        files[path] = {'base_sha256': pins[path], 'overlay_sha256': sha(text.encode()), 'imports_added': imports,
                       'spans': spans, 'census_outside_spans': after}
    helper = (bo.HERE / 'runtime.py').read_text()
    helper_calls = sorted({ast.unparse(n.func) for n in ast.walk(ast.parse(helper)) if isinstance(n, ast.Call)})
    return {'files': files, 'helper': {'path': bo.HELPER, 'sha256': sha(helper.encode()), 'calls': helper_calls}}


def chunks(path: Path) -> list[dict[str, object]]:
    """Full chunks of one trace file: rows grouped by chunk identity (rows are written at drain, out of order)."""
    by: dict[tuple[str, ...], dict[str, object]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        members = row.get('members')
        if not members:
            continue
        key = tuple(str(m['chunk']) for m in members)
        entry = by.setdefault(key, {'rows': [], 'new': sum(int(m['new_tokens']) for m in members)})
        if row['phase'] == 'chunk.begin':
            entry['begin'] = int(row['host_start_ns'])
        elif row['phase'] == 'chunk.end':
            entry['end'] = int(row['host_end_ns'])
        else:
            cast(list[dict[str, object]], entry['rows']).append(row)
    out = [c for c in by.values() if cast(int, c['new']) >= FULL_ROWS and 'begin' in c and 'end' in c]
    for c in out:
        cast(list[dict[str, object]], c['rows']).sort(key=lambda r: cast(int, r.get('event', -1)))
    return out


def lead(paths: list[Path]) -> dict[str, object]:
    result: dict[str, object] = {}
    for path in paths:
        full = chunks(path)
        if not full:
            raise ValueError(f'no full chunks in {path}')
        spans, totals, host = [], [], []
        offsets: dict[str, list[float]] = {name: [] for name in AR_ORDINALS}
        for c in full:
            rows = cast(list[dict[str, object]], c['rows'])
            timed = [r for r in rows if 'cuda_ms' in r]
            spans.append(len(timed))
            total = [float(cast(float, r['cuda_ms'])) for r in timed if r['phase'] == 'chunk.total']
            if len(total) != 1:
                raise ValueError(f'{path}: a full chunk lacks exactly one completed chunk.total')
            totals.append(total[0])
            host.append((cast(int, c['end']) - cast(int, c['begin'])) / 1e6)
            ars = [r for r in rows if r['phase'] == 'collective.all_reduce']
            if len(ars) != 81:
                raise ValueError(f'{path}: a full chunk has {len(ars)} all-reduces, expected 81')
            for name, i in AR_ORDINALS.items():
                offsets[name].append((cast(int, ars[i]['host_start_ns']) - cast(int, c['begin'])) / 1e6)
        med = {name: statistics.median(v) for name, v in offsets.items()}
        gpu = statistics.median(totals)
        result[str(path)] = {
            'sha256': sha(path.read_bytes()), 'full_chunks': len(full),
            'event_pairs_per_chunk': statistics.median(spans), 'chunk_gpu_ms': gpu,
            'chunk_host_ms': statistics.median(host), 'ar_host_offset_ms': med,
            'host_ms_per_layer_before_backpressure': (med['layer9_ffn'] - med['layer0_ffn']) / 9,
            'gpu_ms_queued_at_layer9_ffn': gpu - med['layer9_ffn'],
            'gpu_ms_queued_at_layer19_ffn': gpu - med['layer19_ffn'],
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    dp = sub.add_parser('diff')
    lp = sub.add_parser('lead')
    lp.add_argument('--traces', type=Path, nargs='+', required=True)
    for p in (dp, lp):
        p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = diff() if args.cmd == 'diff' else lead(args.traces)
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}')
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(sha(args.output.read_bytes()))


if __name__ == '__main__':
    main()
