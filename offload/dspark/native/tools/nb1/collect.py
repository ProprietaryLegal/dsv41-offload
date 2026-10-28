"""Reduce NB1 GPU logs (one ``RESULT_JSON`` line each) to committed aggregate evidence and a Markdown summary.

No tensors, token IDs, prompts or text are present in the logs; the evidence keeps counts, error magnitudes and
sha256 digests only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

Result = dict[str, Any]
# A batch gap is "within the operation's own numerical error" when mean|alone - packed| <= sqrt(2) x
# mean|alone - FP64|: two evaluations whose errors against the exact result are independent and equally large
# differ by about sqrt(2) x that error. Above it, the batch change adds error beyond the op's own rounding.
WITHIN = 2 ** 0.5


def load(log: Path) -> Result:
    lines = [line[len('RESULT_JSON '):] for line in log.read_text().splitlines() if line.startswith('RESULT_JSON ')]
    if len(lines) != 1:
        raise ValueError(f'{log}: expected one RESULT_JSON line, found {len(lines)}')
    exit_file = log.with_name(log.name + '.exit')
    if exit_file.exists() and exit_file.read_text().strip() != '0':
        raise ValueError(f'{log}: job exit {exit_file.read_text().strip()}')
    result: Result = json.loads(lines[0])
    result['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
    return result


def family(label: str) -> str:
    """'r1.ffn2.dense:mtp.2.attn.wo_b#0' -> 'ffn2.dense:attn.wo_b' (rank and occurrence folded)."""
    body = label.split('.', 1)[1].split('#')[0]
    return re.sub(r':mtp\.\d+\.', ':', body).replace('[r0]', '').replace('[r1]', '')


def fold(cells: list[dict[str, Any]], key_index: int = 0) -> dict[tuple[str, ...], dict[str, Any]]:
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for c in cells:
        key = list(c['key'])
        fam = family(key[key_index])
        k = (fam, *[str(x) for x in key[1:]])
        cur = out.setdefault(k, dict(comparisons=0, exact=0, max_abs=0.0, max_rel=0.0, differing=0, elements=0))
        cur['comparisons'] += c['comparisons']
        cur['exact'] += c['exact']
        cur['max_abs'] = max(cur['max_abs'], c['max_abs'])
        cur['max_rel'] = max(cur['max_rel'], c['max_abs_over_alone_max'])
        cur['differing'] += c['differing_elements']
        cur['elements'] += c['elements']
    return out


def fp64_summary(cells: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        groups[(family(c['label']), *[c[k] for k in keys])].append(c)
    rows = []
    for k, cs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rows.append(dict(op=k[0], **{name: k[i + 1] for i, name in enumerate(keys)}, cells=len(cs),
                         gap_max=max(c['gap_max'] for c in cs),
                         alone_err_max=max(c['alone_err_max'] for c in cs),
                         gap_mean_over_alone_err_mean=max(c['gap_mean'] / c['alone_err_mean']
                                                          if c['alone_err_mean'] > 0 else float('inf') for c in cs),
                         within_own_error=sum(c['alone_err_mean'] > 0 and c['gap_mean'] <= WITHIN * c['alone_err_mean']
                                              for c in cs),
                         packed_err_mean_over_alone=max(c['packed_err_mean'] / c['alone_err_mean']
                                                        if c['alone_err_mean'] > 0 else float('inf') for c in cs)))
    return rows


def summarize(r: Result) -> Result:
    iso = {k: v for k, v in fold(r['draft_isolated']).items() if v['exact'] != v['comparisons']}
    e2e = fold(r['draft_e2e'])
    ins_iso = {k: v for k, v in fold(r['insert_isolated']).items() if v['exact'] != v['comparisons']}
    return dict(
        label='MEASURED', arm=r['arm'], invariant_env=r['invariant_env'], log_sha256=r['log_sha256'],
        mode=r['mode'] + '; bc62df2a at 225 W under flock -n; no physical pair', repo=r['repo'],
        fixtures=r['fixtures'], batches=r['batches'], final=r['final'], depth_transitions=r['depth_transitions'],
        same_batch_companion_order=r['same_batch_companion_order'], graph_vs_eager=r['graph_vs_eager'],
        first_divergent_e2e=r['first_divergent_e2e'], first_divergent_isolated=r['first_divergent_isolated'],
        isolated_batch_dependent_ops=[dict(op=k[0], kind=k[1], B=int(k[2]), **v) for k, v in sorted(iso.items())],
        isolated_ops_checked=len(fold(r['draft_isolated'])),
        e2e_by_op=[dict(op=k[0], kind=k[1], B=int(k[2]), **v) for k, v in sorted(e2e.items())],
        head_isolated=r['head_isolated'], confidence_matmul_isolated=r['confidence_matmul_isolated'],
        fp64=fp64_summary(r['fp64'], ('kind', 'B')),
        insert_first_divergent=r['insert_first_divergent'], insert_ring=r['insert_ring'],
        insert_graph_vs_eager=r['insert_graph_vs_eager'],
        insert_isolated_batch_dependent_ops=[dict(op=k[0], n=int(k[1]), B=int(k[2]), **v)
                                             for k, v in sorted(ins_iso.items())],
        insert_fp64=fp64_summary(r['insert_fp64'], ('n', 'B')),
        insert_alone_fp64=r.get('insert_alone_fp64', {}),
        dense_configs=r['dense_configs'], b1_digests=r['b1_digests'], output_digest=r.get('output_digest'),
        runtime_sources_sha256=hashlib.sha256(json.dumps(r['runtime_sources'], sort_keys=True).encode()).hexdigest(),
        runtime_source_files=len(r['runtime_sources']), prebuilt_extensions_sha256=r['prebuilt_extensions_sha256'],
        triton_compiled_keys=r['triton_compiled_keys'], torch_version=r['torch_version'], seconds=r['seconds'],
        peak_allocated_bytes=r['peak_allocated_bytes'])


def markdown(s: Result) -> str:
    out = [f"### {s['arm']} (invariant env {s['invariant_env']}) — MEASURED", '',
           '|scope|B|kind|comparisons|IDs equal per position|conf bit-equal|conf max abs|', '|---|---|---|---|---|---|---|']
    for key in sorted(s['final']):
        kind, b, mode = key.split('/')
        v = s['final'][key]
        out.append(f"|{mode}|{b[1:]}|{kind}|{v['comparisons']}|{','.join(map(str, v['ids_exact']))}|"
                   f"{v['conf_exact']}|{v['conf_max_abs']:.6g}|")
    out += ['', '|kind/B|first divergent op, end to end (placements)|', '|---|---|']
    for k, v in sorted(s['first_divergent_e2e'].items()):
        out.append(f"|{k}|{'; '.join(f'{op} ×{n}' for op, n in sorted(v.items(), key=lambda kv: -kv[1]))}|")
    out += ['', '|op (isolated, alone inputs at packed shape)|kind|B|inexact/comparisons|max abs|max abs / alone max|',
            '|---|---|---|---|---|---|']
    for c in s['isolated_batch_dependent_ops']:
        out.append(f"|{c['op']}|{c['kind']}|{c['B']}|{c['comparisons'] - c['exact']}/{c['comparisons']}|"
                   f"{c['max_abs']:.4g}|{c['max_rel']:.3g}|")
    out += ['', '|op (FP64)|kind|B|cells|gap max|alone err max|max gap/err (mean)|within own error (gap/err <= sqrt 2)|'
            'max packed/alone err (mean)|', '|---|---|---|---|---|---|---|---|---|']
    for c in s['fp64']:
        out.append(f"|{c['op']}|{c['kind']}|{c['B']}|{c['cells']}|{c['gap_max']:.4g}|{c['alone_err_max']:.4g}|"
                   f"{c['gap_mean_over_alone_err_mean']:.3g}|{c['within_own_error']}/{c['cells']}|"
                   f"{c['packed_err_mean_over_alone']:.3g}|")
    out += ['', f"Insertion ring exact: {json.dumps(s['insert_ring'])}",
            f"Insertion first divergent: {json.dumps(s['insert_first_divergent'])}", '']
    for c in s['insert_isolated_batch_dependent_ops']:
        out.append(f"- insert isolated {c['op']} n={c['n']} B={c['B']}: {c['comparisons'] - c['exact']}/"
                   f"{c['comparisons']} inexact, max abs {c['max_abs']:.4g}")
    for c in s['insert_fp64']:
        out.append(f"- insert FP64 {c['op']} n={c['n']} B={c['B']}: gap max {c['gap_max']:.4g}, alone err max "
                   f"{c['alone_err_max']:.4g}, within {c['within_own_error']}/{c['cells']}")
    return '\n'.join(out) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    for log in args.logs:
        s = summarize(load(log))
        name = log.name.removesuffix('.log')
        (args.out_dir / f'{name}.json').write_text(json.dumps(s, indent=1, sort_keys=True) + '\n')
        md.append(markdown(s))
    (args.out_dir / 'SUMMARY.md').write_text('\n'.join(md))


if __name__ == '__main__':
    main()
