"""Build PF1-CAUSE2's fix candidate: the PF1-CAUSE MoE-half span positions with event records only. Default OFF = K1 bytes.

The same three K1 files and the same 25 anchors as offload/prefill_cause/build_overlay.py (its ANCHORS, matched the
same way and the same number of times) get `_mf.call(rows, fn, *args)` instead of `_pf.call(label, metadata, fn,
*args)`: no label, no metadata, no trace runtime. `rows` is the pass's row count already in scope (`positions.shape[0]`
in the decoder layer, `m` in tf_prompt.routed, `R` in tf_experts.routed). The helper (fence.py) records the CUDA events
only with VLLM_DSV41_MOE_FENCE=1, so the mounted sources run production's path with the switch off.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import cast

from offload.prefill_cause import build_overlay as bo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
K1 = bo.K1
FILES = bo.FILES
HELPER = 'vllm/v1/worker/gpu/dsv41_moe_fence.py'
IMPORT = 'from vllm.v1.worker.gpu import dsv41_moe_fence as _mf'
ROWS = {'vllm/models/deepseek_v41/nvidia/model.py': 'positions.shape[0]', 'vllm_exl3_moe/tf_prompt.py': 'm',
        'vllm_exl3_moe/tf_experts.py': 'R'}


def instrument(text: str, path: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    futures = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__']
    first = tree.body[0]
    docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    if futures or docstring:
        at = offsets[(futures[-1] if futures else first).end_lineno or 0]
    else:
        at = offsets[first.lineno - 1]
    edits: list[tuple[int, int, str]] = [(at, at, IMPORT + '\n')]
    anchors = bo.ANCHORS[path]
    owners = bo._owner(tree, path)
    hits = {key: 0 for key in anchors}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) not in owners:
            continue
        callee = ast.unparse(node.func)
        if callee == 'mhc_shifted_post_pre':
            pre = next((k.value for k in node.keywords if k.arg == 'pre_mix'), None)
            callee += '@' + (ast.unparse(pre) if pre is not None else '')
        key = (owners[id(node)], callee)
        if key not in anchors:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[cast(int, node.func.end_lineno) - 1] + cast(int, node.func.end_col_offset)
        opening = text.index('(', end)
        has_args = bool(node.args or node.keywords)
        edits.append((start, opening + 1, f'_mf.call({ROWS[path]}, {text[start:end]}' + (', ' if has_args else '')))
        hits[key] += 1
    for key, (_, _, count) in anchors.items():
        if hits[key] != count:
            raise ValueError(f'PF1-CAUSE2 anchor {path}:{key[0]}:{key[1]} found {hits[key]} times, expected {count}')
    edits.sort(reverse=True)
    last = len(text) + 1
    for start, end, replacement in edits:
        if end > last:
            raise ValueError(f'PF1-CAUSE2 overlapping edits in {path}')
        text = text[:start] + replacement + text[end:]
        last = start
    compile(text, path, 'exec')
    return text


def build(output: Path, enabled: bool = False) -> dict[str, str]:
    pins = cast(dict[str, str], json.loads((HERE / 'BASES.json').read_text()))
    for forbidden in (K1, REPO / 'offload/prefill_trace/generated', REPO / 'offload/prefill_cause/generated',
                      Path('/opt/dsv41/data/src-image')):
        if output.resolve().is_relative_to(forbidden.resolve()):
            raise ValueError('PF1-CAUSE2 output must not modify a reference, production, PF1-R2 or PF1-CAUSE tree')
    sources: dict[str, bytes] = {}
    for path in FILES:
        data = (K1 / path).read_bytes()
        if bo.sha(data) != pins[path]:
            raise ValueError(f'PF1-CAUSE2 base SHA mismatch: {path}')
        sources[path] = instrument(data.decode(), path).encode() if enabled else data
    if enabled:
        sources[HELPER] = (HERE / 'fence.py').read_bytes()
    hashes: dict[str, str] = {}
    for path, data in sources.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'PF1-CAUSE2 refuses existing different output: {target}')
        target.write_bytes(data)
        hashes[path] = bo.sha(data)
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--enabled', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build(args.output_root, args.enabled), indent=2))


if __name__ == '__main__':
    main()
