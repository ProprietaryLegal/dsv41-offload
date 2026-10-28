"""Build PF1-CAUSE's MoE-half spans, an add-on to the PF1-R2 trace overlay. Default OFF emits byte-identical bases.

Three K1 files gain narrow call spans through the existing PF1-R2 runtime (`dsv41_prefill_trace`, unchanged bytes), so
the rows join the same chunk context and JSONL: the decoder layer's attention call, FFN pre-mix and FFN (MoE) call
(`nvidia/model.py`), the TensorFold prompt call (`tf_prompt.py`) and each TensorFold prompt kernel launch
(`tf_experts.py` routed()). A small helper adds host-only allocator counters to the prompt call's row. Nothing else is
changed: arguments, their evaluation order, return values, streams and allocations are the original ones (CPU tests
strip the spans and compare the whole AST, and run the real routed() against a recording kernel stub).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
K1 = REPO / 'offload/kvpack/overlay'
FILES = ('vllm/models/deepseek_v41/nvidia/model.py', 'vllm_exl3_moe/tf_prompt.py', 'vllm_exl3_moe/tf_experts.py')
HELPER = 'vllm/v1/worker/gpu/dsv41_prefill_cause.py'
IMPORT = 'from vllm.v1.worker.gpu import dsv41_prefill_trace as _pf'
IMPORT_HELPER = 'from vllm.v1.worker.gpu import dsv41_prefill_cause as _pc'

LAYER_META = "{'layer': self.layer_idx, 'rows': positions.shape[0]}"
KERNEL_META = "{'rows': R, 'slots_total': P}"
# Per file: (enclosing function, callee source) -> (label, metadata source). Every anchor must be found exactly as
# often as listed, so a changed base cannot silently lose a span.
ANCHORS: dict[str, dict[tuple[str, str], tuple[str, str, int]]] = {
    'vllm/models/deepseek_v41/nvidia/model.py': {
        ('forward', 'self.attn'): ('attn.total', LAYER_META, 1),
        ('forward', 'mhc_shifted_post_pre@attn_pre'): ('mhc.ffn_pre', LAYER_META, 1),
        ('forward', 'self.ffn'): ('moe.ffn', LAYER_META, 1),
        ('forward', 'self.ffn.forward_unfinalized'): ('moe.ffn_unfinalized', LAYER_META, 1),
    },
    'vllm_exl3_moe/tf_prompt.py': {
        ('routed', 'T.routed'): ('moe.routed.tf', "_pc.alloc(rows=m, slots=k, xd_ptr=s.xd.data_ptr())", 2),
    },
    'vllm_exl3_moe/tf_experts.py': {
        ('routed', 'ext.group_count'): ('tf.group_count', KERNEL_META, 1),
        ('routed', 'ext.group_place'): ('tf.group_place', KERNEL_META, 1),
        ('routed', 'ext.group'): ('tf.group', KERNEL_META, 1),
        ('routed', 'ext.rot_in'): ('tf.rot_in', "{'rows': R, 'slots_total': P, 'xg_ptr': s.xg.data_ptr()}", 1),
        ('routed', 'ext.work_list'): ('tf.work_list', KERNEL_META, 1),
        ('routed', 'ext.grouped_mma3'): ('tf.grouped_mma3.gateup', KERNEL_META, 1),
        ('routed', 'ext.grouped_mma2'): ('tf.grouped_mma2', KERNEL_META, 2),
        ('routed', 'ext.grouped_mma'): ('tf.grouped_mma', KERNEL_META, 2),
        ('routed', 'ext.grouped_rows'): ('tf.grouped_rows', KERNEL_META, 2),
        ('routed', 'ext.grouped'): ('tf.grouped', KERNEL_META, 2),
        ('routed', 'ext.gateup_epilogue'): ('tf.gateup_epilogue', KERNEL_META, 1),
        ('routed', 'ext.grouped_down3'): ('tf.grouped_down3', KERNEL_META, 1),
        ('routed', 'ext.combine_y'): ('tf.combine_y', KERNEL_META, 1),
        ('routed', 'ext.down_epilogue'): ('tf.down_epilogue', KERNEL_META, 1),
        ('routed', 'ext.down_combine'): ('tf.down_combine', "{'rows': R, 'slots_total': P, 'out_ptr': out.data_ptr()}", 1),
    },
}
# A grouped launch reads gate/up's rotated rows (s.xg) or the down input (s.xd): that names its projection.
PROJECTION = {'s.xd': 'down', 's.xg': 'gateup'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _owner(tree: ast.Module, path: str) -> dict[int, str]:
    """Call node id -> anchored function name. model.py: DeepseekV4DecoderLayer.forward; the others: module-level."""
    scope: list[ast.stmt] = tree.body
    if path.endswith('model.py'):
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV4DecoderLayer']
        if len(classes) != 1:
            raise ValueError('PF1-CAUSE expects exactly one DeepseekV4DecoderLayer')
        scope = classes[0].body
    wanted = {fn for fn, _ in ANCHORS[path]}
    owners: dict[int, str] = {}
    for fn in scope:
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            if fn.name in owners.values():
                raise ValueError(f'PF1-CAUSE duplicate anchored function {fn.name} in {path}')
            owners.update({id(sub): fn.name for sub in ast.walk(fn) if isinstance(sub, ast.Call)})
    return owners


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
        at = offsets[first.lineno - 1]       # after the licence comments, before the first statement
    header = IMPORT + '\n' + (IMPORT_HELPER + '\n' if path.endswith('tf_prompt.py') else '')
    edits: list[tuple[int, int, str]] = [(at, at, header)]
    anchors = ANCHORS[path]
    owners = _owner(tree, path)
    hits: dict[tuple[str, str], int] = {key: 0 for key in anchors}
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
        label, meta, _ = anchors[key]
        if label in ('tf.grouped_mma2', 'tf.grouped_mma', 'tf.grouped_rows', 'tf.grouped'):
            label += '.' + PROJECTION[ast.unparse(node.args[0])]
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[cast(int, node.func.end_lineno) - 1] + cast(int, node.func.end_col_offset)
        opening = text.index('(', end)
        has_args = bool(node.args or node.keywords)
        edits.append((start, opening + 1, f'_pf.call({label!r}, {meta}, {text[start:end]}' + (', ' if has_args else '')))
        hits[key] += 1
    for key, (_, _, count) in anchors.items():
        if hits[key] != count:
            raise ValueError(f'PF1-CAUSE anchor {path}:{key[0]}:{key[1]} found {hits[key]} times, expected {count}')
    edits.sort(reverse=True)
    last = len(text) + 1
    for start, end, replacement in edits:
        if end > last:
            raise ValueError(f'PF1-CAUSE overlapping instrumentation edits in {path}')
        text = text[:start] + replacement + text[end:]
        last = start
    compile(text, path, 'exec')
    return text


def build(output: Path, enabled: bool = False) -> dict[str, str]:
    pins = cast(dict[str, str], json.loads((HERE / 'BASES.json').read_text()))
    for forbidden in (K1, REPO / 'offload/prefill_trace/generated', Path('/opt/dsv41/data/src-image')):
        if output.resolve().is_relative_to(forbidden.resolve()):
            raise ValueError('PF1-CAUSE output must not modify a reference, production or PF1-R2 overlay tree')
    sources: dict[str, bytes] = {}
    for path in FILES:
        data = (K1 / path).read_bytes()
        if sha(data) != pins[path]:
            raise ValueError(f'PF1-CAUSE base SHA mismatch: {path}')
        sources[path] = instrument(data.decode(), path).encode() if enabled else data
    if enabled:
        sources[HELPER] = (HERE / 'runtime.py').read_bytes()
    hashes: dict[str, str] = {}
    for path, data in sources.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'PF1-CAUSE refuses existing different output: {target}')
        target.write_bytes(data)
        hashes[path] = sha(data)
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--enabled', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build(args.output_root, args.enabled), indent=2))


if __name__ == '__main__':
    main()
