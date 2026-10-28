"""Build a hash-pinned diagnostic overlay. Default OFF emits byte-identical bases."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FILES = {
    'vllm/v1/worker/gpu/model_runner.py': 'k1',
    'vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py': 'k1',
    'vllm/model_executor/layers/sparse_attn_indexer.py': 'image',
    'vllm/v1/attention/backends/mla/indexer.py': 'image',
    'vllm/distributed/parallel_state.py': 'image',
}
HELPER = 'vllm/v1/worker/gpu/dsv41_prefill_trace.py'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def instrument(text: str, path: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits: list[tuple[int, int, str]] = []
    # Imports after __future__ if present; these bases have no futures, but support them.
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__']
    at = offsets[imports[-1].end_lineno or 0] if imports else offsets[tree.body[0].end_lineno or 0]
    edits.append((at, at, '\nfrom vllm.v1.worker.gpu import dsv41_prefill_trace as _pf\n'))
    targets = {
        'cp_gather_indexer_k_quant_cache': 'indexer.gather',
        'fp8_fp4_mqa_logits': 'indexer.mqa',
        'top_k_per_row_prefill': 'indexer.topk',
        '_select_candidate_blocks': 'indexer.candidate_select',
        '_apply_candidate_mask': 'indexer.candidate_mask',
        'exchange_topk': 'collective.indexer_exchange',
        'compute_global_topk_indices_and_lens': 'sparse.metadata_scan',
        'sparse_attention': 'sparse.dispatch.triton',
        'flashinfer_trtllm_batch_decode_sparse_mla_dsv4': 'sparse.dispatch.flashinfer',
        'prepare_attn': 'metadata.prepare',
        'build_prefill_chunk_metadata': 'metadata.indexer_chunk',
    }
    hits: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            if name in targets:
                # Replace only the call prefix, leaving arguments and their evaluation intact.
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.func.end_lineno - 1] + node.func.end_col_offset  # type: ignore[operator]
                original_fn = text[start:end]
                opening = text.index('(', end)
                has_args = bool(node.args or node.keywords)
                edits.append((start, opening + 1,
                    f'_pf.call({targets[name]!r}, _pf.meta(locals()), {original_fn}' + (', ' if has_args else '')))
                hits[name] = hits.get(name, 0) + 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == 'execute_model' and path.endswith('model_runner.py'):
                pos = offsets[node.lineno - 1] + node.col_offset
                edits.append((pos, pos, '@_pf.execution\n' + ' ' * node.col_offset))
            if node.name == '_split_indexer_prefill_chunks' and path.endswith('/mla/indexer.py'):
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if len(returns) != 1:
                    raise ValueError('PF1 unexpected planner returns')
                ret = returns[0]
                start = offsets[ret.lineno - 1] + ret.col_offset
                end = offsets[ret.end_lineno - 1] + ret.end_col_offset  # type: ignore[operator]
                edits.append((start, end, 'return _pf.plan(chunks, compressed_seq_lens_cpu, prefill_query_lens_cpu, workspace_size, max_logits_bytes, request_offset)'))
            if path.endswith('parallel_state.py') and node.name in ('all_reduce', 'all_gather', 'reduce_scatter', 'all_gatherv', 'reduce_scatterv') and node.args.args and node.args.args[0].arg == 'self':
                # Decorate the body through a span, preserving returns, exceptions and original signature.
                body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body
                start = offsets[body[0].lineno - 1]
                end = offsets[node.end_lineno or node.lineno]
                original = text[start:end]
                indent = ' ' * body[0].col_offset
                # span implementation below uses current-stream events, inclusive of transfer/wait.
                edits.append((start, end, indent + f'with _pf.span("collective.{node.name}", _pf.meta(locals())):\n' + ''.join('    ' + ln if ln.strip() else ln for ln in original.splitlines(keepends=True))))
    if path.endswith('model_runner.py'):
        anchor = '        attn_metadata = None\n'
        if text.count(anchor) != 1:
            raise ValueError('PF1 unexpected runner anchor')
        pos = text.index(anchor)
        edits.append((pos, pos, '        _pf.begin(input_batch, dummy_run, batch_desc.cg_mode, self.max_model_len)\n'))
        # Time the whole metadata+model interval: start at begin; close in execution finally.
    # Verify expected anchors so a foreign/new source cannot silently lose instrumentation.
    required = {'model_runner.py': ('prepare_attn',), 'flashinfer_sparse.py': ('flashinfer_trtllm_batch_decode_sparse_mla_dsv4',),
                'sparse_attn_indexer.py': ('cp_gather_indexer_k_quant_cache', 'fp8_fp4_mqa_logits', 'top_k_per_row_prefill'),
                'indexer.py': ('build_prefill_chunk_metadata',)}
    for key in required.get(Path(path).name, ()):
        if key not in hits:
            raise ValueError(f'PF1 missing instrumentation anchor {path}:{key}')
    edits.sort(reverse=True)
    last_start = len(text) + 1
    for start, end, replacement in edits:
        if end > last_start:
            raise ValueError(f'PF1 overlapping instrumentation edits in {path}')
        text = text[:start] + replacement + text[end:]
        last_start = start
    compile(text, path, 'exec')
    return text


def build(output: Path, image: Path, enabled: bool = False) -> dict[str, str]:
    pins = cast(dict[str, str], json.loads((HERE / 'BASES.json').read_text()))
    if output.resolve().is_relative_to(image.resolve()) or output.resolve().is_relative_to((REPO / 'offload/kvpack/overlay').resolve()):
        raise ValueError('PF1 output must not modify reference/production overlay')
    sources: dict[str, bytes] = {}
    for path, source in FILES.items():
        base = (REPO / 'offload/kvpack/overlay' if source == 'k1' else image) / path
        data = base.read_bytes()
        if sha(data) != pins[path]:
            raise ValueError(f'PF1 trace base SHA mismatch: {base}')
        sources[path] = instrument(data.decode(), path).encode() if enabled else data
    if enabled:
        sources[HELPER] = (HERE / 'runtime.py').read_bytes()
    hashes: dict[str, str] = {}
    for path, data in sources.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'PF1 refuses existing different output: {target}')
        target.write_bytes(data)
        hashes[path] = sha(data)
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, default=Path('/opt/dsv41/data/src-image'))
    ap.add_argument('--enabled', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build(args.output_root, args.source_root, args.enabled), indent=2))


if __name__ == '__main__':
    main()
