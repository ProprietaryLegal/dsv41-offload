"""Offline PP4 overlay preparation, composed on PP3's exact pinned probability verifier."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3

HELPER = 'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py'
K1_SHA256 = '682eb895b8fb3d68b9580954b6cda255a49f882bddb4b06342d7d96258eb6e62'
ROOT = Path(__file__).resolve().parents[3]
K1_RUNNER = ROOT / 'offload/kvpack/overlay' / pp3.RUNNER


def instrument(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    hits: set[str] = set()
    edits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ('execute_model', 'sample_tokens'):
            if node.name in hits:
                raise ValueError('PP4 duplicate model-runner timing anchor')
            hits.add(node.name)
            decorator = 'execution' if node.name == 'execute_model' else 'sampling'
            at = min([node.lineno, *(x.lineno for x in node.decorator_list)]) - 1
            edits.append((at, ' ' * node.col_offset + '@_pp4_round.' + decorator + '\n'))
    if hits != {'execute_model', 'sample_tokens'}:
        raise ValueError('PP4 model-runner timing anchors changed')
    for at, insert in sorted(edits, reverse=True):
        lines.insert(at, insert)
    result = 'from vllm.v1.worker.gpu import dsv41_pp4_round_trace as _pp4_round\n' + ''.join(lines)
    compile(result, '<pp4-round-overlay>', 'exec')
    return result


def prepare(source: Path, output: Path, k1_runner: Path = K1_RUNNER) -> dict[str, object]:
    if output.exists():
        raise ValueError('output must not exist; refuse overwriting any prepared overlay')
    manifest = json.loads((ROOT / 'offload/kvpack/MANIFEST.json').read_text())
    entries = [entry for entry in manifest['files'] if entry['container_path'].endswith('/' + pp3.RUNNER)]
    if len(entries) != 1 or entries[0]['overlay_sha256'] != K1_SHA256:
        raise ValueError('reviewed K1 manifest runner pin changed')
    runner_data = k1_runner.read_bytes()
    utils_data = (source / pp3.UTILS).read_bytes()
    pins = {pp3.RUNNER: K1_SHA256, pp3.UTILS: pp3.PINS[pp3.UTILS]}
    for name, data in ((pp3.RUNNER, runner_data), (pp3.UTILS, utils_data)):
        if hashlib.sha256(data).hexdigest() != pins[name]:
            raise ValueError('PP4 pinned source mismatch: ' + name)
    # Compose the narrow probability/failure mask and timing additions onto
    # the FULL production K1 source. Never replace it with the old image runner.
    texts = {pp3.RUNNER: instrument(pp3.runner(runner_data.decode())),
             pp3.UTILS: pp3.utilities(utils_data.decode()),
             HELPER: Path(__file__).with_name('pp4_round_trace.py').read_text()}
    for name, text in texts.items():
        compile(text, name, 'exec')
    output.mkdir(parents=True)
    for name, text in texts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    outputs = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in texts}
    receipt: dict[str, object] = {'abi': pp3.ABI, 'source_sha256': pins, 'production_k1_sha256': K1_SHA256,
        'output_sha256': outputs, 'round_scope': 'execute_model through sample_tokens completed wall/CUDA',
        'profiled_only': True}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
