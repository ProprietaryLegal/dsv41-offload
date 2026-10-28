"""Prepare admission-only overlays in a NEW destination; ordinary PP4 is unchanged."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.launch.windows import pp4_prepare as base

HELPER = 'vllm/v1/worker/gpu/dsv41_pp4_admission.py'
IMPORT = 'from vllm.v1.worker.gpu import dsv41_pp4_admission as _pp4_admission\n'
# Round 16: the client seams run where RemoteDraftClient runs, i.e. in the torch-free async proxy process
# (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process), which imports remote_dspark as a top-level package by path. Reaching the
# seam through `vllm` executes vllm/__init__.py -> vllm.env_override -> `import torch` inside the proxy
# (PP4-85b-admission-W-20261027T203434Z cmd/0109: "proxy loaded torch/CUDA library"). The client therefore imports a
# byte-identical copy of the seam from its own package; engine-side seams keep the vllm path.
SEAM = 'dsv41_pp4_admission.py'
CLIENT_IMPORT = 'from . import dsv41_pp4_admission as _pp4_admission\n'
PROXY_ROOTS = ('__init__', 'proxy_proc')
FORBIDDEN = ('torch', 'vllm')


def _module_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """(absolute module, line) of every import executed at import time (TYPE_CHECKING blocks excluded)."""
    found: list[tuple[str, int]] = []
    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.extend((alias.name, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((node.module, node.lineno))
            elif isinstance(node, ast.If):
                test = node.test
                checking = isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING' or (
                    isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING')
                if not checking:
                    visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.Try, ast.With)):
                visit(node.body)
                for handler in getattr(node, 'handlers', []):
                    visit(handler.body)
                visit(getattr(node, 'orelse', []))
                visit(getattr(node, 'finalbody', []))
    visit(tree.body)
    return found


def proxy_closure(package: Path) -> dict[str, ast.Module]:
    """Every module of the package the proxy can import: package-relative imports at any depth, from the proxy roots."""
    seen: dict[str, ast.Module] = {}
    todo = list(PROXY_ROOTS)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        tree = ast.parse((package/(name+'.py')).read_text())
        seen[name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                targets = [node.module] if node.module else [a.name for a in node.names]
                todo.extend(t.split('.')[0] for t in targets if (package/(t.split('.')[0]+'.py')).is_file())
    return seen


def proxy_torch_free(package: Path) -> list[str]:
    """Module-level torch/vllm imports reachable from the async proxy (an empty list is the contract)."""
    return [f'{name}.py:{line} imports {module}' for name, tree in sorted(proxy_closure(package).items())
            for module, line in _module_level_imports(tree) if module.split('.')[0] in FORBIDDEN]


def prepare(source: Path, output: Path) -> dict[str, object]:
    receipt = base.prepare(source, output)
    runner = output/pp3.RUNNER
    text = runner.read_text()
    text = IMPORT + pp3.replace_once(text, '        pp3_valid = getattr(self.speculator, "sampled_valid", None)',
        '        _pp4_admission.invalidate(self, input_batch)\n        pp3_valid = getattr(self.speculator, "sampled_valid", None)')
    anchor = '        sampler_output, num_sampled, num_rejected = self.sample(\n            hidden_states, input_batch, grammar_output\n        )\n'
    text = pp3.replace_once(text, anchor, anchor+'        _pp4_admission.observe_sample(sampler_output)\n')
    runner.write_text(text)
    utils = output/pp3.UTILS
    utils.write_text(IMPORT + utils.read_text()+'\nrejection_sample = _pp4_admission.verify(rejection_sample)\n')
    helper = output/HELPER
    helper.write_bytes(Path(__file__).with_name('pp4_admission_engine.py').read_bytes())
    remote = output/'remote_dspark'
    shutil.copytree(base.ROOT/'offload/dspark/spark/remote_dspark', remote,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if (remote/SEAM).exists():
        raise ValueError('remote_dspark already defines the admission seam module name')
    (remote/SEAM).write_bytes(helper.read_bytes())
    client = remote/'client.py'
    text = client.read_text()
    text = pp3.replace_once(text, '        if self.disabled:\n            return StepOutcome(False, reason="disabled")',
        '        _pp4_admission.client_before(self)\n        if self.disabled:\n            return StepOutcome(False, reason="disabled")')
    text = pp3.replace_once(text, '        except TransportTimeout:\n',
        '            _pp4_admission.client_after(self, self._resp)\n        except TransportTimeout:\n')
    # Future imports must remain first.
    text = pp3.replace_once(text, 'from __future__ import annotations\n', 'from __future__ import annotations\n'+CLIENT_IMPORT)
    client.write_text(text)
    leaks = proxy_torch_free(remote)
    if leaks:
        raise ValueError('admission overlay would load torch/vllm in the async proxy: ' + '; '.join(leaks))
    paths = [output/name for name in (*pp3.PINS, base.HELPER, HELPER)] + sorted(remote.glob('*.py'))
    for path in paths:
        compile(path.read_text(), str(path), 'exec')
    receipt['admission_only'] = True
    receipt['output_sha256'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
