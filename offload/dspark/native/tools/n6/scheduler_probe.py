"""Execute the actual image scheduler's adaptive-window method on CPU stubs."""
from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast


def probe(source: Path) -> dict[str, object]:
    content = source.read_text()
    tree = ast.parse(content)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == '_dsv41_adaptive_k']
    if len(methods) != 1:
        raise ValueError('expected one actual scheduler adaptive method')
    namespace: dict[str, object] = dict(math=math, _DSV41_AK_MIN=2,
                                      _DSV41_AK_MARGIN=1.0, _DSV41_AK_HI=1.6)
    body: list[ast.stmt] = list(methods)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    method = cast(Callable[[SimpleNamespace, dict[str, int], int], int], namespace['_dsv41_adaptive_k'])
    cases = []
    for name, emas, prefill, cap, expected in (
        ('solo-below-threshold', [1.59], False, 5, 3),
        ('packed-low-ema', [1.59, 1.0], False, 5, 3),
        ('packed-high-companion', [1.59, 1.6], False, 5, 5),
        ('b3-low-ema', [1.59, 1.0, 1.0], False, 4, 3),
        ('b3-high-companion', [1.59, 1.0, 1.6], False, 4, 4),
        ('unknown-companion', [1.59, None], False, 5, 5),
        ('prefill-companion', [1.59, 1.0], True, 5, 5),
    ):
        requests = {str(i): SimpleNamespace(_dsv41_ak_ema=ema, is_prefill_chunk=prefill and i == len(emas)-1)
                    for i, ema in enumerate(emas)}
        actual = method(SimpleNamespace(requests=requests), {i: 1 for i in requests}, cap)
        if actual != expected:
            raise AssertionError(f'actual scheduler {name}: {actual} != expected {expected}')
        cases.append(dict(case=name, cap=cap, chosen=actual, expected=expected))
    return dict(label='MEASURED CPU execution of actual scheduler method; configuration examples, not performance',
                source_sha256=hashlib.sha256(content.encode()).hexdigest(), cases=cases,
                interpretation='Cross-request verification-window coupling is intentional policy, '
                               'not drafter row/window coupling. W8 aggregate acceptance is not per-request EMA.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(probe(args.source), indent=2)+'\n')


if __name__ == '__main__':
    main()
