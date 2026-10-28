"""Cheap source/CPU smoke for the composed startup hooks, before HOLD.

This executes real overlay expressions using the module's own stdlib bindings.
It cannot qualify CUDA, the model load, TP startup or numerical identity.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

from offload.prefill_maxlen.evidence import Json


def _mentions(node: ast.AST, value: str) -> bool:
    return any(isinstance(n, ast.Constant) and n.value == value for n in ast.walk(node))


def check(tree: Path) -> dict[str, Json]:
    source = ast.parse((tree / 'vllm/v1/worker/gpu/model_runner.py').read_text())
    # Do not supply os from a fixture: the original compact patch failed here.
    imports: list[ast.stmt] = []
    for node in source.body:
        if isinstance(node, ast.Import) and any(a.name == 'os' for a in node.names):
            imports.append(node)
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            imports.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), '<TV1 actual stdlib imports>', 'exec'), namespace)
    runner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
    execute = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_model')
    init = next(n for n in execute.body if isinstance(n, ast.If) and _mentions(n.test, 'VLLM_TV1_COMPACT'))
    for dummy in (True, False):
        namespace['dummy_run'] = dummy
        result = eval(compile(ast.Expression(body=init.test), '<TV1 actual init condition>', 'eval'), namespace)
        if not isinstance(result, bool) or (dummy and result):
            raise ValueError('TV1 compact dummy-run init isolation changed')
    # Saved OPS1/failed-window env has the ordinary confidence depth policy.
    # Exercise the actual mode guards against that source-derived configuration;
    # this is still a CPU fixture, not an assertion about a future live process.
    policy = SimpleNamespace(mode='conf', calib=False, lookup_opt='0', ctx_pts=[], file=None)
    namespace['self'] = SimpleNamespace(pcp_manager=None, dcp_size=1, dp_size=1,
        pp_handler=None, adaptive_verification=None, depth_policy=policy, lora_config=None,
        batch_sharder=None, speculator=type('RemoteDSparkCapacitySpeculator', (), {})())
    helper = ast.parse((tree / 'vllm_exl3_moe/tv1_compact.py').read_text())
    validate = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'require_depth_policy')
    exec(compile(ast.Module(body=[validate], type_ignores=[]), '<TV1 actual depth guard>', 'exec'), namespace)
    depth_call = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'require_depth_policy')
    eval(compile(ast.Expression(body=depth_call), '<TV1 production depth guard>', 'eval'), namespace)
    reject = next(n for n in ast.walk(init) if isinstance(n, ast.If)
                  and any(isinstance(child, ast.Raise) for child in n.body))
    if eval(compile(ast.Expression(body=reject.test), '<TV1 actual mode guard>', 'eval'), namespace):
        raise ValueError('TV1 compact rejects the recorded production plain conf configuration')
    # Validate the actual threshold parser, without creating a CUDA stream/event.
    threshold_expr = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'float' and _mentions(n, 'VLLM_TV1_COMPACT_TAU'))
    threshold = eval(compile(ast.Expression(body=threshold_expr), '<TV1 threshold>', 'eval'), namespace)
    if not isinstance(threshold, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('TV1 invalid compact init threshold')
    # CUDA graph dummy batches have no BatchReqState. Check the actual added
    # dispatch expression leaves eager dispatch unchanged in that case.
    eager = next(n.value for n in ast.walk(execute) if isinstance(n, ast.keyword) and n.arg == 'need_eager')
    namespace.update(is_profile=False, skip_compiled=False, batch_req_state=None)
    if eval(compile(ast.Expression(body=eager), '<TV1 dummy graph dispatch>', 'eval'), namespace) is not False:
        raise ValueError('TV1 compact changes dummy graph dispatch')
    state = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BatchReqState')
    # Resolve the NamedTuple itself as the module does, with postponed annotations
    # so no engine/NumPy/torch imports or allocations are needed for this probe.
    exec(compile('from __future__ import annotations\n' + ast.unparse(state), '<TV1 BatchReqState>', 'exec'), namespace)
    defaults = eval('BatchReqState._field_defaults', namespace)
    if defaults.get('tv1_compacted') is not False:
        raise ValueError('TV1 compact BatchReqState default changed')
    return {'type': 'tv1-compact-init-source-smoke-v1', 'classification': 'SOURCE/CPU ONLY',
        'module_stdlib_bindings': True, 'real_warmup_condition': True,
        'recorded_production_conf_guard': True,
        'dummy_init_skipped': True, 'dummy_graph_dispatch_unchanged': True,
        'threshold': threshold, 'cuda_or_engine_init_qualified': False}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    print(json.dumps(check(parser.parse_args().tree)))


if __name__ == '__main__':
    main()
