"""CPU preparation for W10; private prompt material is never printed.

prepare sends only idle-gated /tokenize calls. Reference generation is separate.
The historical request is recovered from AST, never by running the W5 driver.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import urllib.request

from .decode_dataset import digest, mapping, number
from .decode_smoke import token_ids
from .replay_window import CAMPAIGN, METRICS, gauges, require

PRIVATE = CAMPAIGN/'W10/private'
W5 = CAMPAIGN/'W5/greedy_check.py'
K1_COMMAND = CAMPAIGN/'W7/k1_prod_327.sh'
K1_CAPTURE = CAMPAIGN/'K1-production/greedy-post.jsonl'
PROMPTS = Path('/opt/dsv41/data/d2/prompts/prompts.jsonl')
MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'


def save(path: Path, value: object) -> None:
    require(path.resolve().is_relative_to(PRIVATE.resolve()), 'private W10 output required')
    require(not path.exists(), 'refusing to overwrite private artifact: '+path.name)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    path.chmod(0o600)


def historical_request(script: Path, capture: Path, row: int) -> dict[str, object]:
    tree = ast.parse(script.read_text())
    prompts = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'PROMPTS' for t in n.targets))
    values = ast.literal_eval(prompts)
    require(isinstance(values, list) and 0 <= row < len(values), 'invalid historical row')
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    require(len(rows) == len(values) and all(r['i'] == i for i, r in enumerate(rows)),
            'historical capture ordering differs from W5')
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    body = next(n.value for n in ast.walk(run) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            require(node.id == 'p', 'unexpected dynamic historical request field')
            return ast.Constant(value=values[row])

    request = mapping(ast.literal_eval(Substitute().visit(body)))
    require(request.get('model') == MODEL and request.get('temperature') == 0
            and request.get('max_tokens') == 300
            and request.get('chat_template_kwargs') == {'thinking': False}, 'W5 request contract changed')
    return request


class Production:
    """Only the two preparation endpoints; generation requires a separate class."""
    def metrics(self) -> str:
        with urllib.request.urlopen(METRICS, timeout=3) as response:
            return response.read().decode()

    def tokenize(self, body: dict[str, object]) -> list[int]:
        require(gauges(self.metrics()), 'production busy; no tokenize call sent')
        request = urllib.request.Request(METRICS.removesuffix('/metrics')+'/tokenize',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=15) as response:
            result = mapping(json.load(response))
        require(gauges(self.metrics()), 'foreign work appeared; stop tokenization')
        ids = token_ids(result.get('tokens'))
        require(result.get('count') == len(ids) and len(ids) <= 8192, 'invalid/oversize server tokenization')
        return ids


def prepare(io: Production, row: int = 4) -> dict[str, object]:
    os.umask(0o077)
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(PRIVATE.stat().st_mode & 0o777 == 0o700, 'W10/private must be mode 700')
    require(not (PRIVATE/'smoke-input.json').exists(), 'inputs already prepared')
    request = historical_request(W5, K1_CAPTURE, row)
    command = K1_COMMAND.read_text()
    require('$PY $C/W5/greedy_check.py $U $P/greedy-post.jsonl' in command
            and 'P=$C/K1-production' in command, 'original K1 invocation not verified')
    selected: list[dict[str, object]] = []
    with PROMPTS.open() as f:
        for line in f:
            p = mapping(json.loads(line))
            if p.get('source') == 'book' and p.get('split') == 'dev' and number(p['est_tokens']) <= 8192:
                selected.append(p)
                if len(selected) == 2:
                    break
    require(len(selected) == 2, 'two bounded public D2 book prompts unavailable')
    rows: list[dict[str, object]] = []
    for i, p in enumerate(selected):
        tok = {'model': MODEL, 'messages': p['messages'], 'add_generation_prompt': True,
               'chat_template_kwargs': p['chat_template_kwargs']}
        ids = io.tokenize(tok)
        body: dict[str, object] = {'model': MODEL, 'max_tokens': 128, 'temperature': 1,
            'top_p': 1, 'priority': 20, 'stream': False, 'return_token_ids': True,
            'request_id': f'd2c.smoke{i}.0.{max(0,len(ids)-128)}.{len(ids)}'}
        if i == 0:
            body['prompt'] = ids
        else:
            body.update(messages=p['messages'], chat_template_kwargs=p['chat_template_kwargs'],
                        logprobs=True, top_logprobs=20, return_tokens_as_token_ids=True)
        rows.append({'endpoint': '/v1/completions' if i == 0 else '/v1/chat/completions',
                     'body': body, 'prompt_ids': ids})
    ids = io.tokenize({'model': MODEL, 'messages': request['messages'], 'add_generation_prompt': True,
                       'chat_template_kwargs': request['chat_template_kwargs']})
    rows.append({'endpoint': '/v1/chat/completions',
                 'body': dict(request, stream=False, return_token_ids=True), 'prompt_ids': ids})
    proof = {'schema': 'd2.k1-request-identity.v1', 'profile': 'K1-327680',
        'capture_sha256': digest(K1_CAPTURE), 'row': row, 'verified': True,
        'endpoint': '/v1/chat/completions', 'request': request,
        'identity_evidence': {str(p): digest(p) for p in (W5, K1_COMMAND)},
        'historical_defaults': {'top_p': 1, 'priority': 0, 'stream': False},
        'timing_qualified': False}
    proof_path = PRIVATE/'k1-request-identity.json'
    save(proof_path, proof)
    inputs = {'schema': 'd2.decode-smoke-input.v1', 'requests': rows, 'greedy_reference': {
        'path': str(K1_CAPTURE), 'sha256': digest(K1_CAPTURE), 'row': row,
        'request_receipt': str(proof_path), 'request_receipt_sha256': digest(proof_path)}}
    save(PRIVATE/'smoke-input.json', inputs)
    provenance = {'schema': 'd2.smoke-input-provenance.v1', 'public_only': True,
        'd2_prompts_path': str(PROMPTS), 'd2_prompts_sha256': digest(PROMPTS),
        'selected': [{k: p[k] for k in ('pid','source','source_docs','doc_ids','prompt_sha256')} for p in selected],
        'prompt_tokens': [len(token_ids(r['prompt_ids'])) for r in rows],
        'identity_sha256': digest(proof_path), 'inputs_sha256': digest(PRIVATE/'smoke-input.json')}
    save(PRIVATE/'input-provenance.json', provenance)
    return {'prepared': True, 'requests': 3, 'public_only': True,
            'prompt_tokens': provenance['prompt_tokens'], 'timing_qualified': False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--row', type=int, default=4)
    args = ap.parse_args()
    print(json.dumps(prepare(Production(), args.row)))


if __name__ == '__main__':
    main()
