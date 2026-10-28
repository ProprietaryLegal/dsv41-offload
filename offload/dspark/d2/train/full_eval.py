"""Full-population reference G2, checkpointed after each request. No pilot sample."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import tempfile
from pathlib import Path
import random
from typing import TypedDict, cast

import torch

from .data import D2Blocks, sha256
from .full_admission import full_run, pilot_recipe, require_full_pilot
from .model import load
from .pilot import RECIPES, Recipe
from .pilot_eval import Metrics, score
from .pilot_sample import complete_rounds, key_hash
from .pilot_verdict import finite, validate_metrics
from .fast_policy import BACKEND, numeric_sources
from .replay import apply_trainables, read_checkpoint

SPLITS = ('dev', 'heldout_legal', 'heldout_general')


class Pair(TypedDict):
    baseline: Metrics
    candidate: Metrics
    k2_sum: int


def write_json(path: Path, value: object) -> None:
    """Exclusive, atomic and fsynced full-run receipt/request boundary."""
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, suffix='.partial', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # exclusive; never overwrite a completed receipt
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink()


def completed(adapter: Path, manifest: Path, pilot_hash: str, recipe: "Recipe" = "kl") -> str:
    state, digest = read_checkpoint(adapter)
    data = D2Blocks(manifest, 'train')
    expected = full_run(recipe)
    if (state.get('full_recipe') != recipe or state.get('run') != asdict(expected) or state.get('dataset') != data.fingerprint
            or state.get('step') != 3*math.ceil(len(data)/128)
            or state.get('fast_training') is not True or state.get('fast_pilot_sha256') != pilot_hash
            or state.get('fast_backend') != BACKEND or state.get('fast_numeric_sources') != numeric_sources()
            or state.get('reference_math') is not False
            or len(state.get('curve', [])) != state.get('step')
            or any(not bool(torch.isfinite(p).all()) for p in state['trainable'].values())):
        raise ValueError('full G2 requires the completed three-epoch qualified FAST checkpoint')
    finite(state['curve'])
    return digest


def summarize(pairs: list[Pair]) -> dict[str, object]:
    """Original G2 request bootstrap, using exact per-request sufficient totals."""
    if not pairs:
        raise ValueError('empty full G2 population')
    for p in pairs:
        a, b = p['candidate'], p['baseline']
        finite(p)
        validate_metrics(a, b)
        validate_metrics(b, b)
        if a['request_totals'][0]['request_sha256'] != b['request_totals'][0]['request_sha256']:
            raise ValueError('full G2 paired request identity mismatch')
        if (a['requests'] != 1 or b['requests'] != 1 or a['rounds'] != b['rounds'] or a['rounds'] < 1
                or a['population_sha256'] != b['population_sha256']
                or len(a['kl_by_position']) != 5 or len(b['kl_by_position']) != 5
                or not 0 <= p['k2_sum'] <= 5*a['rounds']):
            raise ValueError('full G2 paired population mismatch')
    if len({p['candidate']['request_totals'][0]['request_sha256'] for p in pairs}) != len(pairs):
        raise ValueError('duplicate full G2 requests')
    n = sum(p['candidate']['rounds'] for p in pairs)
    gains = [(p['candidate']['rounds'], p['candidate']['request_totals'][0]['accepted_sum']-p['k2_sum']) for p in pairs]
    rng = random.Random(23)
    boots = []
    for _ in range(10000):
        sample = [rng.choice(gains) for _ in gains]
        boots.append(sum(g for _, g in sample)/sum(r for r, _ in sample))
    boots.sort()
    return {'requests': len(pairs), 'rounds': n,
        'candidate': sum(p['candidate']['request_totals'][0]['accepted_sum'] for p in pairs)/n,
        'k2': sum(p['k2_sum'] for p in pairs)/n,
        'gain': sum(g for _, g in gains)/n,
        'native_gain': sum(p['candidate']['request_totals'][0]['accepted_sum']-p['baseline']['request_totals'][0]['accepted_sum'] for p in pairs)/n,
        'ci95_request_cluster': [boots[250], boots[9750]],
        'worst_request_gain': min(g/r for r, g in gains),
        'kl_candidate': [sum(p['candidate']['kl_by_position'][i]*p['candidate']['rounds'] for p in pairs)/n for i in range(5)],
        'kl_baseline': [sum(p['baseline']['kl_by_position'][i]*p['baseline']['rounds'] for p in pairs)/n for i in range(5)]}


def verdict(parts: list[dict[str, object]]) -> dict[str, object]:
    by = {cast(str, p['split']): p for p in parts}
    if len(parts) != 3 or set(by) != set(SPLITS):
        raise ValueError('full G2 requires exactly dev/legal/general parts')
    identities = ('manifest_sha256', 'adapter_sha256', 'pilot_sha256', 'source_sha256', 'recipe')
    if any(k not in p for p in parts for k in identities):
        raise ValueError('full G2 parts missing required identity/recipe')
    if any(any(p[k] != parts[0][k] for k in identities) for p in parts):
        raise ValueError('full G2 parts identity mismatch')
    if any(p.get('schema') != 'dsv41.d2.full-g2-part.v1' for p in parts):
        raise ValueError('wrong full G2 part schema')
    if any(p.get('recipe') not in RECIPES for p in parts):
        raise ValueError('full G2 requires explicit qualified recipe')
    finite(parts)
    for part in parts:
        m = cast(dict[str, object], part['metrics'])
        if (cast(int, m['rounds']) < 1 or part['rounds'] != m['rounds']
                or len(cast(list[float], m['kl_candidate'])) != 5
                or len(cast(list[float], m['kl_baseline'])) != 5
                or len(cast(list[float], m['ci95_request_cluster'])) != 2):
            raise ValueError('incomplete full G2 summary coverage')
    # Amendment 4: FINAL G2 is unchanged, so heldout_general keeps the D2-TRAIN G2 gates of evaluate.g2
    # (>= 10 requests, gain >= -0.03, worst request, KL); amendment 7's "information" is the pilot/G1(b) population.
    dev, legal, general = [cast(dict[str, object], by[s]['metrics']) for s in SPLITS]
    held = (dev, legal, general)
    checks = {'dev_requests_eq30': dev['requests'] == 30, 'legal_requests_ge20': cast(int, legal['requests']) >= 20,
        'general_requests_ge10': cast(int, general['requests']) >= 10,
        'legal_gain_ge015': cast(float, legal['gain']) >= .15,
        'general_gain_ge_minus003': cast(float, general['gain']) >= -.03,
        'no_native_regression': all(cast(float, m['native_gain']) >= -.01 for m in (dev, legal)),
        'legal_ci_positive': cast(list[float], legal['ci95_request_cluster'])[0] > 0,
        'every_request_ge_minus02': all(cast(float, m['worst_request_gain']) >= -.2 for m in held),
        'kl_lower_every_position': all(a < b for m in held for a, b in
            zip(cast(list[float], m['kl_candidate']), cast(list[float], m['kl_baseline'])))}
    return {'schema': 'dsv41.d2.full-g2.v1', 'verdict': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks, **{k: parts[0][k] for k in identities}, 'parts': by,
        'role': 'full development-set G2 (D2-TRAIN G2 on dev, heldout_legal and heldout_general); fresh reserved TEST and G3 still required'}


def evaluate(a: argparse.Namespace) -> None:
    from .safety import assert_gpu_lease
    assert_gpu_lease()
    pilot = json.loads(a.pilot_receipt.read_text())
    pilot_hash = require_full_pilot(a.pilot_receipt, pilot['plan']['frozen_identity'], a.manifest,
        a.parity_receipt, a.g1_receipt, 128, 8192, 32, a.pilot_source,
        {"dev": a.k2_dev, "heldout_legal": a.k2_legal} if getattr(a, "k2_dev", None) and getattr(a, "k2_legal", None) else None)
    recipe = pilot_recipe(a.pilot_receipt, pilot_hash)
    adapter_hash = completed(a.adapter, a.manifest, pilot_hash, recipe)
    state, _ = read_checkpoint(a.adapter)
    data = D2Blocks(a.manifest, a.split)
    rounds = complete_rounds(data, a.k2)
    keys = [k for _, k in rounds]
    from .pilot import sources
    from .evaluate import match_length
    contract = {'manifest_sha256': sha256(a.manifest), 'adapter_sha256': adapter_hash,
        'pilot_sha256': pilot_hash, 'recipe': recipe, 'source_sha256': sources(), 'split': a.split,
        'k2_sha256': sha256(a.k2), 'population_sha256': key_hash(keys), 'rounds': len(keys)}
    if a.out.exists():
        raise FileExistsError('full G2 part already complete; preserve result')
    journal = a.out.with_suffix('.requests')
    if journal.exists() and not a.resume:
        raise FileExistsError('full G2 journal exists; use --resume with unchanged inputs')
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = journal/'contract.json'
    if identity.exists():
        if json.loads(identity.read_text()) != contract:
            raise ValueError('full G2 resume contract changed')
    else:
        write_json(identity, contract)
    # Never let candidate predictions choose which rounds are evaluated.
    requests = sorted({k['request'] for k in keys})
    model = load(str(a.ckpt), 'cuda', rank=32, reference_math=True).eval()
    if (model.frozen_identity != pilot['plan']['frozen_identity']
            or state.get('frozen_identity') != model.frozen_identity
            or state.get('model_config') != asdict(model.c)):
        raise ValueError('full G2 base differs from pilot')
    from .pilot import digest
    from .replay import g1_drafts
    drafts = g1_drafts(a.k2, data)
    pairs: list[Pair] = []
    baseline_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    for request in requests:
        part = journal/(digest(request)+'.json')
        if part.exists():
            pair = cast(Pair, json.loads(part.read_text()))
        else:
            selected = [k for k in keys if k['request'] == request]
            apply_trainables(model, baseline_state)
            baseline = score(model, data, a.k2, selected)
            apply_trainables(model, state['trainable'])
            candidate = score(model, data, a.k2, selected)
            k2_sum = 0
            for index, key in rounds:
                if key['request'] == request:
                    accepted = match_length(drafts[request][key['P']], data[index]['targets']['labels'][0].tolist())
                    if accepted is None:
                        raise ValueError('incomplete full G2 K2 round')
                    k2_sum += accepted
            pair = {'baseline': baseline, 'candidate': candidate, 'k2_sum': k2_sum}
            write_json(part, pair)
        if pair['candidate']['population_sha256'] != key_hash([k for k in keys if k['request'] == request]):
            raise ValueError('full G2 journal request coverage changed')
        pairs.append(pair)
        print(json.dumps({'requests_complete': len(pairs), 'requests_total': len(requests)}), flush=True)
    if sha256(a.adapter) != adapter_hash:
        raise ValueError('full G2 adapter changed during scoring')
    write_json(a.out, {'schema': 'dsv41.d2.full-g2-part.v1', **contract, 'metrics': summarize(pairs)})


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    merge = sub.add_parser('merge')
    merge.add_argument('--parts', type=Path, nargs=3, required=True)
    merge.add_argument('--out', type=Path, required=True)
    ev = sub.add_parser('score')
    for name in ('manifest', 'adapter', 'ckpt', 'pilot-receipt', 'pilot-source', 'parity-receipt', 'g1-receipt', 'k2', 'out'):
        ev.add_argument('--'+name, type=Path, required=True)
    ev.add_argument('--k2-dev', type=Path)
    ev.add_argument('--k2-legal', type=Path)
    ev.add_argument('--split', choices=SPLITS, required=True)
    ev.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.command == 'merge':
        write_json(a.out, verdict([json.loads(path.read_text()) for path in a.parts]))
    else:
        if (a.k2_dev is None) != (a.k2_legal is None):
            p.error('provide both --k2-dev and --k2-legal')
        torch.set_num_threads(8)
        evaluate(a)


if __name__ == '__main__':
    main()
