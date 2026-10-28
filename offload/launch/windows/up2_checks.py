"""UP2 safety checks and informational measurements; stdlib, no remote actions."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

CAP = 327680
KV = 2502600
CONFIG = 'offload/launch/configs/UP2-low-k1.env'
WORKER = 'fb68c5291792ab90c0f468bd94010917967799f3'
NATIVE_CHECKS = (
    'rank_q_kept_position_valid_anchor_owner_agreement', 'final_need_draft_real_complete',
    'source_kv_writes_retained', 'step_generation_echo_and_uniform_broadcast',
    'near_cap_k4_k5_sync_async_clamp', 'remaining129_256_257_and_leftover64_128_peer_progress',
    'partial_full_hits_async_restore_real_suffix', 'cancel_slot_reuse_preempt_backtrack',
    'disconnect_before_after_final_watchdog_late_reply', 'mixed_rejected_rows_mm_logprobs_exclusions',
    'native_dispatch_unchanged', 'no_local_drafter',
)
STOP_LOG = re.compile(
    r'PARTIAL_WINDOW|0075 .*?(?:mismatch|invalid|incomplete|reservation)|'
    r'replay audit.*(?:invalid|incomplete|mismatch|lacks)|'
    r'EngineDeadError|EngineCore.*(?:failed|exception)|Traceback \(most recent call last\)|'
    r'invalid replay tap provenance|replay skip would remove taps|'
    r'CUDA error|out of memory|\bNaN\b|\binf\b|identity.*(?:mismatch|refus)|'
    r'missing.*source.KV|stale.generation.*usable', re.I)
STOP_COUNTERS = ('fallback_internal_error', 'fallback_replay_suffix_incomplete',
                 'replay_suffix_incomplete', 'local_fallback_rounds', 'watchdog_fires',
                 'proxy_respawns', 'proxy_dead', 'provenance_errors', 'partial_window')


def screen_record(data: dict[str, Json], screen: bool) -> dict[str, Json]:
    """SCREEN is an explicit non-adoptable evidence label, independent of timing."""
    return {**data, 'screen': True, 'adoptable': False, 'qualification_level': 'SCREEN'} if screen else data


def read(path: Path) -> dict[str, Json]:
    return record(cast(Json, json.loads(path.read_text())))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError('STOP: ' + message)


def integer(value: Json) -> int:
    result = number(value)
    require(result.is_integer(), 'noninteger value')
    return int(result)


def rows(value: Json) -> list[dict[str, Json]]:
    require(isinstance(value, list), 'expected rows')
    return [record(v) for v in cast(list[Json], value)]


def strings(value: Json) -> list[str]:
    require(isinstance(value, list) and all(isinstance(v, str) for v in value), 'expected strings')
    return cast(list[str], value)


def clean_logs(text: str) -> None:
    found = STOP_LOG.search(text)
    require(found is None, f'engine/replay/audit error: {found.group(0) if found else ""}')


def counters(data: dict[str, Json], audit: int = 0, before: dict[str, Json] | None = None) -> None:
    cs = record(data['counters'])
    previous = record(before['counters']) if before is not None else {}
    for key, value in cs.items():
        n = number(value)
        require(n >= 0, 'negative counter')
        if key in STOP_COUNTERS or any(s in key.lower() for s in ('provenance_error', 'partial_window')):
            require(n == number(previous.get(key, 0)), f'new {key}={n}')
    require(data.get('proxy_torch_loaded') is False, 'proxy torch status absent/true')
    require(data.get('policy') == 'capacity' and data.get('async') is True and
            record(data['async_proxy']).get('mode') == 'process' and
            record(data['async_handoff']).get('mlock') == 'ok', 'process proxy/capacity/mlock failure')
    watchdog = record(data['async_watchdog'])
    previous_watchdog = record(before['async_watchdog']) if before is not None else {}
    require(number(watchdog['fires']) >= number(previous_watchdog.get('fires', 0)), 'watchdog fire count reset')
    require(watchdog.get('running') is True and data.get('async_dead') is not True, 'async proxy not healthy')
    require(number(cs.get('rounds_ok', 0)) > 0, 'no successful drafting rounds')
    # Engine-side audit counters publish every1024 rounds. The native receipt
    # proves audited rounds; never demand that an old engine snapshot be fresh.


def window_deadlines(hold: str, now: float) -> tuple[float, float, float]:
    import datetime
    require('coordinator' in hold, 'HOLD is not owned by coordinator')
    opened = datetime.datetime.fromisoformat(hold.split()[0].replace('Z', '+00:00')).timestamp()
    require(opened <= now < opened+75*60, 'existing HOLD submission window expired/future; never renew it')
    return opened, opened+75*60, opened+90*60


def memory(text: str) -> tuple[int, int]:
    fields = {line.split(':', 1)[0]: int(line.split()[1]) for line in text.splitlines()
              if line.startswith(('MemAvailable:', 'SwapFree:'))}
    require(set(fields) == {'MemAvailable', 'SwapFree'}, 'missing memory telemetry')
    require(fields['MemAvailable'] >= 3 * 1024**2, 'MemAvailable below 3 GiB')
    return fields['MemAvailable'], fields['SwapFree']


def native(packet: dict[str, Json], arm: str, ids: list[str], manifest: str,
           root: Path, *, complete: bool) -> int:
    require(packet.get('schema') == 'up2-native-v1' and packet.get('classification') == 'MEASURED',
            'native qualification absent; NOT RUN/HOLD')
    require(packet.get('arm') == arm and packet.get('container_ids') == ids,
            'native evidence belongs to another arm/start')
    require(packet.get('manifest_sha256') == manifest, 'native source drift')
    q = integer(packet['q'])
    require(128 < q <= 4096, 'native measured q outside policy')
    evidence = rows(packet['evidence'])
    require(bool(evidence), 'native evidence paths missing')
    for entry in evidence:
        path = root / str(entry['path'])
        require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), 'unsafe/missing native evidence')
        require(digest(path) == entry['sha256'], 'native evidence hash drift')
    if complete:
        require(integer(packet['audited_rounds']) > 0, 'no native audited rounds')
        counts = accounting(rows(packet['request_events']))
        require(number(counts['http_opened']) > 0 and counts['unresolved'] == 0,
                'native request accounting missing/unresolved')
        checks = record(packet['checks'])
        for name in NATIVE_CHECKS:
            require(checks.get(name) == 'PASS', f'native {name}: {checks.get(name, "NOT RUN")}; HOLD')
        require(packet.get('tail_sizes') == [36*q+t for t in (1, 127, 128, 129)], 'tail sizes not from measured q')
    return q


def numerical(a: dict[str, Json], aa: dict[str, Json], b: dict[str, Json]) -> dict[str, Json]:
    """Top20+OTHER, sampled-first order independent. Only matched isolated shapes."""
    def compare(x: dict[str, Json], y: dict[str, Json]) -> tuple[float, float]:
        p, q = record(x['probabilities']), record(y['probabilities'])
        keys = p.keys() & q.keys()
        common = min(sum(number(p[k]) for k in keys), sum(number(q[k]) for k in keys))
        px = [number(p[k]) for k in sorted(keys)]
        qx = [number(q[k]) for k in sorted(keys)]
        require(all(0 <= v <= 1 for v in px+qx) and sum(px) <= 1+1e-6 and sum(qx) <= 1+1e-6,
                'invalid captured probabilities')
        px.append(max(0.0, 1-sum(px)))
        qx.append(max(0.0, 1-sum(qx)))
        kl = sum(v * math.log(v/w) if w > 0 else math.inf for v,w in zip(px,qx) if v > 0)
        require(math.isfinite(kl), 'nonfinite coarse KL')
        return kl, common
    floor, aa_mass = compare(a, aa)
    kl, mass = compare(a, b)
    require(min(aa_mass, mass) >= .99, 'numerical captured common mass < .99; full-logit diagnosis required')
    require(kl <= max(.005, 2*floor), 'numerical floor exceeded; full-logit diagnosis required')
    return {'aa_kl': floor, 'on_off_kl': kl, 'common_mass': mass, 'bound': max(.005, 2*floor)}


def cold(row: dict[str, Json], n: int, rep: int) -> None:
    require(row.get('kind') == 'cold' and row.get('target_ctx') == n and row.get('rep') == rep, 'wrong matrix cell')
    usage = record(row['usage'])
    require(row.get('cap') == CAP and usage.get('prompt_tokens') == n and row.get('cached_tokens') == 0,
            'exact prompt/cold cache/cap failure')
    require(row.get('uncached_prompt_tokens') == n, 'uncached count wrong')
    require(row.get('sampling') == {'temperature':1, 'top_p':1, 'seed':42, 'priority':10, 'max_tokens':1500},
            'sampling drift')
    require(row.get('finish') in ('stop', 'length') and integer(usage['completion_tokens']) > 0,
            'incomplete cold request')
    for key in ('prompt_sha256', 'cache_salt_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', str(row.get(key))) is not None, 'missing pairing hash')
    # Timing, including missing/unusable samples, is informational.


def timing(value: Json) -> float | None:
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def measured_contexts(contexts: tuple[int, ...]) -> None:
    require(bool(contexts) and len(set(contexts)) == len(contexts) and
            set(contexts) <= {150000, 300000}, 'contexts must be a unique nonempty subset of 150000/300000')


def summarize(cells: dict[str, list[dict[str, Json]]],
              contexts: tuple[int, ...] = (150000, 300000)) -> dict[str, Json]:
    measured_contexts(contexts)
    all_rows = [r for rs in cells.values() for r in rs]
    require(set(cells) <= {'off-a', 'on', 'off-b'}, 'unknown measured arm')
    windows = {str(r.get('window_id', '')) for r in all_rows}
    require(not all_rows or (len(windows) == 1 and '' not in windows), 'missing/mixed window identity; no splicing')
    for arm, rs in cells.items():
        keys = [(r.get('target_ctx'), r.get('rep')) for r in rs]
        require(all(n in contexts and rep in (0, 1, 2) for n, rep in keys), 'cell outside measured contexts/repetitions')
        require(len(set(keys)) == len(keys), f'duplicate measured cell in {arm}')
    result: dict[str, Json] = {'classification':'MEASURED', 'decision':'HOLD; no automatic adoption'}
    result['contexts'] = list(contexts)
    result['window_id'] = next(iter(windows), None)
    result['stage_b'] = {'status':'INCOMPLETE', 'context':300000}
    points: list[Json] = []
    for n in contexts:
        samples: dict[str, list[float | None]] = {}
        ratios: list[float] = []
        spreads: list[float] = []
        for arm in ('off-a', 'on', 'off-b'):
            selected = [r for r in cells.get(arm, []) if r.get('target_ctx') == n]
            samples[arm] = [timing(r.get('ttft_s')) for r in selected]
        for rep in range(3):
            matched = [[r for r in cells.get(arm, []) if r.get('target_ctx') == n and r.get('rep') == rep]
                       for arm in ('off-a', 'on', 'off-b')]
            if not all(len(rs) == 1 for rs in matched):
                continue
            a,b,c = (rs[0] for rs in matched)
            require(a['prompt_sha256'] == b['prompt_sha256'] == c['prompt_sha256'] and
                    a['cache_salt_sha256'] == b['cache_salt_sha256'] == c['cache_salt_sha256'], 'paired hash mismatch')
            times = [timing(r.get('ttft_s')) for r in (a,b,c)]
            if all(t is not None for t in times):
                x,y,z = cast(list[float],times)
                ratios.append(y / statistics.mean([x,z]))
                spreads.append(abs(x-z) / statistics.mean([x,z]))
        if n == 300000 and len(ratios) == 3:
            gain, spread = 1-statistics.median(ratios), statistics.median(spreads)
            result['stage_b'] = {'status':'PASS' if gain > 0 and gain > spread else 'FAIL',
                                 'context':n, 'gain':gain, 'spread':spread,
                                 'paired_spreads':cast(Json, spreads)}
        points.append({'context': n, 'samples_s':cast(Json, samples), 'paired_on_over_off':cast(Json, ratios),
                       'ratio_median': statistics.median(ratios) if ratios else None,
                       'ratio_min':min(ratios) if ratios else None, 'ratio_max':max(ratios) if ratios else None,
                       'matched_usable_timing_reps':len(ratios), 'expected_reps':3,
                       'timing_samples_missing_or_invalid':sum(v is None for vs in samples.values() for v in vs)})
    result['points'] = points
    result['completed_cells'] = sum(len(rs) for rs in cells.values())
    result['expected_cells'] = 9*len(contexts)
    return result


def accounting(events: list[dict[str, Json]]) -> dict[str, Json]:
    intents = {str(r['id']) for r in events if r.get('event') == 'send_intent'}
    sent = {str(r['id']) for r in events if r.get('event') == 'http_opened'}
    completed = {str(r['id']) for r in events if r.get('event') == 'completed'}
    failed = {str(r['id']) for r in events if r.get('event') == 'failed'}
    canceled = {str(r['id']) for r in events if r.get('event') == 'canceled'}
    require(completed <= sent <= intents and failed <= intents and canceled <= sent and
            not completed & (failed|canceled) and not failed & canceled, 'inconsistent ledger')
    return {'transport_attempts':len(intents), 'http_opened':len(sent), 'completed':len(completed),
            'failed':len(failed), 'canceled':len(canceled), 'unresolved':len(intents-completed-failed-canceled),
            'dispatch_uncertain':len(intents-sent), 'planned_unsent_not_counted':True}


def runtime(prod: dict[str, Json], live: dict[str, Json], replay: int, audit: int) -> None:
    require(live.get('image') == prod.get('image') and isinstance(live.get('image'),str), 'runtime image mismatch')
    require(live.get('files') == prod.get('files') and live.get('engine') == prod.get('engine'), 'target/engine identity drift')
    require(bool(record(live['engram_identity'])) and live.get('engram_identity') == prod.get('engram_identity'),
            'Engram shim/source/manifest/JIT/transport identity drift')
    require(live.get('state') == 'running', 'container not running')
    before, after = record(prod['env']), record(live['env'])
    allowed: dict[str, Json] = {'VLLM_DSV41_REPLAY_SKIP':str(replay), 'VLLM_DSV41_REPLAY_AUDIT':str(audit),
        'VLLM_EXL3_MOE_SPLIT_MAX_TOKENS':'0', 'VLLM_DSV41_VOR_FUSED':'0', 'VLLM_ENGRAM_REMOTE_REQUIRED':'1'}
    for key in before.keys() | after.keys():
        if key in allowed:
            require(after.get(key) == allowed[key], f'wrong experimental flag {key}')
        else:
            require(before.get(key) == after.get(key), f'env drift {key}')
    require(all(after.get(k) == v for k,v in allowed.items()), 'missing experimental flags')
    required = {'VLLM_DSV41_KV_PAIR':'3', 'VLLM_DSV41_DROP_MTP_BUFFER':'1',
        'VLLM_EXL3_TF_PROMPT_TRANSIENT':'1', 'VLLM_EXL3_TF_PROMPT_ALIAS':'1',
        'VLLM_INDEXER_PREFILL_BUFFER_FACTOR':'4', 'VLLM_DSPARK_REMOTE_POLICY':'capacity',
        'VLLM_DSPARK_REMOTE_ASYNC':'1', 'VLLM_DSPARK_REMOTE_ASYNC_PROXY':'process'}
    require(all(after.get(k) == v for k,v in required.items()), 'K1/proxy/local-drafter policy mismatch')
    # Source capture hashes target metadata separately; mounted engine bytes use verify_runtime.
    import shlex
    a = shlex.split(str(prod['argv']).replace('\\\n',' '))
    b = shlex.split(str(live['argv']).replace('\\\n',' '))
    i = a.index('--port')
    require(a[i+1] == '8000', 'rollback argv is not production')
    a[i+1] = '8001'
    require(a == b, 'launch argv differs beyond experiment port')
    for key,value in (('--max-model-len','327680'),('--max-num-seqs','4'),
                      ('--max-num-batched-tokens','4096'),('--kv-cache-memory-bytes','6442450944'),
                      ('--long-prefill-token-threshold','1024')):
        require(b[b.index(key)+1] == value, f'wrong runtime {key}')
