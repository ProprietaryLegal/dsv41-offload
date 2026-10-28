"""ESTIMATED latency model over real image schedule statements, never inference.

No network/GPU. Synthetic delivery keeps one decode anchor pending; expected
accepted+bonus output tokens per round is a separate explicit cost assumption.
All 60K background prompt work is drained before reporting the total makespan.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from offload.bg_prefill.cpu import engine
from offload.dspark.d2.scheduler_cpu import make_request


@dataclass(frozen=True)
class Cost:
    decode_ms: float = 39
    prefill_tokens_s: float = 2200
    delivered_per_round: float = 2
    prefill_batch_ms: float = 0

    def __post_init__(self) -> None:
        if (self.decode_ms <= 0 or self.prefill_tokens_s <= 0 or self.delivered_per_round <= 0
                or self.prefill_batch_ms < 0):
            raise ValueError('SO2 simulation costs must be positive (batch overhead may be zero)')


def simulate(backgrounds: int, foreground_priority: int, enabled: bool,
             cost: Cost = Cost(), rounds: int = 300) -> dict[str, object]:
    if backgrounds not in (1, 2) or rounds < 1:
        raise ValueError('SO2 simulation needs one/two backgrounds and positive round count')
    e, s = engine(enabled=enabled)
    s.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=4)
    for j in range(backgrounds):
        r = make_request(e, f'bg{j}', 60000, priority=20)
        r.status, r.is_prefill_chunk = e.request.RequestStatus.RUNNING, True
        s.requests[r.request_id] = r
        s.running.append(r)
    fg = make_request(e, 'fg', 256, priority=foreground_priority, output=100000)
    fg.num_computed_tokens = 256
    fg.append_output_token_ids(7)
    fg.status, fg.is_prefill_chunk = e.request.RequestStatus.RUNNING, False
    s.requests['fg'] = fg
    s.running.append(fg)
    elapsed = foreground_time = overlap_time = 0.0
    foreground_rounds = overlap_rounds = prompt_total = bg_at_foreground_end = 0
    max_bg_gap = 0
    last_bg_step = {f'bg{j}': 0 for j in range(backgrounds)}
    step = 0
    while s.running:
        step += 1
        if step > 100000:
            raise RuntimeError('SO2 simulation stalled')
        bg_present = any(r.priority >= 20 for r in s.running)
        counts: dict[str, int] = dict(s.schedule(throttle_prefills=step % 24 != 0).num_scheduled_tokens)
        assert counts and sum(counts.values()) + len(counts) * 4 <= 4096
        prompts = sum(n for rid, n in counts.items() if rid.startswith('bg'))
        prompt_batches = sum(rid.startswith('bg') for rid in counts)
        prompt_total += prompts
        dt = ((cost.decode_ms / 1000 if 'fg' in counts else 0)
              + prompts / cost.prefill_tokens_s + prompt_batches * cost.prefill_batch_ms / 1000)
        elapsed += dt
        if 'fg' in counts:
            foreground_rounds += 1
            if bg_present:
                overlap_time += dt
                overlap_rounds += 1
        for rid in counts:
            r = s.requests[rid]
            r.num_output_placeholders = r.num_in_flight_tokens = 0
            if rid.startswith('bg'):
                if 'fg' in counts:
                    max_bg_gap = max(max_bg_gap, step - last_bg_step[rid])
                last_bg_step[rid] = step
                if r.num_computed_tokens >= 60000:
                    s.running.remove(r)
                    del s.requests[rid]
            elif foreground_rounds == rounds:
                foreground_time = elapsed
                bg_at_foreground_end = prompt_total
                s.running.remove(r)
                del s.requests[rid]
            else:
                r.append_output_token_ids(7)
    assert foreground_rounds == rounds
    assert prompt_total == backgrounds * 60000
    return {'classification': 'ESTIMATED', 'enabled': enabled,
        'foreground_priority': foreground_priority, 'backgrounds': backgrounds,
        'cost': asdict(cost), 'foreground_rounds': rounds,
        'estimated_foreground_tokens': rounds * cost.delivered_per_round,
        'foreground_s': foreground_time,
        'foreground_tokens_s': rounds * cost.delivered_per_round / foreground_time,
        'overlap_foreground_tokens_s': overlap_rounds * cost.delivered_per_round / overlap_time,
        'background_tokens_at_foreground_end': bg_at_foreground_end,
        'background_prompt_tokens_total': prompt_total,
        'max_background_service_gap_calls_during_overlap': max_bg_gap,
        'all_work_s': elapsed, 'scheduler_calls': step,
        'so2_stats': getattr(s, 'so2_stats', None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'SO2 simulation output exists: {args.output}')
    rows = [simulate(backgrounds, priority, enabled)
            for priority in (0, 10) for backgrounds in (1, 2) for enabled in (False, True)]
    sensitivity = [simulate(backgrounds, 0, enabled, Cost(decode_ms=ms, prefill_batch_ms=overhead))
                   for backgrounds in (1, 2) for enabled in (False, True)
                   for ms in (37, 41) for overhead in (0, 20, 55)]
    # Simple normalized donor-rate translation, explicitly not another forecast.
    local_solo = 2 / .039
    donor = [{'classification': 'UPSTREAM CLAIM; local translation ESTIMATED',
              'backgrounds': n, 'donor_off': off, 'donor_on': on,
              'speedup': on / off,
              'local_normalized_off': local_solo * off / 55.9,
              'local_normalized_on': local_solo * on / 55.9}
             for n, off, on in ((1, 31.3, 57.4), (2, 22.5, 56.5))]
    report = {'classification': 'ESTIMATED; not GB10 measurements',
        'cost_source': 'COMPARISON.md W8: ~37–41 ms single decode; ~2200 prompt tok/s from 150K/~69s',
        'assumptions': '2 delivered tokens/round, linear additive prompt cost, no fixed overhead by default',
        'native_cadence': 24, 'input_budget': 4096, 'long_prefill_threshold': 1024,
        'background_prompt_tokens_each': 60000, 'rows': rows,
        'sensitivity': sensitivity, 'upstream_normalization': donor}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'scenarios': len(rows),
                      'sensitivity_scenarios': len(sensitivity)}))


if __name__ == '__main__':
    main()
