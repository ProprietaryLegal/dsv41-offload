"""Build a single installed-image scheduler overlay, preserving deployed 0031c.

CPU only. Build outputs are inert files; nothing installs or restarts a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMAGE = Path('/opt/dsv41/data/src-image')
SCHEDULER = 'vllm/v1/core/sched/scheduler.py'
DONOR = Path('/opt/dsv41/data/upstream/henryous')


def verify_inputs() -> dict[str, str]:
    pins: dict[str, str] = json.loads((HERE / 'inputs.lock.json').read_text())
    roots = {'image': IMAGE, 'donor': DONOR}
    for key, expected in pins.items():
        kind, name = key.split(':', 1)
        actual = hashlib.sha256((roots[kind] / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'SO2 input drift: {key}: {actual} != {expected}')
    return pins


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f'SO2 expected one source anchor, found {source.count(old)}: {old[:100]!r}')
    return source.replace(old, new, 1)


START = '''        # SO2 starts only after native step initialization. OFF has no state.
        so2 = None
        if _SO2_OPTIONS.enabled:
            self.so2_step = getattr(self, "so2_step", 0) + 1
            so2_stats = getattr(self, "so2_stats", None)
            if so2_stats is None:
                so2_stats = self.so2_stats = dict(steps=0, native=0, mixed=0,
                    admission=0, maintenance=0, pauses=0)
            dependency = (bool(self.kv_holding_waiting) or self.prefill_capacity_bound
                or self.connector is not None or self.ec_connector is not None
                or self.aux_output_connector is not None or self.lora_config is not None
                or self.need_mamba_block_aligned_split or self.num_prefill_lookahead > 1
                or self.policy != SchedulingPolicy.PRIORITY
                or self._pause_state != PauseState.UNPAUSED
                or any(r.has_encoder_inputs or r.structured_output_request is not None
                       for r in self.running))
            work = [_SO2Work(r.request_id, r.priority,
                    r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens,
                    r.is_prefill_chunk,
                    self.current_step >= r.next_decode_eligible_step and not (
                        r.num_output_placeholders > 0 and
                        r.num_computed_tokens + 2 - r.num_output_placeholders >=
                        r.num_prompt_tokens + r.max_tokens)) for r in self.running]
            so2 = _so2_decide(_SO2_OPTIONS, self.so2_step, work,
                waiting_foreground=any(r.priority < 20 for r in self.waiting),
                has_waiters=bool(self.waiting) or bool(self.kv_holding_waiting),
                dependency=dependency, input_budget=input_budget,
                token_budget=token_budget, draft_slots=draft_slots,
                max_sequences=self.max_num_active_reqs)
            so2_stats["steps"] += 1
            so2_stats[so2.mode] += 1
            if so2.mode == "mixed":
                self.running.sort(key=lambda r: (r.priority, r.arrival_time))
                so2_stats["maintenance"] += int(so2.due)

'''

SKIP = '''            # SO2 never excludes work on the strength of an unadmitted waiter.
            so2_background = so2 is not None and request.priority >= 20
            so2_service = so2_background and (so2.mode == "admission" or
                                               (so2.mode == "mixed" and so2.due))
            if (so2_background and so2.mode == "mixed" and not so2.due
                    and any(self.requests[rid].priority < 20 for rid in num_scheduled_tokens)):
                self.so2_stats["pauses"] += 1
                req_index += 1
                continue

'''

RUN_CAP = '''            # 0074: solo background keeps the configured threshold, even if
            # the installed global eligible-request threshold is zero.
            if _SO2_OPTIONS.enabled and request.priority >= 20 and request.is_prefill_chunk:
                solo_cap = self.scheduler_config.long_prefill_token_threshold
                if solo_cap > 0:
                    num_new_tokens = min(num_new_tokens, solo_cap)
                if so2_service:
                    num_new_tokens = min(num_new_tokens, so2.quantum)
'''
WAIT_CAP = '''                    # SO2: retain 0074 only for auxiliary priority >=20.
                    if _SO2_OPTIONS.enabled and request.priority >= 20:
                        solo_cap = self.scheduler_config.long_prefill_token_threshold
                        if solo_cap > 0:
                            num_new_tokens = min(num_new_tokens, solo_cap)
'''


def overlay_source(source: str) -> str:
    verify_inputs()
    expected = hashlib.sha256((IMAGE / SCHEDULER).read_bytes()).hexdigest()
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError('SO2 only composes on the pinned installed scheduler/0031c; no stacked scheduler')
    source = replace_once(source, 'import itertools\n',
        'from so2_policy import Options as _SO2Options, Work as _SO2Work, decide as _so2_decide\n'
        '_SO2_OPTIONS = _SO2Options.from_env()\nimport itertools\n')
    source = replace_once(source, '        # First, schedule the RUNNING requests.\n',
                          START + '        # First, schedule the RUNNING requests.\n')
    anchor = '            if defer_prefills and request.is_prefill_chunk and request.priority >= _DSV41_THROTTLE_MIN_PRIO:\n'
    source = replace_once(source, anchor, SKIP +
        '            if (defer_prefills and request.is_prefill_chunk\n'
        '                    and request.priority >= _DSV41_THROTTLE_MIN_PRIO and not so2_service):\n')
    anchor = '            num_new_tokens = min(\n                num_new_tokens, token_budget, input_budget - draft_slots\n            )\n'
    source = replace_once(source, anchor, RUN_CAP + anchor)
    anchor = '                    # chunked prefill has to be enabled explicitly to allow\n'
    source = replace_once(source, anchor, WAIT_CAP + '\n' + anchor)
    compile(source, 'so2_scheduler.py', 'exec')
    return source


def build(output: Path) -> dict[str, object]:
    pins = verify_inputs()
    if output.exists():
        raise ValueError(f'SO2 build destination must not exist: {output}')
    scheduler = overlay_source((IMAGE / SCHEDULER).read_text())
    policy = (HERE / 'policy.py').read_text()
    compile(policy, 'so2_policy.py', 'exec')
    files = {SCHEDULER: scheduler, 'so2_policy.py': policy}
    receipt: dict[str, object] = {'qualification': 'CPU only; default OFF; not deployed',
        'inputs': pins, 'files': {name: hashlib.sha256(body.encode()).hexdigest()
                                  for name, body in files.items()},
        'container_prefix': '/usr/local/lib/python3.12/dist-packages/',
        'priority_boundary': 20, 'period': 24, 'prefill_quantum': 32}
    output.mkdir(parents=True)
    for name, body in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    (output / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == '__main__':
    main()
