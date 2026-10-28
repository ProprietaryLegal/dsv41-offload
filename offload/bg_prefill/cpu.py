"""D2-FIX1 installed-statement loader, with only the inert SO2 policy injected."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from offload.bg_prefill import policy
from offload.bg_prefill.build_overlay import IMAGE, SCHEDULER, overlay_source
from offload.dspark.d2.scheduler_cpu import load_engine, load_source, make_scheduler


def engine(enabled: bool = True, asynchronous: bool = False,
           period: int = 24, quantum: int = 32) -> tuple[SimpleNamespace, SimpleNamespace]:
    sys.modules['so2_policy'] = policy
    e = load_engine(overlay_source((IMAGE / SCHEDULER).read_text()))
    utils = load_source('so2_cpu_utils', (IMAGE / 'vllm/v1/core/sched/utils.py').read_text(), {})
    e.scheduler.remove_all = utils.remove_all
    e.scheduler._SO2_OPTIONS = policy.Options(enabled, period, quantum)
    e.scheduler._DSV41_THROTTLE_MIN_PRIO = 5
    s = make_scheduler(e, asynchronous)
    production_speculation(s)
    return e, s


def production_speculation(s: SimpleNamespace) -> None:
    # Production DSpark: five speculative tokens, four new draft slots, and
    # single-module prefill lookahead (a no-op for chunk sizing).
    s.num_prefill_lookahead = 1
    s.num_spec_tokens = 5
    s.num_lookahead_tokens = 5
    s.dynamic_sd_lookup = [0, 5, 5, 4, 4]
    s.vllm_config.speculative_config = SimpleNamespace(
        method='dspark', num_speculative_tokens=5,
        max_num_new_slots_for_drafting=4)
