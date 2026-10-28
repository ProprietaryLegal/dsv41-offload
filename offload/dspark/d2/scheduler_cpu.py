"""Execute the image's scheduler on CPU without importing torch or vLLM plugins.

Only imports and dependency construction are replaced. Scheduler.schedule,
Request, request queues, SchedulerConfig fields/validators and bookkeeping are
compiled from the pinned image source, never rewritten as a toy scheduler.
KV allocation and model completion are deliberately fake; see README limitations.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import sys
import types
from typing import cast

from pydantic.dataclasses import dataclass as config_dataclass

IMAGE = Path('/opt/dsv41/data/src-image')
PINS = Path(__file__).with_name('scheduler-image-pins.json')


def verify_sources(root: Path = IMAGE) -> None:
    for name, expected in json.loads(PINS.read_text()).items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'image source drift: {name}: {actual} != {expected}')


def load_source(name: str, source: str, injected: dict[str, object]) -> types.ModuleType:
    """Remove unavailable imports, retaining original executable class bodies."""
    tree = ast.parse(source)
    body: list[ast.stmt] = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(not alias.name.startswith(('vllm', 'torch', 'numpy')) for alias in node.names):
                body.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith(('vllm', 'torch', 'numpy', '__future__')):
                body.append(node)
        elif not isinstance(node, ast.If):  # TYPE_CHECKING / optional dependency aliases
            body.append(node)
    tree.body = body
    ast.fix_missing_locations(tree)
    module = types.ModuleType(name)
    module.__dict__.update(injected)
    sys.modules[name] = module  # required by dataclasses and annotation resolution
    exec(compile(tree, name, 'exec'), module.__dict__)
    return module


class Blocks:
    def get_block_ids(self) -> tuple[list[int], ...]:
        return ([],)


class FakeKV:
    """Unlimited fake KV; capacity pressure must be tested separately on GB10."""
    empty_kv_cache_blocks = Blocks()

    def new_step_starts(self) -> None:
        pass

    def allocate_slots(self, *args: object, **kwargs: object) -> Blocks:
        return Blocks()

    def get_blocks(self, request_id: str) -> Blocks:
        return Blocks()

    def record_prefix_cache_stats(self, *args: object) -> None:
        pass

    def get_num_common_prefix_blocks(self, request_id: str) -> list[int]:
        return [0]

    def take_boundary_state_offloads(self) -> dict[str, object]:
        return {}

    def take_kv_cache_block_copies(self) -> tuple[list[object], list[object]]:
        return [], []


def load_engine(scheduler_source: str | None = None) -> types.SimpleNamespace:
    verify_sources()
    logger = logging.getLogger('d2.priority.cpu')
    log_proxy = types.SimpleNamespace(info=logger.info, info_once=logger.info,
        warning=logger.warning, warning_once=logger.warning, debug=logger.debug)
    common: dict[str, object] = {'init_logger': lambda name: log_proxy}
    request = load_source('d2_priority_request', (IMAGE / 'vllm/v1/request.py').read_text(), {
        **common, 'FinishReason': types.SimpleNamespace(STOP=0, LENGTH=1, ABORT=2, ERROR=3, REPETITION=4),
        'StructuredOutputRequest': types.SimpleNamespace(from_sampling_params=lambda params: None),
        'length_from_prompt_token_ids_or_embeds': lambda ids, embeds: len(ids),
        'ConstantList': lambda values: values, 'PrefillStats': lambda: None,
    })
    queue = load_source('d2_priority_queue', (IMAGE / 'vllm/v1/core/sched/request_queue.py').read_text(),
                        {'Request': request.Request})
    config = load_source('d2_priority_config', (IMAGE / 'vllm/config/scheduler.py').read_text(), {
        **common, 'config': config_dataclass,
    })
    output = load_source('d2_priority_output', (IMAGE / 'vllm/v1/core/sched/output.py').read_text(), {
        'strip_covered_mm_data': lambda features, *args, **kwargs: features,
    })
    scheduler = load_source('d2_priority_scheduler', scheduler_source or
                            (IMAGE / 'vllm/v1/core/sched/scheduler.py').read_text(), {
        **common, **{key: value for key, value in vars(queue).items() if not key.startswith('__')},
        **{key: value for key, value in vars(output).items() if not key.startswith('__')},
        'Request': request.Request, 'RequestStatus': request.RequestStatus,
        'SchedulerInterface': object, 'MULTIMODAL_REGISTRY': None,
        'PauseState': types.SimpleNamespace(UNPAUSED=0, PAUSED_ALL=1),
        'record_function_or_nullcontext': lambda *args: nullcontext(),
        'KV_FETCH_WAITING_TO_START': 0,
    })
    async_scheduler = load_source('d2_priority_async', (IMAGE / 'vllm/v1/core/sched/async_scheduler.py').read_text(), {
        **common, 'Scheduler': scheduler.Scheduler, 'RequestStatus': request.RequestStatus,
    })
    return types.SimpleNamespace(scheduler=scheduler, async_scheduler=async_scheduler,
                                  request=request, queue=queue, config=config)


def make_scheduler(engine: types.SimpleNamespace, asynchronous: bool = False) -> types.SimpleNamespace:
    """Bypass GPU-dependent __init__; retain schedule and _update_after_schedule."""
    cls = engine.async_scheduler.AsyncScheduler if asynchronous else engine.scheduler.Scheduler
    sched = cls.__new__(cls)
    sched.scheduler_config = engine.config.SchedulerConfig(
        max_model_len=327680, is_encoder_decoder=False, max_num_seqs=4,
        max_num_batched_tokens=4096, long_prefill_token_threshold=1024, policy='priority')
    fields: dict[str, object] = {
        'current_step': 0, 'vllm_config': types.SimpleNamespace(speculative_config=None),
        'max_num_scheduled_tokens': 4096, '_pause_state': 0, 'max_num_encoder_input_tokens': 0,
        'kv_cache_manager': FakeKV(), 'prefill_capacity_bound': False,
        'adaptive_long_prefill_threshold': False, 'running': [], 'requests': {},
        'waiting': engine.queue.create_request_queue(engine.queue.SchedulingPolicy.PRIORITY),
        'kv_holding_waiting': engine.queue.create_request_queue(engine.queue.SchedulingPolicy.PRIORITY),
        'deferred_waiting': set(), 'num_waiting_for_streaming_input': 0,
        'max_num_active_reqs': 4, 'max_num_running_reqs': 4, 'max_model_len': 327680,
        'num_sampled_tokens_per_step': 1, 'need_mamba_block_aligned_split': False,
        'num_prefill_lookahead': 0, 'num_lookahead_tokens': 0, 'num_spec_tokens': 0,
        'ec_connector': None, 'connector': None, 'aux_output_connector': None, 'lora_config': None,
        'prefix_replay_tokens': 0, 'log_stats': False, 'is_mm_encoder_only': False,
        'is_encoder_decoder': False, 'scheduler_reserve_full_isl': True,
        'use_v2_model_runner': False, 'model_uses_mrope': False, 'prev_step_scheduled_req_ids': set(),
        'kv_cache_config': types.SimpleNamespace(kv_cache_groups=[None]),
        'dynamic_sd_lookup': None, 'reset_preempted_req_ids': set(), 'finished_req_ids': set(),
        'defer_block_free': False, '_inflight_prefills': set(),
        'policy': engine.queue.SchedulingPolicy.PRIORITY,
        'encoder_cache_manager': types.SimpleNamespace(get_freed_mm_hashes=lambda: [], get_manager_metadata=lambda: None),
    }
    for key, value in fields.items():
        setattr(sched, key, value)
    sched._handle_blocked_waiting_request = lambda req: True
    sched._holds_kv_blocks = lambda req: False
    sched._get_local_prefix_cache_hit = lambda req: (Blocks(), 0, 0, False)
    sched._ec_transfer_pending = lambda *args: False
    sched._mark_prefix_replay = lambda *args: 0
    sched._set_kv_fetch_stage = lambda *args: None
    sched._make_cached_request_data = lambda *args: None
    sched._get_new_block_ids_to_zero = lambda: None
    return cast(types.SimpleNamespace, sched)


def make_request(engine: types.SimpleNamespace, request_id: str, tokens: int,
                 priority: int = 10, arrival: float = 0, output: int = 1500) -> types.SimpleNamespace:
    return cast(types.SimpleNamespace, engine.request.Request(request_id, [1] * tokens,
        types.SimpleNamespace(max_tokens=output, extra_args=None, skip_reading_prefix_cache=False),
        None, priority=priority, arrival_time=arrival))
