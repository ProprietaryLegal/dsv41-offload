"""Shared in-worker qualification primitives. Importing this module does no GPU work.

These functions require the existing worker, not another model or CUDA context.
They deliberately expose no deserialization/RPC/listener or process launcher.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json

Cases = dict[str, Json]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError('UP3 engine qualification: ' + message)


def attr(obj: object, name: str) -> object:
    return cast(object, getattr(obj, name))


def invoke(fn: object, *args: object, **kwargs: object) -> object:
    require(callable(fn), 'expected engine callable')
    return cast(Callable[..., object], fn)(*args, **kwargs)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(obj: object) -> bytes:
    return inspect.getsource(cast(Callable[..., object], obj)).encode()


def passed(samples: int, **details: Json) -> Json:
    require(samples > 0, 'no actual samples')
    return {'pass': True, 'samples': samples, **details}


@contextmanager
def replaced(obj: object, name: str, value: object) -> Iterator[None]:
    """Always put instrumentation/switches back, including on mismatch or signal."""
    original = attr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def finite(value: float, name: str) -> float:
    require(math.isfinite(value), name + ' is not finite')
    return value


def runner(worker: object) -> object:
    result = attr(worker, 'model_runner')
    config = attr(worker, 'vllm_config')
    parallel = attr(config, 'parallel_config')
    require(attr(parallel, 'tensor_parallel_size') == 2, 'TP2 required')
    require(attr(attr(config, 'model_config'), 'max_model_len') == 327680, 'production context changed')
    require(attr(attr(config, 'scheduler_config'), 'max_num_seqs') == 4, 'production sequence cap changed')
    return result
