"""Coordinator pair ownership checks without CUDA imports."""
from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path

PAIR = 'GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc'


def require_pair_ownership() -> None:
    marker = Path.home()/'dsv41-offload/PAIR_OK'
    authorized = os.environ.get('N3_PAIR_BY_COORD') == '1'
    if marker.is_file():
        authorized |= bool(re.match(r'^N3(?:\s|-|$)',marker.read_text()))
    if not authorized or os.environ.get('CUDA_VISIBLE_DEVICES') != PAIR:
        raise RuntimeError('N3 coordinator authorization and exact UUID pair required')
    for fd,name in ((8,'gpu-c4e7a371.lock'),(9,'gpu-7744b44b.lock')):
        path = Path.home()/'dsv41-offload/locks'/name
        held, actual = os.fstat(fd),path.stat()
        if (held.st_dev,held.st_ino) != (actual.st_dev,actual.st_ino):
            raise RuntimeError(f'fd {fd} is not the {name} lock')
        # First prove the lock was held; do not accidentally acquire an unheld
        # inherited descriptor and then report it as preexisting ownership.
        with path.open('a') as probe:
            try:
                fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(probe,fcntl.LOCK_UN)
                raise RuntimeError(f'{name} was not held')
        # A foreign owner also blocks the probe; only our inherited descriptor
        # can reacquire the same flock without blocking.
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
