"""CPU-only launch preflight: replay skipping requires UP2's pinned scheduler mount."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
from typing import cast

from offload.upstream_adopt.build_overlay import Manifest, PREFIX, ROOT

SCHEDULER = 'vllm/v1/core/sched/scheduler.py'
SOURCE = 'offload/upstream_adopt/overlay/' + SCHEDULER
DESTINATION = PREFIX + SCHEDULER


def verify(overlays: str, extra_args: str = '', root: Path = ROOT) -> None:
    """Reject absent, replaced, shadowed or tampered scheduler before remote work."""
    matching: list[str] = []
    destination = PurePosixPath(DESTINATION)
    for pair in overlays.split():
        source, separator, target = pair.partition('=')
        if not separator:
            raise ValueError(f'UP2 replay scheduler guard: malformed overlay {pair!r}')
        mount = PurePosixPath(target)
        if mount == destination:
            matching.append(source)
        elif mount in destination.parents:
            raise ValueError(f'UP2 replay scheduler guard: shadowing scheduler mount {target}')
    if matching != [SOURCE]:
        raise ValueError('UP2 replay scheduler guard: REPLAY=1 requires exactly the UP2 scheduler mount; '
                         'image/SO2 schedulers do not compose with replay skipping')
    # Extra raw Docker mounts could replace this scheduler or its parent after
    # the reviewed OVERLAYS list. Require every replay mount to use OVERLAYS.
    raw = iter(shlex.split(extra_args))
    for arg in raw:
        # K1's existing transport directory is outside the Python package tree.
        if arg == '-v':
            mount_arg = next(raw, '')
            if mount_arg == '$HOME/dsv41-offload-run/ofl:/opt/dsv41-ofl:ro':
                continue
            raise ValueError('UP2 replay scheduler guard: raw Docker mounts require review; use OVERLAYS')
        if arg in ('-v', '--volume', '--mount') or arg.startswith(('-v', '--volume=', '--mount=')):
            raise ValueError('UP2 replay scheduler guard: raw Docker mounts require review; use OVERLAYS')
    manifest = cast(Manifest, json.loads((root/'offload/upstream_adopt/MANIFEST.json').read_text()))
    expected = [entry['sha256'] for entry in manifest['files'] if entry['path'] == SCHEDULER]
    actual = hashlib.sha256((root/SOURCE).read_bytes()).hexdigest()
    if expected != [actual]:
        raise ValueError('UP2 replay scheduler guard: scheduler hash differs from UP2 manifest; rebuild/review')


def main() -> None:
    verify(os.environ.get('OVERLAYS', ''), os.environ.get('EXTRA_DOCKER_ARGS', ''))


if __name__ == '__main__':
    main()
