"""CPU-only launch preflight: replay skipping requires UP2's pinned scheduler mount (UP2 experiment, UP3 replay arm, or a
0075 production promoted by offload/promote/up2)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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


UP3_SOURCE = 'offload/up3/overlay/replay/' + SCHEDULER
UP3_REMOTE = PREFIX + 'vllm/v1/worker/gpu/spec_decode/remote_dspark'
UP3_REMOTE_SOURCE = 'offload/up3/overlay/replay/vllm/v1/worker/gpu/spec_decode/remote_dspark'


def verify_up3(overlays: str, extra_args: str, root: Path = ROOT) -> None:
    """UP2-ADOPT isolated replay arm on CURRENT production: the scheduler is the UP3 replay manifest's, byte-identical to
    UP2's reviewed scheduler; production's reviewed raw mounts stay, but none may replace or shadow the scheduler, and
    the remote DSpark package must be exactly the arm's directory overlay (production's directory mount removed)."""
    destination = PurePosixPath(DESTINATION)
    remote = PurePosixPath(UP3_REMOTE)
    schedulers: list[str] = []
    remotes: list[str] = []
    targets: list[PurePosixPath] = []
    for pair in overlays.split():
        source, separator, target = pair.partition('=')
        if not separator:
            raise ValueError(f'UP3 replay guard: malformed overlay {pair!r}')
        mount = PurePosixPath(target)
        if mount in targets:
            raise ValueError(f'UP3 replay guard: duplicate overlay target {target}')
        targets.append(mount)
        if mount == destination:
            schedulers.append(source)
        elif mount in destination.parents:
            raise ValueError(f'UP3 replay guard: shadowing scheduler mount {target}')
        if mount == remote:
            remotes.append(source)
    if schedulers != [UP3_SOURCE] or remotes != [UP3_REMOTE_SOURCE]:
        raise ValueError('UP3 replay guard: REPLAY=1 requires exactly the UP3 replay scheduler and remote_dspark overlays')
    raw = shlex.split(extra_args)
    for i, arg in enumerate(raw):
        if arg in ('--volume', '--mount') or arg.startswith(('--volume=', '--mount=')) or (arg.startswith('-v') and arg != '-v'):
            raise ValueError('UP3 replay guard: only reviewed "-v source:target[:ro]" raw mounts are allowed')
        if arg == '-v':
            spec = raw[i + 1] if i + 1 < len(raw) else ''
            parts = spec.split(':')
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != 'ro'):
                raise ValueError(f'UP3 replay guard: malformed raw mount {spec!r}')
            raw_target = PurePosixPath(parts[1])
            if raw_target == destination or raw_target in destination.parents:
                raise ValueError(f'UP3 replay guard: raw mount replaces/shadows the scheduler: {spec}')
            if raw_target == remote or raw_target in remote.parents or remote in raw_target.parents:
                raise ValueError(f'UP3 replay guard: raw mount overlaps the replay remote_dspark package: {spec}')
            if any(raw_target == mount or raw_target in mount.parents for mount in targets):
                raise ValueError(f'UP3 replay guard: raw mount replaces/shadows a replay overlay: {spec}')
    manifest = json.loads((root/'offload/up3/manifests/replay.json').read_text())
    expected = manifest['files'][SCHEDULER]
    up2 = cast(Manifest, json.loads((root/'offload/upstream_adopt/MANIFEST.json').read_text()))
    if [entry['sha256'] for entry in up2['files'] if entry['path'] == SCHEDULER] != [expected]:
        raise ValueError('UP3 replay guard: replay scheduler is not UP2\'s reviewed scheduler')
    if hashlib.sha256((root/UP3_SOURCE).read_bytes()).hexdigest() != expected:
        raise ValueError('UP3 replay guard: scheduler hash differs from the UP3 replay manifest; rebuild/review')


UP2_STAGE = '$HOME/dsv41-offload-prod/up2/'
UP2_OWNED = ('vllm/v1/worker/gpu/input_batch.py', 'vllm/models/deepseek_v41/nvidia/model_state.py',
             'vllm/models/deepseek_v41/replay_skip.py')


def verify_production(overlays: str, extra_args: str, package_id: str, home: Path | None = None,
                      root: Path = ROOT) -> None:
    """0075 IS production (dsv41.env PROD_UP2, offload/promote/up2): every arm built from production's env inherits the
    flag and production's staged scheduler/remote package as raw mounts. Accept exactly those mounts, with UP2's reviewed
    scheduler bytes on this host, and refuse any experiment overlay on a path 0075 owns (no unreviewed composition)."""
    if re.fullmatch('[0-9a-f]{12}', package_id) is None:
        raise ValueError(f'UP2 production guard: invalid PROD_UP2 {package_id!r}')
    stage = UP2_STAGE + package_id
    destination, remote = PurePosixPath(DESTINATION), PurePosixPath(UP3_REMOTE)
    owned = [destination, remote, *(PurePosixPath(PREFIX + p) for p in UP2_OWNED)]
    for pair in overlays.split():
        _, separator, target = pair.partition('=')
        if not separator:
            raise ValueError(f'UP2 production guard: malformed overlay {pair!r}')
        mount = PurePosixPath(target)
        if any(mount == path or mount in path.parents or path in mount.parents for path in owned):
            raise ValueError(f'UP2 production guard: experiment overlay {target} composes with a 0075 file; unreviewed')
    wanted = {destination: f'{stage}/{SCHEDULER}', remote: f'{stage}/remote_dspark',
              **{PurePosixPath(PREFIX + p): f'{stage}/{p}' for p in UP2_OWNED}}
    seen: dict[PurePosixPath, list[str]] = {path: [] for path in wanted}
    raw = shlex.split(extra_args)
    for i, arg in enumerate(raw):
        if arg in ('--volume', '--mount') or arg.startswith(('--volume=', '--mount=')) or (arg.startswith('-v') and arg != '-v'):
            raise ValueError('UP2 production guard: only "-v source:target[:ro]" raw mounts are allowed')
        if arg != '-v':
            continue
        spec = raw[i + 1] if i + 1 < len(raw) else ''
        parts = spec.split(':')
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != 'ro'):
            raise ValueError(f'UP2 production guard: malformed raw mount {spec!r}')
        raw_target = PurePosixPath(parts[1])
        if raw_target in seen:
            seen[raw_target].append(parts[0])
        elif any(raw_target in path.parents for path in owned) or remote in raw_target.parents:
            raise ValueError(f'UP2 production guard: raw mount shadows or enters a 0075 path: {spec}')
    for path, source in wanted.items():
        if seen[path] != [source]:
            raise ValueError(f'UP2 production guard: {path} must be mounted once from {source}, found {seen[path]}')
    manifest = cast(Manifest, json.loads((root/'offload/upstream_adopt/MANIFEST.json').read_text()))
    expected = [entry['sha256'] for entry in manifest['files'] if entry['path'] == SCHEDULER]
    staged = (home or Path.home()) / f'dsv41-offload-prod/up2/{package_id}' / SCHEDULER
    if expected != [hashlib.sha256(staged.read_bytes()).hexdigest()]:
        raise ValueError('UP2 production guard: staged production scheduler differs from UP2 manifest')


def main() -> None:
    if os.environ.get('UP3_ARM') == 'replay':
        verify_up3(os.environ.get('OVERLAYS', ''), os.environ.get('EXTRA_DOCKER_ARGS', ''))
        return
    # start_exp.sh sources dsv41.env with set -a: a 0075 production (UP2-PROMOTE) exports PROD_UP2.
    if os.environ.get('PROD_UP2'):
        verify_production(os.environ.get('OVERLAYS', ''), os.environ.get('EXTRA_DOCKER_ARGS', ''), os.environ['PROD_UP2'])
        return
    verify(os.environ.get('OVERLAYS', ''), os.environ.get('EXTRA_DOCKER_ARGS', ''))


if __name__ == '__main__':
    main()
