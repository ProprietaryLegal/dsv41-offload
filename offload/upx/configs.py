"""R-UPX 0085b arms: what each arm mounts and sets on top of its base config, and the live threshold file.

offload/upx/UPX.env is the config every 0085b arm starts from (offload/launch/start_exp.sh); an arm is that file plus the
environment below. offload/upx/x0085b.env is the append step UPX.env ends with; another lane's window.env can end with
the same line to compose 0085b onto its own arm (PC1's chunk16384, MEASURE-0085B.md M2). This module states the
composition in Python so tests can check the sourced shell result against it, and runners can take ARMS instead of
re-deriving mounts.

    python3 -m offload.upx.configs arms               # ARMS as JSON
    python3 -m offload.upx.configs release-file N     # the command a runner sends to EACH node to set threshold N
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from offload.upx import build_0085b as b

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = 'offload/upx/UPX.env'
SNIPPET = 'offload/upx/x0085b.env'
PY = b.CONTAINER
BASES = {'k1': 'offload/launch/configs/D-kv6-async-pack.env', '0075': 'offload/launch/configs/UP2-0075.env'}
DEFAULT_THRESHOLD = 131072          # the donor's serve setting (README:581, :598)
RELEASE_FILE_CONTAINER = '/root/.cache/dsv41-release-tokens'
RELEASE_FILE_NODE = '~/.cache/dsv41-serve/dsv41-release-tokens'   # the launcher mounts ~/.cache/dsv41-serve at /root/.cache

# Arm -> UPX.env environment. Controls are the base configs themselves (UPX_0085B=0).
ARMS: dict[str, dict[str, str]] = {
    'C-k1': {'UPX_BASE': 'k1', 'UPX_0085B': '0'},
    'release-tokens': {'UPX_BASE': 'k1', 'UPX_0085B': 'tokens'},
    'release-file': {'UPX_BASE': 'k1', 'UPX_0085B': 'file'},
    'C-0075': {'UPX_BASE': '0075', 'UPX_0085B': '0', 'UP2_REPLAY': '0'},
    'release-file+0075': {'UPX_BASE': '0075', 'UPX_0085B': 'file', 'UP2_REPLAY': '0'},
}


def pairs(overlays: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for pair in overlays.split():
        source, separator, target = pair.partition('=')
        if not separator or not source or not target:
            raise ValueError(f'malformed overlay pair {pair!r}')
        found.append((source, target))
    return found


def added(arm: Mapping[str, str]) -> list[tuple[str, str]]:
    """Pairs the arm adds, in x0085b.env's order."""
    if arm['UPX_0085B'] == '0':
        return []
    return [('offload/upx/overlay/' + rel, PY + rel) for rel in b.FILES]


def switch(arm: Mapping[str, str], threshold: int = DEFAULT_THRESHOLD) -> list[str]:
    mode = arm['UPX_0085B']
    if mode == '0':
        return []
    if mode == 'tokens':
        return [f'VLLM_DSV41_RELEASE_TOKENS={threshold}']
    if mode == 'file':
        return [f'VLLM_DSV41_RELEASE_FILE={RELEASE_FILE_CONTAINER}']
    raise ValueError(f'invalid UPX_0085B={mode}')


def compose(base: Sequence[tuple[str, str]], arm: Mapping[str, str]) -> list[tuple[str, str]]:
    """The arm's full mount list. A target the base already mounts, or one that shadows or is shadowed by a base mount
    (a directory mount above or below it), is refused: 0085b never replaces another lane's file."""
    out = list(base)
    for source, target in added(arm):
        mount = PurePosixPath(target)
        for _, other in out:
            o = PurePosixPath(other)
            if o == mount:
                raise ValueError(f'base already mounts {target}')
            if o in mount.parents or mount in o.parents:
                raise ValueError(f'{target} shadows or is shadowed by base mount {other}')
        out.append((source, target))
    return out


def release_file_command(threshold: int) -> str:
    """Write the live threshold atomically (temp file + rename), so a worker's idle read never sees a partial value
    (a partial value would read as invalid -> the env threshold, 0 in file mode: no release, never an error)."""
    if threshold < 0:
        raise ValueError('threshold must be >= 0 (0 = no release)')
    directory = '~/.cache/dsv41-serve'
    temp = f'{directory}/.dsv41-release-tokens.new'
    return f"mkdir -p {directory} && printf '%d\\n' {threshold} > {temp} && mv -f {temp} {RELEASE_FILE_NODE}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('arms', 'release-file'))
    parser.add_argument('threshold', nargs='?', type=int)
    args = parser.parse_args()
    if args.action == 'arms':
        print(json.dumps({'config': CONFIG, 'snippet': SNIPPET, 'bases': BASES, 'arms': ARMS,
                          'default_threshold': DEFAULT_THRESHOLD, 'release_file': RELEASE_FILE_CONTAINER}, indent=1))
        return
    if args.threshold is None:
        parser.error('release-file needs a threshold')
    print(release_file_command(args.threshold))


if __name__ == '__main__':
    main()
