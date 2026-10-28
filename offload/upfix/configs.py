"""R-UPF arms: what each arm mounts on top of its base config, and the CPU-only start gate for the fixed top-k op.

offload/upfix/UPF.env is the one config every arm starts from (offload/launch/start_exp.sh); an arm is that file plus the
environment below. Nothing is on unless an arm sets it. This module states the same composition in Python, so tests can
check the sourced shell result against it, and runners (the WIN2 config-only runner of R-UP3 is the intended consumer) can
take ARMS instead of re-deriving mounts.

  python3 -m offload.upfix.configs arms           # print ARMS as JSON
  python3 -m offload.upfix.configs verify-topk    # start gate for UPF_TOPK=1 (UPF.env runs it; CPU only, no host contact)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = 'offload/upfix/UPF.env'
PY = '/usr/local/lib/python3.12/dist-packages/'
BASES = {'k1': 'offload/launch/configs/D-kv6-async-pack.env', '0075': 'offload/launch/configs/UP2-0075.env'}
SCHEDULER = 'vllm/v1/core/sched/scheduler.py'
ADOPT_SCHEDULER = 'offload/upstream_adopt/overlay/' + SCHEDULER
FILES_0091 = ('vllm/distributed/device_communicators/shm_broadcast.py', 'vllm/v1/kv_offload/tiering/async_lookup.py',
              'vllm/v1/worker/gpu/block_table.py')
TOPK_REL = 'vllm/model_executor/layers/indexer_topk.py'
TOPK_BUILD = 'offload/upfix/topk/build'
TOPK_SO = TOPK_BUILD + '/jit/dsv41_topk.so'
TOPK_SO_CONTAINER = '/opt/dsv41-upfix/dsv41_topk.so'
TOPK_ENV = {'VLLM_DSV41_TOPK_FIX': '1', 'VLLM_DSV41_TOPK_FIX_SO': TOPK_SO_CONTAINER}
# Greedy identity threshold for both items (NDM 2026-10-27 "You can loosen the nats to .9"; nbw_cells.greedy near_tie_nats).
GREEDY_NEAR_TIE_NATS = 0.9 + 1e-3

# Arm -> UPF.env environment. Controls are the base configs themselves (UPF_0091=0, UPF_TOPK=0).
ARMS: dict[str, dict[str, str]] = {
    'C-k1': {'UPF_BASE': 'k1', 'UPF_0091': '0', 'UPF_TOPK': '0'},
    'upf-0091': {'UPF_BASE': 'k1', 'UPF_0091': '1', 'UPF_TOPK': '0'},
    'topk-fix': {'UPF_BASE': 'k1', 'UPF_0091': '0', 'UPF_TOPK': '1'},
    'upf-0091+topk-fix': {'UPF_BASE': 'k1', 'UPF_0091': '1', 'UPF_TOPK': '1'},
    'C-0075': {'UPF_BASE': '0075', 'UPF_0091': '0', 'UPF_TOPK': '0', 'UP2_REPLAY': '0'},
    'upf-0091+0075': {'UPF_BASE': '0075', 'UPF_0091': '1', 'UPF_TOPK': '0', 'UP2_REPLAY': '0'},
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
    """Pairs the arm adds or substitutes, in UPF.env's order."""
    out: list[tuple[str, str]] = []
    if arm['UPF_0091'] == '1':
        out += [('offload/upfix/overlay/' + rel, PY + rel) for rel in FILES_0091]
        variant = 'overlay' if arm['UPF_BASE'] == 'k1' else 'overlay-0075'
        out.append((f'offload/upfix/{variant}/{SCHEDULER}', PY + SCHEDULER))
    if arm['UPF_TOPK'] == '1':
        out += [('offload/upfix/topk/overlay/' + TOPK_REL, PY + TOPK_REL), (TOPK_SO, TOPK_SO_CONTAINER)]
    return out


def compose(base: Sequence[tuple[str, str]], arm: Mapping[str, str]) -> list[tuple[str, str]]:
    """The arm's full mount list: a pair whose target the base already mounts replaces it in place (the 0075 scheduler
    only); any other collision, or a mount that shadows a parent or child path, is refused."""
    out = list(base)
    for source, target in added(arm):
        same = [i for i, (_, t) in enumerate(out) if t == target]
        mount = PurePosixPath(target)
        for _, other in out:
            o = PurePosixPath(other)
            if o != mount and (o in mount.parents or mount in o.parents):
                raise ValueError(f'{target} shadows or is shadowed by base mount {other}')
        if same:
            if not (target == PY + SCHEDULER and arm['UPF_BASE'] == '0075' and out[same[0]][0] == ADOPT_SCHEDULER):
                raise ValueError(f'base already mounts {target}')
            out[same[0]] = (source, target)
        else:
            out.append((source, target))
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f'UPF_TOPK=1 needs {path.relative_to(ROOT)} (WINDOW.md steps T1/T2)')
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'{path.relative_to(ROOT)}: not a JSON object')
    return cast(dict[str, object], data)


def verify_topk(root: Path = ROOT) -> str:
    """The built library exists, its build receipt and a PASSING SM121 bit-compare name the same bytes, and the overlay
    is the reviewed one. Returns the library sha256."""
    library = root / TOPK_SO
    if not library.is_file():
        raise ValueError(f'UPF_TOPK=1 needs the built library {TOPK_SO} (WINDOW.md step T1)')
    digest = sha256(library)
    build = _receipt(root / TOPK_BUILD / 'receipt.json')
    if build.get('library_sha256') != digest:
        raise ValueError('dsv41_topk library differs from its jit_build receipt')
    if build.get('arch') != '12.1':
        raise ValueError(f'dsv41_topk library was built for {build.get("arch")!r}, not 12.1')
    compare = _receipt(root / TOPK_BUILD / 'bitcompare.json')
    if compare.get('library_sha256') != digest:
        raise ValueError('the bit-compare receipt is for different library bytes')
    if compare.get('complete') is not True or compare.get('cases_done') != compare.get('cases'):
        raise ValueError(f'the bit-compare receipt is incomplete ({compare.get("cases_done")!r} of '
                         f'{compare.get("cases")!r} cases; T2 crashed or timed out); the arm may not start')
    if compare.get('verdict') != 'PASS':
        raise ValueError(f'dsv41_topk bit-compare verdict {compare.get("verdict")!r}; the arm may not start')
    if compare.get('capability') != [12, 1]:
        raise ValueError(f'bit-compare ran on capability {compare.get("capability")!r}, not SM121')
    manifest = json.loads((root / 'offload/upfix/topk/MANIFEST-topk.json').read_text())
    if sha256(root / 'offload/upfix/topk/overlay' / TOPK_REL) != manifest['overlay']['overlay_sha256']:
        raise ValueError('indexer_topk.py overlay differs from MANIFEST-topk.json')
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('arms', 'verify-topk'))
    args = parser.parse_args()
    if args.action == 'arms':
        print(json.dumps({'config': CONFIG, 'bases': BASES, 'arms': ARMS,
                          'greedy_near_tie_nats': GREEDY_NEAR_TIE_NATS}, indent=1))
        return
    try:
        digest = verify_topk()
    except ValueError as error:
        raise SystemExit(f'UPF_TOPK start gate: {error}') from error
    print(f'UPF_TOPK start gate: OK (library {digest[:12]})', file=sys.stderr)  # stdout stays clean for sourcing


if __name__ == '__main__':
    main()
