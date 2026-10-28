"""UP2-PROMOTE production code package: two inert K1-guard hooks + the switch tool, applied by exact anchors.

Coordinator use, on Controller, BEFORE the HOLD (code only; dsv41.env is never touched here):
  git -C /opt/dsv41/repos/dsv41-deploy worktree add --detach <tmp> <current production commit>
  python3 -m offload.promote.up2.hooks apply --tree <tmp>
  git -C <tmp> add <printed paths> && git -C <tmp> commit ...   ->  push to sparkhead main (updateInstead), like PROMO1 af5eaf6
Works on 596ba1c, on PROMO1's accepted tree and on a PP4-hooked tree (anchors verified on all three by the tests). Every
hook is inert until dsv41.env selects 0075 (PROD_UP2) or the UP2 switch transaction exists. A missing, repeated or
ambiguous anchor refuses; nothing is written on refusal.

Why only two hooks: 0075 replaces none of K1's 13 files, so `stage_overlay.sh verify-running` passes unchanged, and it
does not change the speculative schedule. What remains is (1) the K1 fallback, which on ANY failed start restores the
PRE-K1 manager backup (a wrong-version restore with the switch applied), and (2) running verification of the 0075 bytes
and flag on both ranks inside the guarded start, so W8 can require `UP2 RUNNING VERIFIED`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SWITCH = 'offload/promote/up2/switch.py'
PACKAGES = ('offload/promote/up2/packages/qualified.json', 'offload/promote/up2/packages/pp4-composed.json')
TAG = 'UP2-PROMOTE'


@dataclass(frozen=True)
class Hook:
    path: str
    anchor: str
    replacement: str


RESTORE_LINE = '  if ! python3 offload/promote/k1/manage.py restore >> "$log" 2>&1; then\n'
VERIFY_LINE = 'offload/kvpack/stage_overlay.sh verify-running --expect "$K1_EXPECT" "$HEAD" "$WORKER"\n'

HOOKS = (
    Hook('offload/promote/k1/start_checked.sh', RESTORE_LINE,
         f'  # {TAG} owns an exclusive snapshot of the installed state (logs/up2-switch). The pre-K1 manager backup\n'
         '  # must never restore a 0075 production; the UP2 runner (or its watchdog G9) restores under its own HOLD.\n'
         '  if [ -n "${PROD_UP2:-}" ] || [ -e logs/up2-switch/transaction.json ]; then\n'
         '    echo "K1 UP2 REFUSED; candidate stopped; restoration delegated to the UP2 promotion owner; keep HOLD." >&2\n'
         '    exit "$rc"\n'
         '  fi\n' + RESTORE_LINE),
    Hook('offload/promote/k1/verify.sh', VERIFY_LINE, VERIFY_LINE +
         f'# {TAG}: 0075 files, remote package and flag on both ranks equal the staged package (inert when unset).\n'
         'if [ -n "${PROD_UP2:-}" ]; then python3 offload/promote/up2/switch.py verify-running; fi\n'),
)

HOOKS_PATHS = tuple(dict.fromkeys(hook.path for hook in HOOKS))


def plan(tree: Path) -> dict[str, str]:
    """New text for every hooked file; refuses unless each anchor occurs exactly once and no hook is present."""
    result: dict[str, str] = {}
    for hook in HOOKS:
        text = result.get(hook.path) or (tree / hook.path).read_text()
        if TAG in text:
            raise ValueError(f'{hook.path}: {TAG} hook already present')
        count = text.count(hook.anchor)
        if count != 1:
            raise ValueError(f'{hook.path}: anchor found {count} times, expected exactly once')
        result[hook.path] = text.replace(hook.anchor, hook.replacement, 1)
    return result


def apply(tree: Path) -> list[str]:
    texts = plan(tree)
    copied = (SWITCH, *PACKAGES)
    for path in copied:
        if (tree / path).exists():
            raise ValueError(f'{path} already exists in the production tree')
    for path, text in texts.items():
        (tree / path).write_text(text)
    for path in copied:
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, tree / path)
    return [*texts, *copied]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'apply'))
    parser.add_argument('--tree', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'check':
            plan(args.tree)
            print('UP2 PACKAGE ANCHORS OK')
        else:
            print('\n'.join(apply(args.tree)))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'UP2 PACKAGE REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
