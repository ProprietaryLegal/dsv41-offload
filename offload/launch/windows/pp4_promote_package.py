"""PP4-PROMOTE production code package: four inert hooks + the switch tool, applied by exact anchors.

Coordinator use, on Controller, BEFORE the HOLD (code only; dsv41.env is never touched here):
  git -C /opt/dsv41/repos/dsv41-deploy worktree add --detach <tmp> <current production commit>
  git -C <tmp> cherry-pick 420be24     # repo plan Phase 0: the 2300 clock fold rides this production commit, before the hooks
  python3 -m offload.launch.windows.pp4_promote_package apply --tree <tmp>
  git -C <tmp> add <printed paths> && git -C <tmp> commit ...
  git -C <tmp> push sparkhead HEAD:main  # name the production remote explicitly (`origin` is the private GitHub repo)
  then, in the same step on sparkhead: python3 offload/launch/windows/pp4_promote_switch.py fold-clock  (K1 record rebase)
`apply` refuses a tree whose dsv41.env lacks exactly the folded 420be24 clock line.
Every hook is inert while no PP4 setting is present. An override (PROD_SPEC_OVERRIDE, the declared model_runner.py
replacement) is honoured only while `pp4_promote_switch.py active` validates the switch (open transaction, not restored,
switched bytes installed); a PP4 setting without one refuses. The K1 fallback delegates to the PP4 owner only while
dsv41.env selects PP4 or the transaction is open; after an exact rollback (logs/pp4-switch/restored.json) the original
K1 fallback applies again and the ledger stays as evidence (S91 #4). A missing, repeated or ambiguous anchor refuses;
nothing is written on refusal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from offload.launch.windows import pp4_promote_switch as switch

ROOT = Path(__file__).resolve().parents[3]
SWITCH = 'offload/launch/windows/pp4_promote_switch.py'
TAG = 'PP4-PROMOTE'


@dataclass(frozen=True)
class Hook:
    path: str
    anchor: str
    replacement: str


SPEC_LINE = ("export SPEC='{\"method\":\"dspark\",\"num_speculative_tokens\":5,"
             "\"num_speculative_tokens_per_batch_size\":[[1,2,5],[3,4,4]]}'\n")
RESTORE_LINE = '  if ! python3 offload/promote/k1/manage.py restore >> "$log" 2>&1; then\n'
VERIFY_LINE = 'offload/kvpack/stage_overlay.sh verify-running --expect "$K1_EXPECT" "$HEAD" "$WORKER"\n'
RUNNING_LOOP = ('  for row in "${ROWS[@]}"; do read -r path _ ovl <<<"$row"; want+="$ovl $path"$\'\\n\'; '
                'paths+=" $PREFIX$path"; done\n  want=${want%$\'\\n\'}\n')

HOOKS = (
    Hook('serve.sh', SPEC_LINE, SPEC_LINE +
         f'# {TAG}: a switched production fixes the verify window (dsv41.env PROD_SPEC_OVERRIDE), honoured only while the\n'
         '# validated PP4 switch is active; an override without one refuses the start (never silently applied or ignored).\n'
         'if [ -n "${PROD_SPEC_OVERRIDE:-}" ]; then\n'
         '  python3 "$(cd "$(dirname "$0")" && pwd)/offload/launch/windows/pp4_promote_switch.py" active '
         '--spec "$PROD_SPEC_OVERRIDE" >&2 || { echo "PROD_SPEC_OVERRIDE without an active PP4 switch; refused" >&2; exit 1; }\n'
         '  export SPEC=$PROD_SPEC_OVERRIDE\n'
         'fi\n'),
    Hook('offload/promote/k1/start_checked.sh', RESTORE_LINE,
         f'  # {TAG} owns an exclusive snapshot of the installed state. While dsv41.env selects PP4 or its switch\n'
         '  # transaction is open, the pre-K1 manager backup must never restore over it: the PP4 runner (or its watchdog G9)\n'
         '  # restores under its own HOLD. After an exact rollback (restored.json) the original K1 fallback applies again.\n'
         "  if grep -q '^PROD_PP4=' dsv41.env || { [ -e logs/pp4-switch/transaction.json ] && "
         '[ ! -e logs/pp4-switch/restored.json ]; }; then\n'
         '    echo "K1 PP4 REFUSED; candidate stopped; restoration delegated to the PP4 promotion owner; keep HOLD." >&2\n'
         '    exit "$rc"\n'
         '  fi\n' + RESTORE_LINE),
    Hook('offload/promote/k1/verify.sh', VERIFY_LINE,
         f'# {TAG}: the reviewed PP4 runner replaces exactly the K1 model_runner.py path, only while the validated PP4 switch\n'
         '# is active (inert when no PP4 setting is present; a PP4 setting without an active switch refuses).\n'
         'pp4_runner=\n'
         'if [ -n "${PROD_PP4:-}${PROD_PP4_RUNNER_SHA256:-}${PROD_SPEC_OVERRIDE:-}" ]; then\n'
         '  python3 offload/launch/windows/pp4_promote_switch.py active --runner-sha256 "${PROD_PP4_RUNNER_SHA256:-}" '
         '|| { echo "K1 verification refuses PP4 settings without an active PP4 switch" >&2; exit 1; }\n'
         '  pp4_runner=$PROD_PP4_RUNNER_SHA256\n'
         'fi\n'
         'KVPACK_PP4_RUNNER_SHA256=$pp4_runner ' + VERIFY_LINE +
         'if [ -n "$pp4_runner" ]; then python3 offload/launch/windows/pp4_promote_switch.py verify-running; fi\n'),
    Hook('offload/kvpack/stage_overlay.sh', RUNNING_LOOP, RUNNING_LOOP +
         f'  if [ -n "${{KVPACK_PP4_RUNNER_SHA256:-}}" ]; then  # {TAG}: one declared replacement, exact sha, active switch only\n'
         '    [[ "$KVPACK_PP4_RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "$n: bad KVPACK_PP4_RUNNER_SHA256" >&2; return 1; }\n'
         '    python3 "$here/../launch/windows/pp4_promote_switch.py" active --runner-sha256 "$KVPACK_PP4_RUNNER_SHA256" >&2 '
         '|| { echo "$n: KVPACK_PP4_RUNNER_SHA256 without an active PP4 switch" >&2; return 1; }\n'
         '    [ "$(grep -c \' vllm/v1/worker/gpu/model_runner.py$\' <<<"$want")" = 1 ] '
         '|| { echo "$n: manifest lacks one model_runner.py" >&2; return 1; }\n'
         '    want=$(sed -E "s#^[0-9a-f]{64} (vllm/v1/worker/gpu/model_runner.py)\\$#$KVPACK_PP4_RUNNER_SHA256 \\1#" <<<"$want")\n'
         '  fi\n'),
)

HOOKS_PATHS = tuple(dict.fromkeys(hook.path for hook in HOOKS))


def plan(tree: Path) -> dict[str, str]:
    """New text for every hooked file; refuses unless each anchor occurs exactly once and no hook is present."""
    env = (tree / 'dsv41.env').read_text()
    if env.count(switch.CLOCK_FOLDED) != 1 or switch.CLOCK_PRE in env:
        raise ValueError('dsv41.env lacks the 420be24 clock fold; cherry-pick it before the PP4 hooks')
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
    target = tree / SWITCH
    if target.exists():
        raise ValueError(f'{SWITCH} already exists in the production tree')
    for path, text in texts.items():
        (tree / path).write_text(text)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / SWITCH, target)
    return [*texts, SWITCH]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'apply'))
    parser.add_argument('--tree', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'check':
            plan(args.tree)
            print('PP4 PACKAGE ANCHORS OK')
        else:
            print('\n'.join(apply(args.tree)))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'PP4 PACKAGE REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
