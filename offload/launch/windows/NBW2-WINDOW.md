# NBW2 coordinator sheet — native kx2 re-measure under WIN2 (prepared, not run)

NDM 2026-10-27: "Do everything fable recommended. Do better deeper testing and measuring. Fairer win rule is
essential." / "go window after window". Runner `offload/launch/windows/nbw2_window.py` (dry by default); greedy through
the shared `nbw_cells --near-tie-nats 0.901 --record-distribution`; KL driver `nbw2_cells.py`. The native drafter is
"Later" in the expert ranking: at most two NBW2 windows. Everything not listed here is the NBW runner and sheet (`NBW-WINDOW.md`): staging of the native
tree, solo receipt, preflight, HOLD admission, fences, pair, guards, W8 restore and recovery are unchanged.

## What one window does

| Start | Cells (each a fresh cache salt, T=1, 1,500-token answer) | Extra |
|---|---|---|
| s1-K | short-b1, 32k-b1, 32k-b3 ×1 | public cuts built once |
| s2-I | short-b1 ×2, 32k-b1 ×2, 32k-b3 ×1 | six greedy checks at 0.9 nats, divergence top-20 recorded |
| s3-K | as s1-K | KL re-probe of s2-I's divergences (reported only) |
| s4-I | as s2-I | as s2-I |
| s5-K | as s1-K, **skipped** if its block no longer fits before HOLD+72 | KL re-probe of s4-I |

Decision cells short-b1 and 32k-b1: 3 K values (one per K start) against 4 I values. 32k-b3 is reported only.

## Timing (MEASURED NBW-W-20261026T231445Z, ×1.15 margin)

| Item | Minutes |
|---|---:|
| K block: start ≈ 375 s + 40 + 51 + 108 s cells | 11.0 |
| I block: start ≈ 670 s incl. native worker and greedy + 2×40 + 2×51 + 108 s | 18.4 |
| Mandatory K I K I | 58.8 |
| With closing K | 69.8 (bound 72) |
| W8 start, check, close (every window today, see Chain mode) | ≈ 8.5 (S94: 8.4–8.7 measured), then 15-min watch |

ESTIMATED HOLD per window (drain excluded): ≈ 70 min at W3 rates, ≈ 78 with the margin (≤ HOLD+90), then the 15-minute
watch. Two windows ≈ 2 × 85 min including watches, with production back between them.

## Rule (`decision_v2` win2-v1, revision 2)

- Primary: pooled server ms per spec request-round (lower is better). Guards: accepted/round, client tok/s; a guard
  causes LOSS only when its loss-repeat test fires AND its 95 % interval is below zero. Reported, never deciding:
  committed tok/s = 1000·(1+accepted/round)/ms-per-round, TTFT, 32k-b3, and the per-start-mean view of the I arm
  (S94: its 4 repeats come from 2 starts; a one-window CANDIDATE stays provisional until a second window agrees).
- WIN: point gain > 2 % and the 95 % interval excludes 0. LOSS: worse than the control spread (1.69 × SD / c4(n) /
  mean) in ≥ ceil(2n/3) repeats. Window: any decision-cell LOSS → NOT A CANDIDATE; ≥ 1 WIN and no LOSS → CANDIDATE;
  else INCONCLUSIVE.
- One clock: every start records its proven cap (its own start log). Mixed or unproven caps across the pooled starts
  (e.g. THERMAL-WATCH dialled back to 2150) → INCONCLUSIVE.
- Arms are unpaired (Welch). The two repeats of one I start share that start's offset; NBW-W3 showed a ~7 % K
  start-to-start tok/s spread. One control per K start keeps that offset inside the control spread and the interval.
- Fewer than three values per arm (closing K skipped) → this window alone is INCONCLUSIVE; pooling decides.
- Archive (v3): every K repeat goes to `$C/control-archive.jsonl` with max_tokens 1500, the production commit, that
  start's proven clock cap, the window date, start and repeat; the archive head is recorded before and after. Spreads
  are read at max_tokens 1500 + commit + the window's cap; NBW cells are never pooled with 512-token cells.
- Orientation from NBW-W3 (3 K vs 2 I, one run each, not a verdict): ms/round I vs K was −1.3 % short-b1, −0.8 % 32k-b1,
  −0.2 % 32k-b3 (I slower). The +3.3 % short tok/s came from acceptance (T=1), not from faster rounds.

## Chain mode (WIN2-R2 `chain.py`)

Wired as R2 specifies: chained preflight with `placeholders_present`, admission from `allow_unreachable(mode)`,
production from `chain.production_snapshot()`, the end from `end_action`, `chain.abort` for a stop without an owned
HOLD while the marker is present, `chain.CHAINED_CLOSE` for the close, and `clear()` only after RESTORED + watch PASS.

**At offload/win2 2b48eec a chained NBW2 window cannot leave production down.** Every scoped runner opens its HOLD
placeholders through `hold_admission.open_script` inside its `-hold` scope; scoped_remote's tmux shim puts
`dsv41-hold` on that scope's private socket. `chain.PLACEHOLDER_PROBE` (default socket) never sees them, and
`CHAINED_CLOSE`'s check, which must run inside the `-restore-close` scope, is sent to that scope's own private socket.
NBW2 therefore runs the exact `CHAINED_CLOSE` check in a throwaway scope first; it fails, and the window ends with the
strict W8 restore and watch (summary `chain_end`, `chain: ENDED (strict restore)`), never burning its one close scope.
A window started inside another runner's chain refuses before any HOLD (placeholders not provable) and runs
`chain.abort`. Run NBW2 as single windows and pool the second with `--pool-with <first window dir>`.

Exit codes: 0 restored without stop; 1 stop with production restored (own W8 + watch PASS, or chain abort RESTORED);
2 restore, watch or abort not verified (read the printed reason; HOLD, marker and `chain-abort/` tell which).

## Launch (coordinator only)

Stage the reviewed HEAD on sparkhead exactly as `NBW-WINDOW.md` "Staging" (bundle clone into a NEW checkout, e.g.
`~/dsv41-nbw2-window`). The native tree, N4 prebuilt caches, reviewed tune cache and `NBW-native-solo-reviewed.json` from
NBW are reused and re-verified by preflight. Then, in a coordinator-owned tmux:

```bash
set -euo pipefail
cd /opt/dsv41/repos/.worktrees/dsv41-nbw-win2   # clean, at the staged HEAD
unset TMUX TMUX_PANE
export WINDOW_COORDINATOR=Deepseek/coordinator CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PWD:$PWD/offload/dspark
# second window: append '--pool-with', '<first NBW2-W-... dir>' to the argument list below
OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/NBW2-W-$(date -u +%Y%m%dT%H%M%SZ)
/home/operator/miniconda3/bin/python3 - "$OUT" "$(git rev-parse HEAD)" dsv41-nbw2-window <<'PY'
import json, sys
from pathlib import Path
from offload.launch.windows.nbw2_window import main
r = json.loads(Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/NBW-native-solo-reviewed.json').read_text())
raise SystemExit(main(['--execute', '--out', sys.argv[1], '--expected-experiment-commit', sys.argv[2],
 '--remote-repo', sys.argv[3], '--native-tree', r['native_tree'], '--native-commit', r['native_commit'],
 '--native-identity', r['native_identity'], '--native-markov-identity', r['native_markov_identity']]))
PY
```

Offline pooling of finished windows: `python -m offload.launch.windows.nbw2_window --decide <dir> [<dir> ...]`.
`--out` must be named `NBW2-W-YYYYMMDDTHHMMSSZ` (the archive date comes from it).

## NBW-W2 (NBW-W-20261026T224001Z) stop, diagnosed

The STOP came from the strict identity gate of the runner then deployed (ea61adf): a native greedy output differed from
K1 on prompt 0 and no near-tie probe ran, so no margin was recorded. It was not a length/prefix mismatch. Its prompt-0
output hash equals NBW-W3's prompt 0, which W3 measured as an exact tie (0.0 nats, token 97). The flip is deterministic
and inside both the 0.501 and the 0.9 near-tie limits, so it cannot recur as a STOP under the near-tie gate.
