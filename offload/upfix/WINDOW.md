# R-UPF windows — identity / no-loss, rule win2-v1, chain mode

Designs for the two R-UPF items against `offload/launch/windows/WIN2.md` (merged here at 48bc7a7). The same designs exist
as data in `windows.py`, and `tests/test_windows.py` checks them against WIN2's limits. Arms are config-only
(`offload/upfix/UPF.env` plus the environment in `configs.ARMS`), so the WIN2 config-only chain runner can run them
(R-UP3's `offload/up3-win2` runner is the intended one). This lane builds no runner and contacts no host.

## Verdict: identity and no loss, not a win

Both items are correctness backports. Upstream and the donor (henryous, MEASURED on its own GB10 TP2 stack, not ours)
report that they cost nothing. A WIN is therefore not required.

| Outcome | Condition |
|---|---|
| **PROMOTABLE (no loss)** | Every identity gate of the arm passes, and `decide_window` reports no decision-cell LOSS (status CANDIDATE or INCONCLUSIVE). |
| **NOT A CANDIDATE** | Any decision-cell LOSS: worse than the control spread in at least 2 of 3 repeats (`decide_cell`; the same LOSS semantics as R-PP4's promotion gate). |
| **STOP** | Any gate fails. Stop that arm; the window goes on to the next start. |

`windows.arm_verdict(arm, decision, gates)` implements this table.

## Arms (`configs.ARMS`, all on the K1 production-equivalent base `D-kv6-async-pack.env`)

| Arm | Environment | Mounts added |
|---|---|---|
| `C-k1` | `UPF_0091=0 UPF_TOPK=0` | none. Equals the base config exactly (tested). |
| `upf-0091` | `UPF_0091=1` | the four 0091 files (`overlay/`) |
| `topk-fix` | `UPF_TOPK=1` | `topk/overlay/.../indexer_topk.py` and the built library at `/opt/dsv41-upfix/dsv41_topk.so`; env `VLLM_DSV41_TOPK_FIX=1 VLLM_DSV41_TOPK_FIX_SO=...` |
| `upf-0091+0075` | `UPF_BASE=0075 UPF_0091=1 UP2_REPLAY=0` | three 0091 files, plus the UP2 scheduler mount replaced in place by `overlay-0075/` |

`UPF_TOPK=1` refuses to source unless `python3 -m offload.upfix.configs verify-topk` passes. That check is CPU only. It
requires that the library exists, that its jit_build receipt and a PASSING SM121 bit-compare receipt name the same
sha256, and that the overlay matches `MANIFEST-topk.json`.

## Recommended: one chained window, `UPF-W1` (both items, one shared control pair)

Run on sparkhead from `~/dsv41-deploy`, with the `offload/upfix` branch checked out, inside a HOLD. **T0 comes
first in either preflight mode.** In `'production'` mode (the first window of a chain, or an unchained window)
production is still serving when the HOLD opens. In `'chained'` mode `./stop.sh` is a no-op, but the idle check still
runs. T1 and T2 never run beside a serving `vllm`.

| # | Step | Arm | Content (in order) | ESTIMATED min |
|---|---|---|---|---|
| T0 | stop + idle | none | `./stop.sh`, then `start_exp.sh`'s GPU-idle check (lines 24–28) on both nodes: no compute process on either GB10, or STOP. | 1 |
| T1 | jit_build | none | Build `dsv41_topk.so` for SM121 inside the image (command below). T1 always rebuilds: the receipt records no image digest. | 10 |
| T2 | bitcompare | none | SM121 bit-compare, image `_C` vs the fixed op, 149 cases; the 12 NaN cases run last on the fixed op only (command below; `timeout 600`). On FAIL, timeout or crash, skip X-topk and keep the receipt. | 3 |
| 1 | C1 | `C-k1` | warm-up · greedy (must be **6/6 IDENTICAL** to the K1 reference, or the window STOPs: the base drifted) · `ctl-150k-b1` ×2 · `ctl-32k-b4` ×2 · soak OFF | 30.8 |
| 2 | X-0091 | `upf-0091` | warm-up · greedy (0.9) · needle 150K · `ctl-150k-b1` ×3 · `ctl-32k-b4` ×3 · soak ON | 38.8 |
| 3 | X-topk | `topk-fix` | warm-up · greedy (0.9) · needle 150K · needle 300K · `ctl-150k-b1` ×3 · `ctl-32k-b4` ×3 | 32.8 |
| 4 | C2 | `C-k1` | warm-up · greedy KL re-probe of the candidates' recorded divergences · `ctl-150k-b1` ×2 · `ctl-32k-b4` ×2 | 20.8 |

**ESTIMATED HOLD: 143 min chained; 165 min as the last window of a chain** (adds the MEASURED 7.4-min strict W8 restore
and the 15-min post-close watch). The figures come from the MEASURED arm start (6.8 min, upper end of 6.5–6.8) and the
shared catalogue's cell bounds. Gates, soak, T0–T2 and 5 min of fences are ESTIMATED (`windows.py`). The cell bounds
are generous: `ctl-150k-b1` is bounded at 180 s against a MEASURED 69 s TTFT plus a 512-token answer.

S93 estimated 145–147 / 167–169, assuming T2 would grow to 4–6 min with the wider and extra cases. It did not. Row
generation is now vectorised, and the whole 149-case harness (generation, references and scoring of 990 kernel calls)
MEASURED about 12 s on Controller's CPU. T2 keeps its 3-min bound (container start, `vllm._custom_ops` import, CUDA init).
Only T0 (1 min) is added.

Fallbacks, if the coordinator prefers shorter HOLDs: `UPF-0091` (C1 / X-0091 / C2) **96 min chained, 118 min last**;
`UPF-TOPK` (T0–T2, C1 / X-topk / C2, with no soak in C1) **94 min chained, 116 min last**. Together that is 190 min
against 143 min for `UPF-W1`.

## Decision cells (exactly two, ×3 per candidate, `max_tokens` 512)

| Cell | Group | Primary metric | Guards (LOSS only) | Why this cell |
|---|---|---|---|---|
| `ctl-150k-b1` | single-user | pooled server ms/round | accepted/round, TTFT | After a 150K prefill, each decode row's top-k runs over about 150,000 columns at the five ratio-1 indexers (layers 20–36) and about 75,000 at the three ratio-2 indexers (layers 2/8/14). Both are above the 32,768 radix threshold, so the cooperative radix path (#63475) is live at all eight. It is also a long-context check for 0091 (block-table gathers and SHM traffic on every step). |
| `ctl-32k-b4` | multi-user | pooled server ms/round | accepted/round | Four 32K users: up to 24 decode rows of about 32,000 columns (ratio-1 indexers) and 16,000 (ratio-2), on the medium top-k path (#62936's `recover_medium`; #62467's tail lanes). The batch shrinks as streams finish, which is the #61174 stale-row case. (#59880, #59504 and #59096 are inert in our launch: their code runs only with a KV connector's async loads, and we have none.) |

- Control repeats are C1 + C2 = 4, and each candidate has 3. The arms are unpaired, so the Welch interval is used
  (`windows.series(..., paired=False)`). `decide_window(series, DECISION_CELLS, groups=GROUPS)` is called once per
  candidate.
- `max_tokens` 512 on every decision cell is the archive's majority definition. Report `archive_spread(...,
  where={'max_tokens': 512})` next to the in-window spread. The 2 % effect is not raised.
- Archive (control-archive v3, WIN2-R2): `windows.archive_rows(out, 'C1' | 'C2', values)` builds the rows for every
  C1 and C2 repeat of both cells (`ms_per_round`, `accepted_per_round`, and for 150K `ttft_s`). Each row carries:
  - `max_tokens: 512`;
  - the 40-hex **production** commit (`win2_history.production_commit`; never the experiment commit);
  - that start's proven `clock_cap_mhz` and `clock_source` (`start_clock_cap`; both Sparks must agree);
  - `date` YYYYMMDD, `start` and `repeat`.

  Record `archive_head` before and after `archive_append`. Spreads use `where=windows.spread_where(out, 'C1')`. Append
  at window end, and also after a stop for the starts that finished.
- Rule revision 2 (WIN2-R2): `arm_verdict` refuses a decision without `rule_revision: 2`.
  - A guard metric now LOSES only when its interval is also below zero. Guard noise alone cannot fail a no-loss gate
    (tested).
  - The primary metric's LOSS still has no interval condition. Its control spread is 1.69 × s/c4(n), about 1.6 % on
    a quiet 4-repeat control.
  - WIN2-R2 simulated that unpaired 4 v 3 rejects about 15 % of no-effect windows (two cells, with guards). That
    false NOT A CANDIDATE rate applies to these correctness backports too. The fix would be to give the primary
    metric the guard's interval condition: about 6 % rejected; one line in `decide_decision_cell`; a coordinator
    decision.

**Where the fixed kernels run in production.** The indexer's decode logits are `max_model_len // compress_ratio`
wide. The served model's indexers use ratio 1 (layers 20–36) and ratio 2 (layers 2/8/14), so at the 327,680 cap the
rows are 327,680 and 163,840 columns wide. A row takes the radix path (#63475) once its context passes 32,768 tokens
on the ratio-1 indexers, or 65,536 tokens on the ratio-2 ones. It takes the medium path (#62936) from 8,193 to 32,768,
and the short path below that. **The 524,288 cap:** at 524,288 columns and 1–4 rows a group needs about 62 CTAs
against 48 resident, so the launcher hands off to the image's `_C::top_k_per_row_decode`. T2 includes those rows, so a
PASS qualifies topk-fix **at the kernel level at both caps**. The in-window identity and speed evidence (greedy,
needles, decision cells) is at the 327,680 cap only.

## Identity gates

| Gate | Arm | Registration |
|---|---|---|
| greedy | both | `nbw_cells.greedy(..., near_tie_nats=0.9 + 1e-3, record_distribution=True, probe_salt_prefix=<window tag>)` on the six K1 reference prompts (NDM 2026-10-27 ".9"). Any divergence must be a registered near-tie. C2 runs `greedy_kl.compare_recorded(..., bound=max(0.005, 2 x the C1/C2 A/A floor))`. The C1 greedy result is the A/A baseline. |
| needle 150K | both | `up2_request.needle(target=150000, ...)`: recall 6/6, no foreign codes, `cached_tokens` 0, finish `stop`. |
| needle 300K | topk | The same at `target=300000` (under the 327,680 cap). This is the regime where every decode step uses the radix path. |
| bitcompare | topk | T2 receipt `verdict` PASS: zero bad rows for the fixed op over 149 cases. **Paths:** the production shapes (k 512 × 163,840 and 327,680 columns × 1–24 rows, aligned, lengths on the short, medium and radix paths), the 524,288 cap at 1–4 rows (the launcher's hand-off to `_C::top_k_per_row_decode`), radix rows ×50 repeats. **Per fix:** packed rows with more than 4,096 candidates in the threshold bin on the short, medium (`recover_medium`) and radix paths (#62936); upstream's masked-row-tail trigger at an aligned stride (L = k+1..k+3 and 20,001; 0 or 100 finite values; -inf or -1e30 fill; finite padding past L) and -inf rows on the medium and radix paths (#62467); exact ties straddling the k-th value on all three paths; NaN rows (#63118), fixed op only and last. Tie-free rows are scored as the exact set. Rows with ties, -inf or NaN are scored by validity: indices in [0, L), min(k, L) distinct, the reference top-k value multiset (NaN above +inf, as `torch.topk`), and every finite position of a masked row selected. Image-kernel bad rows are reported, never a failure: they are the bugs. MEASURED on Controller's CPU: the whole harness (generation, references, scoring of 990 kernel calls) takes about 12 s. The receipt is rewritten atomically after every case. A crash or timeout (even in the last NaN cases) leaves `complete: false` with every finished case kept for the report, and `verify-topk` refuses it. |
| soak | 0091 | `windows.soak_ok`: for every process role, RSS growth ON ≤ max(16 MiB, 2 × the same role OFF). |

**Soak (C1 OFF, X-0091 ON).** Run for 600 s. Keep four concurrent streams of the `short` prompt (`max_tokens` 512, a
fresh salt per request), re-issuing each stream as soon as it finishes. Every 10 s, sample `docker exec <serving
container> ps -eo pid,rss,args` on both nodes, and `df -B1 /dev/shm` on the head. A role's growth is the median of its
last three samples minus the median of its first three. Limit: #63165 leaks only a libzmq command per notification
(about 17,000 notifications in 10 min at about 35 ms/round, ESTIMATED), so 10 minutes cannot show the fix's benefit.
The soak proves that 0091 adds no growth of its own. The benefit is upstream's claim for days-long sessions.

## T0 / T1 / T2 commands (sparkhead, `~/dsv41-deploy`, inside the HOLD; `IMAGE`, `HEAD`, `WORKER` from `dsv41.env`)

```
# T0, both preflight modes: stop, then start_exp.sh:24-28's idle check on both nodes. Any compute process: STOP.
./stop.sh
for n in $HEAD $WORKER; do
  busy=$(ssh $n "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" | grep -v '^$' || true)
  [ -z "$busy" ] || { echo "GPU busy on $n after stop: $busy" >&2; exit 1; }
done
B=offload/upfix/topk/build
# T1, CPU only. MEASURED reference: 104 s to compile the same source for sm_86 on Controller (nvcc 12.4, --threads=4).
rm -rf $B/jit $B/receipt.json $B/bitcompare.json
docker run --rm --pull never --network none --cpus 16 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/opt/repo:ro" -v "$PWD/$B:/opt/repo/$B" -w /opt/repo --entrypoint python3 "$IMAGE" \
  -m offload.upfix.topk.jit_build --execute --build-dir $B/jit --receipt $B/receipt.json
# T2, one GB10 GPU, only after T0, bounded at 600 s so a hung kernel cannot stall the window. Exit 0 = PASS;
# 2 = FAIL; 124/137 = timed out (the named container is removed so it cannot hold the GPU); anything else = crashed.
# Every non-zero exit skips X-topk (verify-topk refuses a missing, FAIL or incomplete receipt).
timeout -k 30 600 docker run --rm --name upfix-t2 --pull never --network none --gpus all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/opt/repo:ro" -v "$PWD/$B:/opt/repo/$B" -w /opt/repo --entrypoint python3 "$IMAGE" \
  -m offload.upfix.topk.bitcompare --execute --library $B/jit/dsv41_topk.so --out $B/bitcompare.json; t2=$?
case $t2 in 124|137) docker rm -f upfix-t2 >/dev/null 2>&1 || true ;; esac
```

C1's `start_exp.sh` repeats the stop and the GPU-idle check, so a container that outlived T2 stops the window before
any measurement.

`start_exp.sh` copies the library to both nodes like any other overlay. Both Sparks are SM121 on the same image, so one
build serves both. The build directory is git-ignored.

Runner preflight notes (S93, not fixes):
- **Toolchain.** T1 needs more than nvcc with sm_121 support. `cpp_extension.load` also needs ninja, g++ and
  `Python.h` in the image. This is likely, because `serve_exp.sh` JIT-builds plugin kernels at start, but it is
  unproven. A failed T1 skips X-topk only.
- **Image bases.** UPF has no in-image base check like K1's `stage_overlay.sh check`. Before the first start, compare
  each overlay's base file inside `$IMAGE` with `base_sha256` in `MANIFEST-0091.json` and `MANIFEST-topk.json`. The
  five UPF overlays replace whole files built from src-image.
- **Archive history** for these cells is filed under `150k-b1` and `32k-b4` at `max_tokens` 512. There are no
  `ctl-` rows yet, so report `archive_spread` for the old names as well.
- **max_tokens.** The catalogue's default is 1,500. Pass 512 explicitly.

## Chain mode (WIN2-R2, offload/win2 2b48eec)

**Launch.** Use `offload/upfix/launch/UPF-RUN.sh WINDOW COMMIT PREV_TMUX PREV_FILE` (a template; copy it to the campaign
`launch/`). In either mode it gates on `offload/launch/windows/chain_prev_ok.sh`.
- Exit 3 means the chain stopped and the gate has already run the chain abort (strict W8 restore + watch). The script
  passes 3 through and does not start.
- Exit 2 means refused with no marker.
- Before any gate it refuses while the runner `offload.upfix.window` is absent. **No runner exists yet** (R-UPF report).

**Runner rules** (the WIN2-R2 re-merge instructions, for whichever runner executes these windows):
1. **Preflight:** `chain.preflight(serving, hold, placeholders_present=chain.placeholders_present())`.
2. **Admission:** `allow_unreachable=chain.allow_unreachable(mode)`. In `'chained'` mode, prove the Sparks quiet
   before T0 and before every `start_exp.sh`. C1 is the baseline, and production comes from
   `chain.production_snapshot()`.
3. **End:** `chain.end_action(stopped, hold_written)`.
   - `'chained'`: close the window's own HOLD with `['bash', '-c', chain.CHAINED_CLOSE]` inside the `-restore-close`
     scope, then `finish_chained`. Never kill the `dsv41-hold` placeholders.
   - `'restore'`: the strict six-IDENTICAL W8 restore and watch, then `chain.clear()`.
   - `'abort'`: `result['restore'] = chain.abort(out / 'chain-abort')`.
4. **Arm-level stops are not window stops.** A STOP of one arm, a FAIL or timeout at T2, or a refused `verify-topk`
   skips that arm, records why, and keeps the controls. A STOP of C1's identity baseline is a window stop.

## Not run now: `upf-0091+0075`

This arm is needed only if UP2-ADOPT promotes the 0075 scheduler. With replay skipping on (`UP2_REPLAY=1`),
`UPF.env` refuses it, because `offload/upstream_adopt/replay_guard.py` admits only the UP2 scheduler source. The
shared-module change is in the R-UPF report: admit `offload/upfix/overlay-0075/.../scheduler.py` pinned by its
`MANIFEST-0091.json` sha256. The composed file carries exactly the image variant's 0091 lines (tested). Then one
identity-only start (warm-up, greedy, needle 150K; ESTIMATED 12 min with its start) appended to the first chained
window after UP2-ADOPT qualifies it.
