Status 2026-10-23: Benchmark tooling in use; only coordinator executes generation suites. Measured results and W4 percentage correction are in COMPARISON. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Extended benchmark harness (S3)

`bench2.py` is a copy and extension of the deployment's `bench.py`; the original is unchanged. Legal workload strings, fictional-record builder, and public CourtListener/RECAP builder are preserved. No new runtime dependencies: Python standard library only. Run from any directory.

Only the coordinator runs generation benchmarks. This lane sent **zero** generation requests. Connectivity-only validation (no `/tokenize` or completions):

```bash
source ~/.config/ndm/paths.sh
python3 offload/bench/bench2.py --dry-run --out offload/bench/logs/connectivity.jsonl
```

The real corpus files are ignored by git and absent from the lane worktree. Point `--data-dir` to an existing directory containing `opinions_cl.jsonl` and `filings_recap.jsonl`. Do not substitute synthetic filler in an A/B comparison. Missing/insufficient real corpus fails explicitly.

Coordinator examples, after reserving an idle window and arranging HOLD with other workloads:

```bash
source ~/.config/ndm/paths.sh
python3 offload/bench/bench2.py --label A --ctx 0 32000 150000 --workload legal synth --data-dir /path/to/existing/data --concurrency 1 2 3 --mixed --mixed-short-ctx 8000 --max-tokens 1500 --reps 3 --thinking --reasoning-effort high --priority 10 --cache-mode cold --out offload/results/A-cold.jsonl
python3 offload/bench/bench2.py --label A --ctx 32000 --workload synth --data-dir /path/to/existing/data --concurrency 1 2 3 --max-tokens 1500 --reps 4 --cache-mode warm --out offload/results/A-warm.jsonl
python3 offload/bench/report.py offload/results/A-cold.jsonl offload/results/A-warm.jsonl --out offload/results/comparison.md
```

Match all flags and corpus across variants. Repeat warm groups at least twice: rep zero primes a fresh salt and is labelled `cold`; later reps reuse exactly the prompt/salt and are labelled `warm_repeat`. Cold reps reuse the same workload text but get fresh salts. Prompt SHA256 is retained per request; prompt tokens come from `/tokenize` with the same messages/template kwargs used by generation. `target_ctx=0` means instruction only. Nonzero context budgets cover the entire occupied chat prompt, including template/instruction. A character-boundary trim can finish a few tokens below target (at most 16); actual occupied and server usage counts are retained. Output reservation is enforced against the smaller of `--max-model-len` and the advertised live model limit.

`--mixed` adds a separate case at each context: two `--mixed-short-ctx` requests start together, then the long-context request arrives after both have emitted tokens and `--mixed-delay` seconds have elapsed. If both are no longer actively generating, the mixed case fails rather than claiming overlap. Increase the output budget or reduce delay if needed. Normal groups use a start barrier for 1/2/3 requests. Every group requires server running/waiting gauges to be present and zero before submission. The coordinator must isolate the run from external requests: speculation counters are global and cannot be attributed per stream.

## Accounting

- Raw JSONL includes each request and its run summary, full model-list identity/root, inference settings, cache salt, reserved output, measured prompt count, usage, stream timing and telemetry. Text is opt-in via `--keep-text`; these files stay local.
- `ttft_s`: first reasoning or answer text; `first_visible_s`: first answer text. Missing channels are null. `total_s` ends at stream completion, before post-hoc tokenization. `finish=length` is budget exhaustion, not completion of a full legal task.
- Authoritative delivered totals use server `completion_tokens` only. Rejected speculative proposals are never delivered tokens. Speculation summary separately reports proposed/accepted counts, rounds, **accepted/round without adding a bonus token**, and per-position accepted/round.
- Per-channel counts use server reasoning usage where available; answer count is then completion minus reasoning, explicitly labelled as including control tokens. Otherwise `/tokenize` counts emitted channel text and labels it `retokenized_text`. This is not guaranteed to recover original generation tokens. No token count is inferred from SSE event counts. Per-channel decode rates use `(count-1)/(last-first text time)` and retain this count-source limitation.
- `decode_tps` uses server completion usage over first-to-last delivered text; aggregate rate uses delivered totals over group wall time including prefill and staggered arrivals. Post-hoc tokenization and telemetry shutdown are excluded; `client_elapsed_s` separately includes postprocessing. Reported request rates and aggregate rates measure different intervals.
- Missing speculation metrics remain null. Resets or disappearing counters fail the run and write an error summary. `--metrics-settle` (default 2 seconds) waits for periodically published counters after the group, outside throughput timing. Match this to the engine's actual publish cadence if changed; external traffic during this wait contaminates global deltas.
- Comparison tables show mean ± sample standard deviation of repeated **run means**, keeping cache/workload/concurrency/sampling separate. They do not count concurrent streams as independent repetitions. Errors remain visible and excluded from success averages. With one repetition SD is shown as zero; this is not evidence of repeatability.

## Telemetry

Every 2 seconds, bounded parallel SSH reads sample `sparkhead`, `sparkworker`, and `x86host`; an additional terminal sample captures the last counter interval. Slow probes record `sample_duration_s` rather than pretend to maintain a 2-second cadence. No GPU contexts are created. The Spark GPU's unified memory usage may appear `[N/A]`; `/proc/meminfo` is retained.

X86host GPU queries resolve only UUID prefixes `c4e7a371` and `7744b44b`. `--process-name NAME` (repeatable) matches exact `/proc/comm` or executable basename; use a process's actual name, e.g. `python3`, if workers run under an interpreter. Each observation includes PID and start ticks to distinguish restarts, RSS and cumulative major faults. No names by default means an empty process list. NICs default to `enp130s0f0np0` and `enp130s0f1np1`; override both with `--x86host-nic`. Byte, retransmit, retry, error, drop and discard counters and interval deltas are retained independently; do not sum overlapping ethtool counters. Baseline `nic_delta` is empty. Negative deltas indicate reset/wrap and must not be interpreted as traffic.

`--extra-metrics URL` adds raw per-sample scrapes for offload timings. Every probe error is printed and recorded. Dry-run exits nonzero on telemetry errors; generation results retain errors so missing telemetry cannot silently look healthy. Run summaries retain raw observations instead of inventing peaks between samples.

## Local checks

```bash
source ~/.config/ndm/paths.sh
python3 -m unittest discover -s offload/bench -p 'test_*.py'
python3 -m mypy --strict offload/bench/
python3 -m ruff check offload/bench/
```

Tests mock transport and telemetry. They check simultaneous reasoning/content parsing, usage versus retokenization, truncated streams, speculation deltas/reset/positions, idle refusal, full-prompt sizing, request settings, actual concurrency/mixed overlap in fixture threads, JSONL/report accounting, duplicate refusal and zero-generation dry-run behavior. They are not model-performance or legal-quality benchmarks.

## Rate convention

Use `R = (1 + E[A]) / C` for steady B1: accepted draft tokens A exclude the correction/bonus, and C is complete
round time. Actual committed counts govern clipped/terminal/mixed rounds. SSE p50 remains a delivery metric,
not an engine/GPU timer. See [repo metric and paper-unit audit](../evidence/THROUGHPUT.md).
