# L-SUITE: legal task suite from public documents

The quality bar for promotions and for any model-changing decision (bit reallocation, pruning, display-backed KV, a new
drafter). It is not run in every speed window. Spec: F-METHOD plan section 3.5 (`documents/plans/lanes/F-METHOD.plan.md`),
brief `documents/plans/lanes/L-SUITE.md`. A small suite is a floor, not proof of parity.

## What it contains

`tasks.v1.json` holds 46 tasks built offline from the pinned public corpus (`bench-data`: 351 CourtListener opinions,
61 RECAP filings). Every prompt is one user message: a fixed instruction, the materials, and the question. The
instruction tells the model to reply `NOT IN RECORD` when the materials do not answer the question, so every
answerable task also tests that the model does not refuse.

| Family | n | Task | Scoring |
|---|---|---|---|
| quote | 12 | Quote verbatim the sentence that contains a registered 5-word phrase (5 SC family, 2 SC civil, 3 other-state family, 2 Fourth Circuit) | the expected sentence is a substring of the reply after normalization |
| figure | 12 | The dollar amount (5), date (4) or percentage (3) in the sentence with a registered phrase; the value occurs once in the document; 3 tasks come from RECAP filings | first value of that kind in the reply equals the key (canonical Decimal / ISO date) |
| citation | 12 | The full citation of a case the opinion cites. 8 in South Carolina style (S.C. Reports, then S.E.2d, then `(Ct. App. year)` or `(year)`), 4 as the source gives them (4th Cir. x2, Tenn., N.C. App.) | each reporter cite in order, then a year parenthetical with the right court; SC rule: Ct. App. required for Court of Appeals cases, refused for Supreme Court cases |
| refusal | 4 | A plausible question the document cannot answer (fee amount, marriage date, GAL name, marital-estate percentage) | reply carries `NOT IN RECORD` and invents no value of the asked kind |
| needle | 6 | One synthetic docket number (`2031-DR-..`) planted at depth 0.05 to 0.95 of a 524,665-character haystack of public opinions (about 150K tokens at the 3.5 chars/token planning ratio; rows record the real count) | the docket number appears in the reply |

Normalization (both sides, every family): NFKC, curly quotes to straight, Unicode dashes to `-`, every whitespace run
to one space. A reply that contains `NOT IN RECORD` fails every answerable task.

## Data boundary and licences

- Opinions are public domain (government edicts doctrine), so their spans sit in the task file under `value` for review.
- RECAP filings are marked "never redistributed". Their 3 tasks store only character offsets and sha256 pins. The
  phrase and the key are rebuilt from the corpus at run time. A test checks that no four consecutive source words from a
  filing appear anywhere in the task.
- The needles are synthetic. Each docket number and the phrase "sealed companion matter" are checked absent from the
  whole corpus.
- Client-name screen: no suite input may contain a client's full name, not even a public opinion's counsel block. The
  build reads the client folder names (open and `!Closed`) and the matter-cache exclusion list and refuses every
  document whose rendered text holds a surname and first name in either order ("First Last", "First M. Last",
  "Last First", "Last, First"). Names are needles in memory only: never written, printed or hashed, and no list of
  screened documents is stored. The screen covers every single-document task and every haystack opinion.
- The task file uses none of the data-boundary text keys (`text`, `prompt`, `answer`, `content`, ...). It is under
  5 MiB and has no secret patterns. DOC2's `tools/data_boundary.py --staged` passed on every commit.
- The runner never stores generated text. `rows.jsonl` holds the outcome, a reason code, the reply length, token
  counts, and sha256 fingerprints of the sent prompt and of the other sent settings. Run outputs belong in the
  campaign directory, never in git; the runner refuses an `--out` inside the repository.

## Build and check (offline, CPU)

```bash
python3 -m offload.lsuite.build --bench-data /opt/dsv41/data/bench-data           # CHECK PASS = byte-identical
python3 -m offload.lsuite.build --bench-data /opt/dsv41/data/bench-data --write   # only for a new suite version
python3 -m pytest -q offload/lsuite && python3 -m mypy --strict offload/lsuite
```

Both build modes need the client-name screen sources (defaults `/opt/dsv41/shared/client folders` and
`/opt/dsv41/data/private/matter-exclusions.txt`; `--client-files` and `--matter-exclusions` override),
so they run on Controller; a missing source refuses the build (exit 2). The runner does not need them: it only rebuilds
the already-screened selection from the corpus.

Selection is deterministic. Documents are ranked by `sha256("lsuite-v1|family|doc_id")`, each single-document task
uses a different document (40 in all), and the candidate nearest a target depth wins. Every materialization
re-verifies each document and span pin, inline value, licence class, probe absence and the haystack hash. Changing
the suite means a new file (`tasks.v2.json`, suite `lsuite-v2`) and a new rule id, never an edit in place.

## Running it (coordinator)

```bash
# Plan only (default): no network. Verifies the tasks against the corpus and prints the ESTIMATED time.
python3 -m offload.lsuite.run_suite --bench-data /opt/dsv41/data/bench-data --arm X
# One arm (from Controller; on sparkhead the default --bench-data ~/dsv41-deploy/data has the same pins):
python3 -m offload.lsuite.run_suite --bench-data /opt/dsv41/data/bench-data --arm X \
    --url http://HEAD_IP:8001 --out "$CAMPAIGN_LOGS_ROOT/DSV41-OFFLOAD/<window>/lsuite-X" --execute
python3 -m offload.lsuite.run_suite ... --arm C --url http://HEAD_IP:8000 --out .../lsuite-C --execute
python3 -m offload.lsuite.compare .../lsuite-C .../lsuite-X --json .../lsuite-verdict.json
```

Each request runs alone (B1), with temperature 0, seed 0, thinking off, priority 10 and a fresh `cache_salt`, so
every prefill is cold. The repetition, presence and frequency penalties are pinned to their neutral values (1.0, 0, 0):
a request that omits them takes the served model directory's `generation_config.json` defaults, and penalties apply
before the greedy argmax.

Idle gate (COMMON hard rule 2): before each task the runner reads the arm's `/metrics` and sends only when
`vllm:num_requests_running` + `vllm:num_requests_waiting` is 0. It polls every `--idle-poll` seconds (default 5) for
up to `--idle-wait` seconds (default 600), then stops with a partial run (exit 1). After each reply it reads the gauges
again; anything else running or waiting turns that row into error `concurrent`, which makes compare.py INVALID. A
`/metrics` page without both gauges fails loud. Confirm once in the first smoke window that an idle arm reports 0
right after a reply (no row marked `concurrent`). `--tasks quote-01,needle-06` runs a smoke subset; compare.py refuses a partial run.

## Pre-registered pass rule (`lsuite-rule-v1`)

A task is lost when the control passes it and the candidate fails it.

- **FAIL** if the candidate loses any quote or needle task, or loses more than one task across figure, citation and
  refusal.
- **PASS** otherwise. Gains are reported and never offset a loss.
- **INVALID** if either run is partial or has a request error, if any row carries reasoning text (thinking was not
  off), if both directories are the same run, or if the runs differ in suite, task file, request settings, priority,
  corpus pins, scorer code (`code_sha256` in `run.json`), task set, or any task's sent prompt (`request_sha256`) or
  sent settings (`body_sha256`: the whole request body minus prompt and model, the salt counted as its per-task
  policy). An INVALID result decides nothing; rerun.

## Time (ESTIMATED)

About 10.1 minutes per arm at B1: 46 requests, roughly 295K prompt tokens for the 40 short tasks and 900K for the
6 needles. The figure assumes about 2,170 prompt tokens per second (archive, `offload/results/COMPARISON.md`: 150K cold TTFT
68.7 to 69.7 s, 32K 14.5 to 14.7 s), about 50 decode tokens per second, and 0.5 s overhead per request. The needles take about 7 minutes of
that. If the corpus tokenizes at 3.0 characters per token instead of 3.5, add about 1.5 minutes.
