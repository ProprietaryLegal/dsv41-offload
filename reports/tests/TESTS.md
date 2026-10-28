# Test results for the published trees

Run 2026-10-28 on the exact trees published here, one fresh single-commit git repository per branch. CPU only (no GPU visible), Python 3.13.15, pytest 9.0.2, mypy 2.3.1, inside a network namespace with only loopback, as an unprivileged user; `ssh`, `scp`, `curl`, `docker` and `nvidia-smi` are replaced by stubs that refuse. Command, from the repository root:

```bash
python3 -m pytest -p no:cacheprovider -q --continue-on-collection-errors offload
```

`test_*.py` files that contain no test function are GPU-host scripts (some parse command-line arguments when imported); they are passed to `--ignore` and counted below as "scripts".

## How each failure is classified

Every failing or erroring test of the whole-tree run was re-run three ways:

1. on the **private source commit** the branch was cut from (unmodified, in the same sandbox). A test that fails there too is an **environment** failure: it also fails on the private source commit in the same CPU-only, single-commit sandbox (pre-existing: a GPU, serving hosts or deployment state, git history that a single-commit tree lacks, or a failure the private commit already has);
2. **alone** (its own file only) on the published tree. A test that passes alone failed only because of the whole-tree session (module-name collisions or state shared between test files): **passes alone**;
3. on the private source commit again, with our host data directories hidden (an empty file system mounted over them). A test that fails there **needs our host data**: the serving image's vLLM source tree, campaign scripts, benchmark data or a second worktree (the published tree names those places as paths under `/opt/dsv41`).

What remains passes on the private commit even without our host data, so the **publication caused it**: it needs a withheld prompt set or a file outside the data boundary. Any other case is listed by node id.

## Results by branch

| Branch | Source | Passed | Failed | Errors | Skipped | Failing ids | Environment | Passes alone | Needs host data | Caused by publication | Scripts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `main` | `27d3ca2b78b1` | 2,559 | 556 | 223 | 300 | 714 | 339 | 14 | 353 | 8 | 3 |
| `production` | `596ba1cf4e3b` | 420 | 96 | 21 | 54 | 117 | 111 | 0 | 5 | 1 | 3 |
| `lane/nbw-win2` | `e120ccf31c33` | 2,742 | 571 | 223 | 319 | 727 | 339 | 16 | 364 | 8 | 3 |
| `lane/pc1-win2` | `8f155598b902` | 2,921 | 629 | 224 | 319 | 784 | 404 | 16 | 356 | 8 | 3 |
| `lane/pp4` | `c4ccc05a3a42` | 2,890 | 596 | 267 | 308 | 798 | 396 | 16 | 378 | 8 | 3 |
| `lane/pp4-promote` | `1d3b5b749548` | 3,170 | 652 | 267 | 340 | 834 | 432 | 16 | 378 | 8 | 3 |
| `lane/pp4-win2` | `81969f70009a` | 3,132 | 595 | 267 | 341 | 797 | 398 | 13 | 378 | 8 | 3 |
| `lane/sp-prefill` | `d3070736b078` | 2,651 | 575 | 223 | 306 | 733 | 339 | 13 | 373 | 8 | 3 |
| `lane/tv1` | `25faa8473297` | 2,695 | 650 | 254 | 308 | 839 | 357 | 14 | 460 | 8 | 3 |
| `lane/up2-adopt` | `924fc834a091` | 3,079 | 707 | 245 | 319 | 880 | 482 | 13 | 377 | 8 | 3 |
| `lane/up3-win2` | `d413fa5322ea` | 3,021 | 587 | 263 | 324 | 783 | 359 | 13 | 403 | 8 | 3 |
| `lane/upfix` | `42fe86a7f954` | 2,780 | 568 | 223 | 319 | 724 | 339 | 15 | 362 | 8 | 3 |
| `lane/upx` | `d88d2cbf90c5` | 2,797 | 608 | 224 | 324 | 756 | 358 | 13 | 377 | 8 | 3 |
| `lane/win2` | `06b505a3656d` | 2,823 | 559 | 227 | 320 | 721 | 339 | 15 | 357 | 10 | 3 |
| `lane/census` | `54baa89a94fc` | 2,796 | 578 | 223 | 319 | 732 | 339 | 16 | 369 | 8 | 3 |
| `lane/lsuite` | `1a7e70088e76` | 2,761 | 556 | 226 | 311 | 717 | 339 | 13 | 353 | 12 | 3 |
| `lane/matter-cache` | `0962424b2df2` | 474 | 106 | 22 | 54 | 128 | 119 | 1 | 7 | 1 | 3 |
| `lane/dw1` | `58d770f73f97` | 2,553 | 556 | 223 | 300 | 714 | 339 | 14 | 353 | 8 | 3 |
| `lane/dp5` | `dbaaafc1d2dc` | 2,551 | 557 | 223 | 300 | 715 | 351 | 3 | 353 | 8 | 3 |
| `focus:dsv41-window-harness` | `subset of lane/win2` | 391 | 2 | 43 | 20 | 45 | 39 | 0 | 4 | 2 | 0 |
| `focus:dsv41-sampled-drafting` | `subset of lane/pp4-win2` | 529 | 75 | 75 | 28 | 146 | 84 | 0 | 54 | 8 | 0 |

"Failed", "Errors" and "Skipped" are pytest's own counts (a test that fails and then errors in teardown counts twice; a module that cannot be imported counts as one error). "Failing ids" is the number of distinct failing or erroring node ids; the four classes add up to it.

## Failures caused by the publication, by reason

| Branch | Withheld prompt set | Needs full tree (focused repos) | Outside data boundary | Other |
|---|---:|---:|---:|---:|
| `main` | 4 | 0 | 4 | 0 |
| `production` | 0 | 0 | 1 | 0 |
| `lane/nbw-win2` | 4 | 0 | 4 | 0 |
| `lane/pc1-win2` | 4 | 0 | 4 | 0 |
| `lane/pp4` | 4 | 0 | 4 | 0 |
| `lane/pp4-promote` | 4 | 0 | 4 | 0 |
| `lane/pp4-win2` | 4 | 0 | 4 | 0 |
| `lane/sp-prefill` | 4 | 0 | 4 | 0 |
| `lane/tv1` | 4 | 0 | 4 | 0 |
| `lane/up2-adopt` | 4 | 0 | 4 | 0 |
| `lane/up3-win2` | 4 | 0 | 4 | 0 |
| `lane/upfix` | 4 | 0 | 4 | 0 |
| `lane/upx` | 4 | 0 | 4 | 0 |
| `lane/win2` | 6 | 0 | 4 | 0 |
| `lane/census` | 4 | 0 | 4 | 0 |
| `lane/lsuite` | 4 | 0 | 8 | 0 |
| `lane/matter-cache` | 0 | 0 | 1 | 0 |
| `lane/dw1` | 4 | 0 | 4 | 0 |
| `lane/dp5` | 4 | 0 | 4 | 0 |
| `focus:dsv41-window-harness` | 2 | 0 | 0 | 0 |
| `focus:dsv41-sampled-drafting` | 0 | 8 | 0 | 0 |

- **Withheld prompt set**: needs a withheld prompt set.
- **Needs full tree (focused repos)**: focused repos only: needs a file of the full tree that the focused repo does not carry (the same test passes in the dsv41-offload branch it was cut from).
- **Outside data boundary**: reads a file outside the data boundary (results, evidence, upstream metadata snapshots, suite data).

## mypy --strict

Typed packages of the newest work, where the branch has them (`mypy --strict --explicit-package-bases --follow-imports=silent`; a directory of plain scripts goes on `MYPYPATH`). Each cell: errors / files checked; "private" is the same check on the private source commit (for a focused repo, on the full lane it was cut from).

| Branch | census | acc_split | lsuite | matter | win2_r3 | dw1_dp5 | gb10_budget | bench |
|---|---|---|---|---|---|---|---|---|
| `main` | — | 0 / 5 files (private 0) | — | — | — | — | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `production` | — | — | — | — | — | — | — | 1 / 12 files (private 1) |
| `lane/nbw-win2` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/pc1-win2` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/pp4` | — | — | — | — | — | — | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `lane/pp4-promote` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/pp4-win2` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/sp-prefill` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/tv1` | — | — | — | — | — | — | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `lane/up2-adopt` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/up3-win2` | — | — | — | — | 0 / 7 files (private 0) | — | 0 / 13 files (private 0) | 1 / 12 files (private 1) |
| `lane/upfix` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/upx` | — | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/win2` | — | — | — | — | 0 / 7 files (private 0) | — | 0 / 13 files (private 0) | 1 / 12 files (private 1) |
| `lane/census` | 0 / 22 files (private 0) | — | — | — | 0 / 2 files (private 0) | — | 0 / 12 files (private 0) | 1 / 12 files (private 1) |
| `lane/lsuite` | — | — | 0 / 14 files (private 0) | — | — | — | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `lane/matter-cache` | — | — | — | 0 / 4 files (private 0) | — | — | — | 1 / 12 files (private 1) |
| `lane/dw1` | — | — | — | — | — | 53 / 5 files (private 53) | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `lane/dp5` | — | — | — | — | — | 24 / 3 files (private 24) | 0 / 11 files (private 0) | 1 / 12 files (private 1) |
| `focus:dsv41-window-harness` | — | — | — | — | 0 / 7 files (private 0) | — | 0 / 7 files (private 0) | 0 / 2 files (private 1) |
| `focus:dsv41-sampled-drafting` | — | — | — | — | 0 / 1 files (private 0) | — | 0 / 4 files (private 0) | 0 / 2 files (private 1) |
