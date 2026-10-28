Status 2026-10-23: Stage A cleared for GB10 P0–P2 after S15; no physical qualification or production adoption. Stage B remains a separate unmounted prototype; 1M extension dropped. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# RP1: prefixes beyond the GPU pool

Stage A rebuilds the upstream CPU + persistent SSD KV tier on image v5 and K1's page-packing overlay. It is ready
for coordinator GB10 tests; no Spark/GPU tests or service changes were made by this lane. Stage B is a separate
x86host RAM/RDMA design and working CPU prototype with fake clients, described in [remote/README.md](remote/README.md).
It is not connected to vLLM.

```bash
source ~/.config/ndm/paths.sh
python3 offload/prefixstore/build_overlay.py
python3 offload/prefixstore/make_configs.py --check
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m unittest discover -s offload/prefixstore/tests -v
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m mypy --strict --follow-imports=skip offload/prefixstore/{build_overlay,make_configs,verify_overlay,restart_probe}.py
```

The rebuild reads only the pinned local donor and image-source snapshots; it runs patch(1), never upstream launchers.
[MANIFEST.json](MANIFEST.json) pins four donor patches, the local safety patch, the seven overlay files and K1's
manifest. K1's 13 mounted files remain byte-identical. [AUDIT.md](AUDIT.md) records the overlap and lifecycle audit.
Use [GB10-TEST-PLAN.md](GB10-TEST-PLAN.md) for all coordinator starts, stop lines, correctness gates and rollback.

The three generated configs are `D-kv6-async-pack-prefix.env` (256 MiB CPU tier/rank), `...-prefix-1g.env` (1 GiB),
and `...-prefix-1m.env` (1M model limit, 256 MiB). Each extends K1's generator and retains the async process proxy,
remote Engram, capacity drafter and 6 GiB GPU KV pool. `PREFIX_ENABLE=0` is the same-overlay no-connector control.
Enabled starts require `PREFIX_MODEL_SHA256`, the exact verified checkpoint-manifest SHA256. The caller must not
substitute a model name/path or carry the digest across changed weights. All launch controls are review candidates;
no production config is changed. `serve_exp.sh` has one inert optional EXTRA hook for connector arguments.

A 150K packed prefix is about 270 MB raw (272 MB aligned) per rank. One identical replicated copy is stored.
The donor-proven load path uses rank 0's file/CPU tier, then gloo to replace rank 1's chunks before GPU copy.
Four simultaneous reloads use at most a 0.2 share each. The 256 MiB tier therefore caps each request at **29,568
loaded tokens**, and computes the suffix. Larger cold prefixes do not imply larger hits. A 1 GiB tier caps around
118K; full 150K requires roughly 1.3 GiB landing memory/rank at .2 plus measured >=5 GiB MemAvailable throughout.
Do not spend that memory before the GB10 gate proves it exists.

`restart_probe.py` stores needle prompts and cache salts outside git, streams TTFT, checks codes and cold-control
misses, and compares first-token top-20 distributions with an OTHER bucket. This coarse KL is a lower bound; it
cannot certify full-vocabulary KL. The correctness diagnostic fixes thinking off. Production speed comparisons keep
the campaign's reasoning/sampling settings and use the existing suite unchanged.
