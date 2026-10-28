# offload/upx — the unowned upstream gaps from UPAUDIT (R-UPX): 0085b, 0064 + CPU pin, 0067

Everything here is **off unless an arm turns it on**. Production and every existing config are untouched: nothing here is
mounted by `dsv41.env`, by K1's promotion package, or by any other config. CPU-built and CPU-tested on Controller; no
Spark, x86host or production contact.

| Item | What | Where | Switch |
|---|---|---|---|
| UPAUDIT B33 (fix 5) | henryous patch 0085b: after a long prefill and 5 s without a request, each rank empties its allocator cache once (GB10: the cache comes back as MemAvailable) | `build_0085b.py` → `overlay/` (engine/core.py, worker/gpu_worker.py), `patches/0085b-v5.patch` (the exact v5 port), `MANIFEST-0085b.json` | `UPX.env` arm `UPX_0085B=tokens` (`VLLM_DSV41_RELEASE_TOKENS`) or `file` (`VLLM_DSV41_RELEASE_FILE`, live toggle); `x0085b.env` appends it to any base |
| UPAUDIT B13, C2, B16 (fixes 4, 9) | 0064 (33–63-row MoE calls as exl3_smoe row chunks), `EXL3_CPU_PIN=5-9,15-19`, 0067 (fused SWA metadata) | R-UP3's arms (`offload/up3`, production-exact base, read at `ac23d4f`, never edited here); window groups as data in `windows.py`, band cells and band probes in `band.py` | `UP3_ARM=split|cpu-pin|vor` (R-UP3's `window.env`) |

## Commands (repository root; Controller, CPU only)

```
python3 -m offload.upx.build_0085b check     # overlay + port patch + MANIFEST-0085b.json equal a fresh build
python3 -m offload.upx.configs arms          # 0085b arm -> UPX.env environment
python3 -m offload.upx.windows plan          # window groups, gates, ESTIMATED HOLD minutes
python3 -m offload.upx.battery_window        # UPX-0085B runner: dry plan (--execute: coordinator, attended)
CUDA_VISIBLE_DEVICES= python3 -m pytest offload/upx
# test_chain_placeholders uses real user scopes (test_scoped_remote's sandbox): TMUX unset, and TMPDIR plus a
# short TMUX_TMPDIR (socket path < 108 bytes) both private and outside /tmp, e.g.
#   env -u TMUX -u TMUX_PANE TMPDIR=<private dir> TMUX_TMPDIR=<short private dir> CUDA_VISIBLE_DEVICES= python3 -m pytest offload/upx
python3 -m mypy --strict --explicit-package-bases offload/upx
```

Window designs, gates and HOLD estimates: `WINDOWS.md` (data in `windows.py`). The 0085b memory measurement:
`MEASURE-0085B.md` (arithmetic in `measure_0085b.py`, runner `battery_window.py`, requests `battery_cells.py`).
Licences: `NOTICE`.
