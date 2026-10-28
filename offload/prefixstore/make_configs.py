#!/usr/bin/env python3
"""Generate Stage A experiments from K1's generator; never start a service."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_overlay import verify_checkout


def layout_sha() -> str:
    # Pin target runtime bytes including all K1 and RP1 overlay files.
    manifests = [json.loads((ROOT / 'offload/kvpack/MANIFEST.json').read_text()), verify_checkout()]
    files = sorted((e['path'], e['overlay_sha256']) for m in manifests for e in m['files'])
    return hashlib.sha256(json.dumps(files, separators=(',', ':')).encode()).hexdigest()


def render(name: str, cpu_bytes: int, one_m: bool) -> str:
    spec = importlib.util.spec_from_file_location('k1_configs', ROOT / 'offload/kvpack/make_configs.py')
    assert spec and spec.loader
    k1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k1)
    text: str = k1.render(name.removesuffix('.env'), 'RP1 Stage A: packed prefixes to CPU + local SSD.',
                          1_048_576 if one_m else None, '3', k1.ASYNC_BASE)
    m = verify_checkout()
    pairs = ' '.join(f'offload/prefixstore/overlay/{e["path"]}={e["container_path"]}' for e in m['files'])
    text += f'OVERLAYS="$OVERLAYS {pairs}"\n'
    text += f'''# RP1 coordinator-only candidate. PREFIX_ENABLE=0 gives the same-overlay no-connector control.
PREFIX_ENABLE=${{PREFIX_ENABLE-1}}
PREFIX_CPU_BYTES=${{PREFIX_CPU_BYTES-{cpu_bytes}}}
PREFIX_LOAD_FRACTION=${{PREFIX_LOAD_FRACTION-0.2}} # four requests x .2, leaving 20% for stores/in-flight promotions
PREFIX_IO_CPUS=${{PREFIX_IO_CPUS-0-4,10-13}} # verify A725 topology/allowed CPUs before use
PREFIX_MN_VERIFY=${{PREFIX_MN_VERIFY-0}}    # diagnostic: both ranks store for byte comparison
if [ "$PREFIX_ENABLE" = 1 ]; then
  PREFIX_MODEL_SHA256=${{PREFIX_MODEL_SHA256:?SHA256 of exact checkpoint manifest required (not model path)}}
  case "$PREFIX_MODEL_SHA256" in *[!0-9a-f]*) echo "PREFIX_MODEL_SHA256 must be lowercase SHA256" >&2; return 1;; esac
  [ "${{#PREFIX_MODEL_SHA256}}" = 64 ] || {{ echo "PREFIX_MODEL_SHA256 must be 64 hex characters" >&2; return 1; }}
  case "$PREFIX_CPU_BYTES" in ''|*[!0-9]*) echo "PREFIX_CPU_BYTES must be positive bytes" >&2; return 1;; esac
  [ "$PREFIX_CPU_BYTES" -ge 16777216 ] && [ "$PREFIX_CPU_BYTES" -le 2147483648 ] || {{ echo "PREFIX_CPU_BYTES outside 16MiB..2GiB" >&2; return 1; }}
  case "$PREFIX_LOAD_FRACTION" in 0.05|0.1|0.15|0.2) ;; *) echo "four-request load fraction must be .05/.1/.15/.2 (with leading zero)" >&2; return 1;; esac
  case "$PREFIX_IO_CPUS" in ''|*[!0-9,-]*) echo "PREFIX_IO_CPUS must be an explicit CPU list" >&2; return 1;; esac
  case "$PREFIX_MN_VERIFY" in 0|1) ;; *) echo "PREFIX_MN_VERIFY must be 0 or 1" >&2; return 1;; esac
  PREFIX_TRANSFER_JSON="{{\\"kv_connector\\":\\"OffloadingConnector\\",\\"kv_role\\":\\"kv_both\\",\\"kv_connector_extra_config\\":{{\\"spec_name\\":\\"TieringOffloadingSpec\\",\\"cpu_bytes_to_use\\":$PREFIX_CPU_BYTES,\\"prefix_store_identity\\":{{\\"checkpoint_sha256\\":\\"$PREFIX_MODEL_SHA256\\",\\"layout_sha256\\":\\"{layout_sha()}\\"}},\\"secondary_tiers\\":[{{\\"type\\":\\"fs\\",\\"root_dir\\":\\"/root/.cache/kv_prefix_rp1\\",\\"n_read_threads\\":2,\\"n_write_threads\\":1}}]}}}}"
  EXP_EXTRA_ARGS="${{EXP_EXTRA_ARGS:+$EXP_EXTRA_ARGS }}--kv-transfer-config '$PREFIX_TRANSFER_JSON'"
  EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_DSV41_KV_CRC=1 VLLM_DSV41_OFFLOAD_EXPANDABLE_OK=1 VLLM_DSV41_OFFLOAD_MULTINODE=1 VLLM_DSV41_OFFLOAD_MN_VERIFY=$PREFIX_MN_VERIFY VLLM_DSV41_OFFLOAD_MN_STAGE_MB=8 VLLM_DSV41_OFFLOAD_LOAD_FRACTION=$PREFIX_LOAD_FRACTION VLLM_DSV41_OFFLOAD_IO_CPUS=$PREFIX_IO_CPUS"
elif [ "$PREFIX_ENABLE" != 0 ]; then
  echo "PREFIX_ENABLE must be 0 or 1" >&2; return 1
fi
'''
    return text


CONFIGS = {'D-kv6-async-pack-prefix.env': (268435456, False),
           'D-kv6-async-pack-prefix-1g.env': (1073741824, False),
           'D-kv6-async-pack-prefix-1m.env': (268435456, True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, (cpu_bytes, one_m) in CONFIGS.items():
        text = render(name, cpu_bytes, one_m)
        path = ROOT / 'offload/launch/configs' / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError(f'config drift: {name}')
        else:
            path.write_text(text)
    print(f'{"verified" if args.check else "wrote"} {len(CONFIGS)} RP1 configs')


if __name__ == '__main__':
    main()
