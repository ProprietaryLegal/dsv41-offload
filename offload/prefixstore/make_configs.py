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


def overlay_pairs(text: str) -> list[tuple[str, str]]:
    """(repo source, container target) for every OVERLAYS assignment in a rendered config, in order."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith('OVERLAYS="'):
            continue
        for item in line[len('OVERLAYS="'):-1].split():
            if item == '$OVERLAYS':
                continue
            source, sep, target = item.partition('=')
            if not sep:
                raise ValueError(f'malformed overlay pair {item!r}')
            pairs.append((source, target))
    return pairs


def layout_files(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Container path -> sha256 of every file the overlay pairs mount (directories expanded)."""
    files: dict[str, str] = {}
    for source, target in pairs:
        path = ROOT / source
        members = sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts) \
            if path.is_dir() else [path]
        if not members:
            raise ValueError(f'overlay source {source} has no files')
        for member in members:
            mounted = target if member == path else target + '/' + member.relative_to(path).as_posix()
            if mounted in files:
                raise ValueError(f'two overlays mount {mounted}')
            files[mounted] = hashlib.sha256(member.read_bytes()).hexdigest()
    return files


def layout_sha(pairs: list[tuple[str, str]]) -> str:
    """MC1: the file-tier namespace pins the bytes of EVERY mounted overlay file (K1, remote DSpark/async proxy,
    spec_decode, RP1/MC1 and anything a future generator composes), not only the K1 and RP1 manifests. Any promotion
    that changes a mounted file lands in a new namespace, so the cache is cold after it by design (F-MATTER 3.2)."""
    files = layout_files(pairs)
    return hashlib.sha256(json.dumps(sorted(files.items()), separators=(',', ':')).encode()).hexdigest()


def render(name: str, cpu_bytes: int, one_m: bool) -> str:
    spec = importlib.util.spec_from_file_location('k1_configs', ROOT / 'offload/kvpack/make_configs.py')
    assert spec and spec.loader
    k1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k1)
    text: str = k1.render(name.removesuffix('.env'), 'RP1 Stage A: packed prefixes to CPU + local SSD.',
                          1_048_576 if one_m else None, '3', k1.ASYNC_BASE)
    m = verify_checkout()
    entries = list(m['files']) + list(m.get('new_files', []))
    pairs = ' '.join(f'offload/prefixstore/overlay/{e["path"]}={e["container_path"]}' for e in entries)
    text += f'OVERLAYS="$OVERLAYS {pairs}"\n'
    layout = layout_sha(overlay_pairs(text))
    text += f'''# RP1 coordinator-only candidate. PREFIX_ENABLE=0 gives the same-overlay no-connector control.
PREFIX_ENABLE=${{PREFIX_ENABLE-1}}
PREFIX_CPU_BYTES=${{PREFIX_CPU_BYTES-{cpu_bytes}}}
PREFIX_LOAD_FRACTION=${{PREFIX_LOAD_FRACTION-0.2}} # four requests x .2, leaving 20% for stores/in-flight promotions
PREFIX_IO_CPUS=${{PREFIX_IO_CPUS-0-4,10-13}} # verify A725 topology/allowed CPUs before use
PREFIX_MN_VERIFY=${{PREFIX_MN_VERIFY-0}}    # diagnostic: both ranks store for byte comparison
# MC1 matter tier (Tier 1). Paths are container paths under the launcher's /root/.cache mount (~/.cache/dsv41-serve).
PREFIX_MATTER_ROOT=${{PREFIX_MATTER_ROOT-/root/.cache/kv_matter_mc1}}
PREFIX_MATTER_EXCLUDED=${{PREFIX_MATTER_EXCLUDED-/root/.cache/dsv41-matter/matter-cache-excluded-ns.txt}}
PREFIX_MATTER_AUDIT=${{PREFIX_MATTER_AUDIT-/root/.cache/dsv41-matter/audit.jsonl}}
PREFIX_MATTER_MAX_GB=${{PREFIX_MATTER_MAX_GB-8}}       # tier byte bound (LRU over namespaces); size it to the measured free disk
PREFIX_MATTER_NS_MAX_GB=${{PREFIX_MATTER_NS_MAX_GB-4}}  # per-namespace byte bound (oldest files first)
PREFIX_MATTER_TTL_DAYS=${{PREFIX_MATTER_TTL_DAYS-30}}   # files unused this long expire (F-MATTER 3.7; NDM to confirm)
PREFIX_STREAM=${{PREFIX_STREAM-1}}          # MC1 streamed landing: matter-tier hits land window by window (no 0081 cap)
PREFIX_STREAM_MB=${{PREFIX_STREAM_MB-64}}   # per-request landing budget in the CPU tier (two windows), also <= load fraction
PREFIX_HASH_DUMP=${{PREFIX_HASH_DUMP-}}      # MC1 diagnostic: /root/.cache/<name> -> <name>_r<rank>.jsonl block hashes; empty = off
PREFIX_HASH_SEGMENT=${{PREFIX_HASH_SEGMENT-4096}}  # bytes per segment digest (localizes padding); 0 = whole-block sha only
if [ "$PREFIX_ENABLE" = 1 ]; then
  PREFIX_MODEL_SHA256=${{PREFIX_MODEL_SHA256:?SHA256 of exact checkpoint manifest required (not model path)}}
  case "$PREFIX_MODEL_SHA256" in *[!0-9a-f]*) echo "PREFIX_MODEL_SHA256 must be lowercase SHA256" >&2; return 1;; esac
  [ "${{#PREFIX_MODEL_SHA256}}" = 64 ] || {{ echo "PREFIX_MODEL_SHA256 must be 64 hex characters" >&2; return 1; }}
  case "$PREFIX_CPU_BYTES" in ''|*[!0-9]*) echo "PREFIX_CPU_BYTES must be positive bytes" >&2; return 1;; esac
  [ "$PREFIX_CPU_BYTES" -ge 16777216 ] && [ "$PREFIX_CPU_BYTES" -le 2147483648 ] || {{ echo "PREFIX_CPU_BYTES outside 16MiB..2GiB" >&2; return 1; }}
  case "$PREFIX_LOAD_FRACTION" in 0.05|0.1|0.15|0.2) ;; *) echo "four-request load fraction must be .05/.1/.15/.2 (with leading zero)" >&2; return 1;; esac
  case "$PREFIX_IO_CPUS" in ''|*[!0-9,-]*) echo "PREFIX_IO_CPUS must be an explicit CPU list" >&2; return 1;; esac
  case "$PREFIX_MN_VERIFY" in 0|1) ;; *) echo "PREFIX_MN_VERIFY must be 0 or 1" >&2; return 1;; esac
  for v in "$PREFIX_MATTER_MAX_GB" "$PREFIX_MATTER_NS_MAX_GB" "$PREFIX_MATTER_TTL_DAYS"; do
    case "$v" in ''|*[!0-9]*) echo "matter tier bounds and TTL must be whole numbers" >&2; return 1;; esac
  done
  [ "$PREFIX_MATTER_MAX_GB" -ge 1 ] && [ "$PREFIX_MATTER_MAX_GB" -le 4096 ] || {{ echo "PREFIX_MATTER_MAX_GB outside 1..4096" >&2; return 1; }}
  [ "$PREFIX_MATTER_NS_MAX_GB" -ge 1 ] && [ "$PREFIX_MATTER_NS_MAX_GB" -le "$PREFIX_MATTER_MAX_GB" ] || {{ echo "PREFIX_MATTER_NS_MAX_GB outside 1..PREFIX_MATTER_MAX_GB" >&2; return 1; }}
  [ "$PREFIX_MATTER_TTL_DAYS" -ge 1 ] && [ "$PREFIX_MATTER_TTL_DAYS" -le 365 ] || {{ echo "PREFIX_MATTER_TTL_DAYS outside 1..365" >&2; return 1; }}
  case "$PREFIX_STREAM" in 0|1) ;; *) echo "PREFIX_STREAM must be 0 or 1" >&2; return 1;; esac
  case "$PREFIX_STREAM_MB" in ''|*[!0-9]*) echo "PREFIX_STREAM_MB must be whole MiB" >&2; return 1;; esac
  [ "$PREFIX_STREAM_MB" -ge 1 ] && [ "$PREFIX_STREAM_MB" -le 1024 ] || {{ echo "PREFIX_STREAM_MB outside 1..1024" >&2; return 1; }}
  case "$PREFIX_HASH_DUMP" in "") ;; /root/.cache/*[!A-Za-z0-9._/-]*|*..*|*/) echo "PREFIX_HASH_DUMP: [A-Za-z0-9._/-] under /root/.cache/, no trailing /" >&2; return 1;; /root/.cache/?*) ;; *) echo "PREFIX_HASH_DUMP must be under /root/.cache/" >&2; return 1;; esac
  case "$PREFIX_HASH_SEGMENT" in ''|*[!0-9]*) echo "PREFIX_HASH_SEGMENT must be whole bytes" >&2; return 1;; esac
  # The exclusion hash file is pinned per start (derive_excluded_ns.py prints the value); no default exists.
  PREFIX_MATTER_EXCLUDED_SHA256=${{PREFIX_MATTER_EXCLUDED_SHA256:?SHA256 of the staged excluded-namespace hash file required}}
  case "$PREFIX_MATTER_EXCLUDED_SHA256" in *[!0-9a-f]*) echo "PREFIX_MATTER_EXCLUDED_SHA256 must be lowercase SHA256" >&2; return 1;; esac
  [ "${{#PREFIX_MATTER_EXCLUDED_SHA256}}" = 64 ] || {{ echo "PREFIX_MATTER_EXCLUDED_SHA256 must be 64 hex characters" >&2; return 1; }}
  for v in "$PREFIX_MATTER_ROOT" "$PREFIX_MATTER_EXCLUDED" "$PREFIX_MATTER_AUDIT"; do
    case "$v" in /root/.cache/*[!A-Za-z0-9._/-]*|*..*) echo "matter tier path $v: only [A-Za-z0-9._/-] under /root/.cache/" >&2; return 1;; /root/.cache/?*) ;; *) echo "matter tier path $v must be under /root/.cache/" >&2; return 1;; esac
  done
  PREFIX_TRANSFER_JSON="{{\\"kv_connector\\":\\"OffloadingConnector\\",\\"kv_role\\":\\"kv_both\\",\\"kv_connector_extra_config\\":{{\\"spec_name\\":\\"TieringOffloadingSpec\\",\\"cpu_bytes_to_use\\":$PREFIX_CPU_BYTES,\\"prefix_store_identity\\":{{\\"checkpoint_sha256\\":\\"$PREFIX_MODEL_SHA256\\",\\"layout_sha256\\":\\"{layout}\\"}},\\"secondary_tiers\\":[{{\\"type\\":\\"Dsv41MatterTier\\",\\"module_path\\":\\"vllm.v1.kv_offload.tiering.fs.dsv41_matter_tier\\",\\"root_dir\\":\\"$PREFIX_MATTER_ROOT\\",\\"excluded_ns_file\\":\\"$PREFIX_MATTER_EXCLUDED\\",\\"excluded_ns_sha256\\":\\"$PREFIX_MATTER_EXCLUDED_SHA256\\",\\"audit_path\\":\\"$PREFIX_MATTER_AUDIT\\",\\"max_bytes\\":$((PREFIX_MATTER_MAX_GB << 30)),\\"max_ns_bytes\\":$((PREFIX_MATTER_NS_MAX_GB << 30)),\\"ttl_days\\":$PREFIX_MATTER_TTL_DAYS,\\"n_read_threads\\":2,\\"n_write_threads\\":1}}]}}}}"
  EXP_EXTRA_ARGS="${{EXP_EXTRA_ARGS:+$EXP_EXTRA_ARGS }}--kv-transfer-config '$PREFIX_TRANSFER_JSON'"
  EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_DSV41_KV_CRC=1 VLLM_DSV41_OFFLOAD_EXPANDABLE_OK=1 VLLM_DSV41_OFFLOAD_MULTINODE=1 VLLM_DSV41_OFFLOAD_MN_VERIFY=$PREFIX_MN_VERIFY VLLM_DSV41_OFFLOAD_MN_STAGE_MB=8 VLLM_DSV41_OFFLOAD_LOAD_FRACTION=$PREFIX_LOAD_FRACTION VLLM_DSV41_OFFLOAD_IO_CPUS=$PREFIX_IO_CPUS VLLM_DSV41_OFFLOAD_STREAM=$PREFIX_STREAM VLLM_DSV41_OFFLOAD_STREAM_MB=$PREFIX_STREAM_MB${{PREFIX_HASH_DUMP:+ VLLM_DSV41_OFFLOAD_HASH_DUMP=$PREFIX_HASH_DUMP VLLM_DSV41_OFFLOAD_HASH_SEGMENT=$PREFIX_HASH_SEGMENT}}"
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
