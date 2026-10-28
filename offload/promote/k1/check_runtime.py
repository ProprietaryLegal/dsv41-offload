#!/usr/bin/env python3
"""Validate the new container's engine log and /v1/models response (CPU only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from profiles import PROFILES, Profile, get_profile


def check_log(text: str, profile: Profile) -> None:
    # The allocator prints model_config.max_model_len as "tokens per request";
    # see image-v5 vllm/v1/core/kv_cache_utils.py:2486 and K1's overlay equivalent.
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens, Maximum concurrency for "
                      r"([\d,]+) tokens per request:", clean)
    values = [(int(k.replace(",", "")), int(m.replace(",", ""))) for k, m in rows]
    expected = (profile.kv_tokens, profile.max_model_len)
    if not values or any(row != expected for row in values):
        raise ValueError(f"engine KV/max_model_len log mismatch: {values!r}; expected {expected}")


def check_models(text: str, served: str, profile: Profile) -> None:
    payload: object = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("/v1/models has no data list")
    entries = payload["data"]
    if len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("/v1/models must expose exactly the production alias")
    entry = entries[0]
    if entry.get("id") != served or type(entry.get("max_model_len")) is not int or entry["max_model_len"] != profile.max_model_len:
        raise ValueError("/v1/models alias/max_model_len differs from the K1 production contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--served", required=True)
    args = parser.parse_args()
    try:
        profile = get_profile(args.profile)
        check_log(args.log.read_text(), profile)
        check_models(args.models.read_text(), args.served, profile)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"K1 runtime REFUSED: {exc}\n")
    print(f"K1 profile {profile.name}: KV {profile.kv_tokens:,}, "
          f"max_model_len {profile.max_model_len:,}; /v1/models agrees")


if __name__ == "__main__":
    main()
