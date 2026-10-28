"""Explicit reviewed K1 capacity contracts; selection never has a default."""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, NamedTuple


class Profile(NamedTuple):
    name: str
    max_model_len: int
    kv_tokens: int
    requires_r5: bool


PROFILES: Mapping[str, Profile] = MappingProxyType({
    "327680": Profile("327680", 327680, 2502600, False),
    "524288": Profile("524288", 524288, 2820756, True),
})


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"K1 requires an explicit profile: {tuple(PROFILES)}, got {name!r}")
    return PROFILES[name]
