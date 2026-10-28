from __future__ import annotations

from copy import deepcopy

import pytest

from d2 import generate as G
from d2 import prompts as P
from d2.reserved_test import FAMILIES, validate_reserved


def fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    old: list[dict[str, object]] = [{'pid': 'old', 'split': 'train', 'family': 'holding_summary',
        'source_docs': ['old_source'], 'prompt_sha256': 'old_hash',
        'messages': [{'content': 'old content'}]}]
    reserved: list[dict[str, object]] = []
    for family in FAMILIES:
        for j in range(3):
            reserved.append({'pid': family + str(j), 'split': 'test', 'family': family,
                'source_docs': [family], 'prompt_sha256': family + str(j),
                'messages': [{'content': 'fresh content'}]})
    return old, reserved


def test_reserved_families_are_separate_and_not_default_generation() -> None:
    old, reserved = fixture()
    validate_reserved(old, reserved)
    assert 'test' not in G.ORDER
    assert not set(FAMILIES) & set(P.LEGAL_FAMILIES + P.GENERAL_FAMILIES)


def test_refuses_source_split_and_family_contamination() -> None:
    old, reserved = fixture()
    bad = deepcopy(reserved)
    bad[0]['source_docs'] = ['old_source']
    with pytest.raises(RuntimeError, match='source unit'):
        validate_reserved(old, bad)
    bad = deepcopy(reserved)
    bad[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='train/dev'):
        validate_reserved(old, bad)
    old[0]['family'] = FAMILIES[0]
    with pytest.raises(ValueError, match='previously used'):
        validate_reserved(old, reserved)
