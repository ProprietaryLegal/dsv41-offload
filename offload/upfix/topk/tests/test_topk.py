from __future__ import annotations

import dataclasses
import difflib
import json
import re
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.upfix.topk import bitcompare as bc
from offload.upfix.topk import build_overlay as bo
from offload.upfix.topk import jit_build as jb

CSRC = bo.HERE / 'csrc'
# vLLM 6e517b15 csrc/libtorch_stable/torch_bindings.cpp: the image's _C::persistent_topk schema.
IMAGE_SCHEMA = ('persistent_topk(Tensor logits, Tensor lengths, Tensor! output, '
                'Tensor workspace, int k, int max_seq_len) -> ()')


def defined_schema(source: str) -> str:
    body = re.search(r'TORCH_LIBRARY\(dsv41_topk, m\) \{\s*m\.def\(((?:\s*"[^"]*")+)\);', source)
    assert body, 'no TORCH_LIBRARY(dsv41_topk) definition'
    return ''.join(re.findall(r'"([^"]*)"', body.group(1)))


def test_committed_overlay_and_manifest_are_a_fresh_build() -> None:
    bo.check()


def test_vendored_kernels_are_upstream_bytes_and_self_contained() -> None:
    for name, blob in bo.VENDORED.items():
        assert bo.git_blob((CSRC / name).read_bytes()) == blob
    local = {path.name for path in CSRC.iterdir()}
    for path in CSRC.iterdir():
        for include in re.findall(r'#include "([^"]+)"', path.read_text()):
            assert include in local, (path.name, include)
    assert sorted(local) == sorted({*bo.VENDORED, 'dsv41_topk.cu'})
    assert sorted(jb.HEADERS) == sorted(bo.VENDORED) and jb.SOURCES == ('dsv41_topk.cu',)


def test_op_schema_matches_the_image_op_and_the_call_site() -> None:
    source = (CSRC / 'dsv41_topk.cu').read_text()
    manifest = json.loads(bo.MANIFEST.read_text())
    assert defined_schema(source) == IMAGE_SCHEMA == manifest['schema']
    assert 'TORCH_LIBRARY_IMPL(dsv41_topk, CUDA, m)' in source
    assert 'findSchemaOrThrow("_C::top_k_per_row_decode", "")' in source
    call = (bo.IMAGE_SRC / bo.REL).read_text().split(bo.CALL_ANCHOR, 1)[1].split(')', 1)[0]
    assert [a.strip() for a in call.split(',') if a.strip()] == [
        'logits', 'seq_lens', 'topk_indices', 'topk_workspace', 'topk_tokens', 'max_seq_len']


def test_overlay_is_the_image_file_plus_two_edits() -> None:
    base = (bo.IMAGE_SRC / bo.REL).read_text()
    overlay = bo.OVERLAY.read_text()
    removed = [l[1:] for l in difflib.unified_diff(base.splitlines(), overlay.splitlines(), lineterm='', n=0)
               if l.startswith('-') and not l.startswith('---')]
    assert removed == [bo.CALL_ANCHOR.rstrip('\n')]
    assert overlay.count('torch.ops._C.persistent_topk') == 1
    assert overlay.count('torch.ops.dsv41_topk.persistent_topk  # dsv41 R-UPF') == 1
    compile(overlay, str(bo.OVERLAY), 'exec')


class FakeOps:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_library(self, path: str) -> None:
        self.loaded.append(path)


class FakeTorch:
    def __init__(self) -> None:
        self.ops = FakeOps()


def run_switch(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> tuple[dict[str, object], FakeTorch]:
    for key in ('VLLM_DSV41_TOPK_FIX', 'VLLM_DSV41_TOPK_FIX_SO'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    fake = FakeTorch()
    scope: dict[str, object] = {'torch': fake}
    exec(compile(bo.SWITCH, 'switch', 'exec'), scope)
    return scope, fake


def test_switch_default_off_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in ({}, {'VLLM_DSV41_TOPK_FIX': '0'}, {'VLLM_DSV41_TOPK_FIX': 'true'}):
        scope, fake = run_switch(monkeypatch, env)
        assert scope['_DSV41_TOPK_FIX'] is False and fake.ops.loaded == []


def test_switch_on_requires_and_loads_the_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='VLLM_DSV41_TOPK_FIX_SO'):
        run_switch(monkeypatch, {'VLLM_DSV41_TOPK_FIX': '1'})
    with pytest.raises(RuntimeError, match='VLLM_DSV41_TOPK_FIX_SO'):
        run_switch(monkeypatch, {'VLLM_DSV41_TOPK_FIX': '1', 'VLLM_DSV41_TOPK_FIX_SO': str(tmp_path / 'absent.so')})
    library = tmp_path / 'dsv41_topk.so'
    library.write_bytes(b'not loaded by the fake')
    scope, fake = run_switch(monkeypatch, {'VLLM_DSV41_TOPK_FIX': '1', 'VLLM_DSV41_TOPK_FIX_SO': str(library)})
    assert scope['_DSV41_TOPK_FIX'] is True and fake.ops.loaded == [str(library)]


def test_anchor_drift_is_refused() -> None:
    base = (bo.IMAGE_SRC / bo.REL).read_text()
    with pytest.raises(ValueError, match='exactly once'):
        bo.build(base.replace(bo.CALL_ANCHOR, ''))
    with pytest.raises(ValueError, match='exactly once'):
        bo.build(base + bo.SWITCH_ANCHOR)


def test_distinct_breaks_every_tie() -> None:
    values = bc._distinct(torch.tensor([1.0, 1.0, 1.0, 1.0000001, 2.0, 2.0], dtype=torch.float32))
    assert bool((values[1:] > values[:-1]).all())


def path_of(length: int, k: int) -> str:
    if length <= k:
        return 'trivial'
    return 'short' if length <= bc.SHORT_THRESHOLD else 'medium' if length <= bc.RADIX_THRESHOLD else 'radix'


def generated(case: bc.Case, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    lengths = bc.lengths_for(case, gen)
    return lengths, bc.logits_for(case, lengths, gen)


def test_every_planned_case_has_its_shape_paths_values_and_decoys() -> None:
    plan = bc.cases()
    assert len({c.name for c in plan}) == len(plan) == 149
    for i, case in enumerate(plan):
        lengths, logits = generated(case, 41 + i)
        width = case.width - case.view_offset
        assert logits.shape == (case.rows, width) and logits.stride(0) == case.width, case.name
        assert bool((lengths >= 1).all()) and bool((lengths <= width).all()), case.name
        paths = {path_of(int(n), case.k) for n in lengths}
        if case.lengths in ('short', 'short-hi', 'medium', 'radix', 'trivial'):
            assert paths == {case.lengths.removesuffix('-hi')}, case.name
        if case.lengths == 'short-hi':
            assert bool((lengths > 4096).all()), case.name
        if case.lengths not in ('full', 'trivial'):
            assert bool((lengths % 4 != 0).all()), case.name
        if case.view_offset == 0 and case.width != 4099:
            assert case.width % 4 == 0, f'{case.name}: production-like rows use an aligned stride (VEC_SIZE 4)'
        if case.scoring == 'exact':
            assert bc.tie_free(logits, lengths), case.name
        for r in range(case.rows):
            n = int(lengths[r])
            live = logits[r, :n]
            assert bool((logits[r, n:] == bc.DECOY).all()), case.name
            if case.values in ('masked', 'neginf'):
                assert int((live > bc.REQUIRED_ABOVE).sum()) == case.finite, case.name
                assert int((live == float(case.fill)).sum()) == n - case.finite, case.name
            elif case.values.startswith('nan'):
                assert bool(torch.isnan(live).any()) and not bool(torch.isnan(logits[r, n:]).any()), case.name
            elif case.values == 'ties':
                kth = torch.topk(live, case.k).values[-1]
                assert int((live == kth).sum()) == 40 and int((torch.topk(live, case.k).values == kth).sum()) == 20
            else:
                assert bool((live < bc.DECOY).all()) and bool(torch.isfinite(live).all()), case.name


def test_paths_and_fixes_are_all_covered() -> None:
    plan = bc.cases()
    covered: dict[tuple[str, str], set[str]] = {}
    for i, case in enumerate(plan):
        lengths, _ = generated(case, 41 + i)
        for n in lengths:
            covered.setdefault((case.values, case.scoring), set()).add(path_of(int(n), case.k))
    for values in ('normal', 'packed', 'ties', 'nan', 'nan-many'):
        assert {'short', 'medium', 'radix'} <= set().union(*(p for (v, _), p in covered.items() if v == values)), values
    assert {'short', 'medium'} <= covered[('masked', 'valid')] and {'medium', 'radix'} <= covered[('neginf', 'valid')]
    prod = [c for c in plan if c.name.startswith('prod-')]
    assert {c.width for c in prod} == {163_840, 327_680} and {c.rows for c in prod} == {1, 4, 8, 24}
    assert {c.k for c in prod} == {512} and all(c.view_offset == 0 for c in prod)
    for width in (163_840, 327_680):
        paths = set()
        for i, case in enumerate(plan):
            if case.name.startswith(f'prod-w{width}-'):
                paths |= {path_of(int(n), 512) for n in generated(case, 41 + i)[0]}
        assert paths == {'short', 'medium', 'radix'}, width
    assert {c.rows for c in plan if c.width == 524_288} == {1, 2, 3, 4}
    masked = [c for c in plan if c.values == 'masked']
    assert {(c.length - c.k) for c in masked if c.length < 8192} == {1, 2, 3} and any(c.length == 20_001 for c in masked)
    assert {c.finite for c in masked} == {0, 100} and {c.fill for c in masked} == {'-inf', '-1e30'}
    assert all(c.width % 4 == 0 and c.width > c.length for c in masked)


def test_nan_cases_are_fixed_only_and_last() -> None:
    plan = bc.cases()
    flags = [c.fixed_only for c in plan]
    assert flags == sorted(flags) and sum(flags) == 12
    assert all(c.values.startswith('nan') for c in plan if c.fixed_only)
    assert not any(c.values.startswith('nan') for c in plan if not c.fixed_only)


def correct_kernel(logits: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor,
                   k: int, max_seq_len: int) -> None:
    """CPU stand-in with the fixed op's contract: the top min(k, L) of [0, L), -1 elsewhere."""
    assert workspace.numel() == bc.WORKSPACE_BYTES and max_seq_len == int(lengths.max())
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        output[r, :min(k, n)] = torch.topk(logits[r, :n], min(k, n)).indices.to(torch.int32)


def tail_lane_kernel(logits: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor,
                     k: int, max_seq_len: int) -> None:
    """The #62467 bug: when fewer than k values beat the -inf bin, the float4 tail lanes past an L that is not a
    multiple of 4 join the candidates (the padding there is finite, so it is selected)."""
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        end = n
        if int((logits[r, :n] > bc.REQUIRED_ABOVE).sum()) < k and n % 4:
            end = min(n + 4 - n % 4, logits.shape[1])
        output[r, :min(k, n)] = torch.topk(logits[r, :end], min(k, n)).indices.to(torch.int32)


def small(case: bc.Case) -> bc.Case:
    return dataclasses.replace(case, rows=min(case.rows, 4), repeats=1)


def test_correct_kernel_passes_one_case_of_every_kind() -> None:
    seen: set[tuple[str, str, int]] = set()
    results = []
    for i, case in enumerate(bc.cases()):
        kind = (case.values, case.lengths, case.view_offset)
        if kind in seen:
            continue
        seen.add(kind)
        kernels: dict[str, bc.Kernel] = (
            {'fixed': correct_kernel} if case.fixed_only else {'image': correct_kernel, 'fixed': correct_kernel})
        results.append(bc.run_case(small(case), kernels, 'cpu', 41 + i))
    assert len(results) >= 15
    assert bc.verdict(results) == 'PASS' and bc.bad_rows(results, 'image') == 0
    assert all(cast(dict[str, int], r['fixed'])['decoy_hits'] == 0 for r in results)


def test_masked_row_tail_cases_catch_the_tail_lane_bug() -> None:
    plan = bc.cases()
    caught = 0
    for i, case in enumerate(plan):
        if case.values != 'masked':
            continue
        result = bc.run_case(case, {'image': tail_lane_kernel, 'fixed': correct_kernel}, 'cpu', 41 + i)
        image = cast(dict[str, int], result['image'])
        assert bc.verdict([result]) == 'PASS', case.name
        if case.length % 4:
            assert image['bad_rows'] == image['decoy_hits'] == case.rows, case.name
            caught += 1
    assert caught == 48


def test_tie_free_exact_cases_do_not_flag_the_tail_lane_model() -> None:
    """The #62467 model only differs when fewer than k values are finite; exact rows always have >= k."""
    case = small(next(c for c in bc.cases() if c.name == 'prod-w163840-r4'))
    result = bc.run_case(case, {'image': tail_lane_kernel, 'fixed': correct_kernel}, 'cpu', 7)
    assert bc.bad_rows([result], 'image') == 0


def test_validity_scorer() -> None:
    live = torch.tensor([3.0, 1.0, 3.0, float('nan'), float('-inf'), 2.0], dtype=torch.float32)
    assert bc.row_valid(live, [3, 0, 2], 3, None) and bc.row_valid(live, [3, 2, 0], 3, None)
    assert not bc.row_valid(live, [3, 0, 0], 3, None)          # duplicate
    assert not bc.row_valid(live, [3, 0, 6], 3, None)          # out of range
    assert not bc.row_valid(live, [3, 0, 5], 3, None)          # wrong multiset (2.0 for a tied 3.0)
    assert not bc.row_valid(live, [0, 2, 5], 3, None)          # NaN must rank first
    assert not bc.row_valid(live, [3, 0], 3, None)             # too few
    masked = torch.tensor([float('-inf'), 0.5, float('-inf'), float('-inf')], dtype=torch.float32)
    assert bc.row_valid(masked, [1, 0], 2, bc.REQUIRED_ABOVE) and not bc.row_valid(masked, [0, 2], 2, bc.REQUIRED_ABOVE)
    assert bc.same_multiset(torch.tensor([float('nan'), 1.0]), torch.tensor([1.0, float('nan')]))


def test_reference_is_the_exact_top_k_set() -> None:
    logits = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0, bc.DECOY]], dtype=torch.float32)
    lengths = torch.tensor([5], dtype=torch.int32)
    assert bc.reference(logits, lengths, 3) == [frozenset({0, 2, 4})]
    assert bc.reference(logits, lengths, 512) == [frozenset({0, 1, 2, 3, 4})]


def test_jit_build_plan_is_sm121_release_flags_without_fast_math() -> None:
    plan = jb.plan(CSRC, Path('/nonexistent/build'))
    assert plan['arch'] == '12.1' and plan['name'] == 'dsv41_topk'
    flags = [*jb.CUDA_FLAGS, *jb.CXX_FLAGS]
    assert '-O3' in flags and '-DNDEBUG' in flags and not any('fast_math' in f or 'fast-math' in f for f in flags)
    sources = plan['source_sha256']
    assert isinstance(sources, dict) and set(sources) == {*jb.SOURCES, *jb.HEADERS}


def test_bitcompare_without_execute_prints_the_plan(capsys: pytest.CaptureFixture[str],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.argv', ['bitcompare'])
    bc.main()
    printed = json.loads(capsys.readouterr().out)
    assert [row['name'] for row in printed] == [c.name for c in bc.cases()]


def plan_of_three() -> list[bc.Case]:
    names = ('prod-w163840-r4', 'masked-k512-L513-f0-ninf', 'nan-short-k512')
    return [small(next(c for c in bc.cases() if c.name == n)) for n in names]


def test_receipt_is_written_after_every_case_and_completes(tmp_path: Path) -> None:
    out = tmp_path / 'bitcompare.json'
    seen: list[int] = []

    def watching(logits: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor,
                 k: int, max_seq_len: int) -> None:
        seen.append(json.loads(out.read_text())['cases_done'])
        correct_kernel(logits, lengths, output, workspace, k, max_seq_len)

    receipt = bc.run_all(plan_of_three(), {'image': correct_kernel, 'fixed': watching}, 'cpu', 41, out,
                         {'library_sha256': 'x'})
    assert seen == [0, 1, 2] and not (tmp_path / 'bitcompare.json.partial').exists()
    assert json.loads(out.read_text()) == json.loads(json.dumps(receipt))
    assert receipt['complete'] is True and receipt['verdict'] == 'PASS' and receipt['cases_done'] == 3


def test_a_crash_in_the_last_case_keeps_earlier_results_and_fails_safe(tmp_path: Path) -> None:
    out = tmp_path / 'bitcompare.json'

    def faulting(logits: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor,
                 k: int, max_seq_len: int) -> None:
        if bool(torch.isnan(logits).any()):
            raise RuntimeError('CUDA error: an illegal memory access was encountered')
        correct_kernel(logits, lengths, output, workspace, k, max_seq_len)

    with pytest.raises(RuntimeError, match='illegal memory access'):
        bc.run_all(plan_of_three(), {'image': correct_kernel, 'fixed': faulting}, 'cpu', 41, out, {})
    partial = json.loads(out.read_text())
    assert partial['complete'] is False and partial['verdict'] == 'INCOMPLETE'
    assert partial['cases_done'] == len(partial['results']) == 2 and partial['cases'] == 3
    assert partial['error'].startswith('nan-short-k512: RuntimeError')
