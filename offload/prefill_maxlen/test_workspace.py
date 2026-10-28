"""Execute the patched prefill function on CPU with native kernel boundaries replaced.

This tests real control flow, buffer bytes, logits and top-k equality; it is not
a test of CUDA kernels, native rounding, GB10 latency or memory allocation policy.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from itertools import accumulate
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from offload.prefill_maxlen.build_overlay import build, manifest, patched_source

SOURCE = Path("/opt/dsv41/data/src-image")
INDEXER_PATH = manifest()["path"]


def definitions(text: str, names: set[str]) -> dict[str, object]:
    """Extract the actual production definitions, without importing CUDA/vLLM."""
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    body: list[ast.stmt] = list(nodes)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + body,
                        type_ignores=[])
    scope: dict[str, object] = {"torch": torch, "MXFP4_BLOCK_SIZE": 32,
                                "prod": prod, "accumulate": accumulate,
                                "round_up": lambda value, alignment: (value + alignment - 1) // alignment * alignment}
    exec(compile(module, "<PF1 extracted production functions>", "exec"), scope)
    return scope


@pytest.fixture(scope="module")
def patched() -> str:
    return patched_source(SOURCE)


def test_reproducible_and_fail_closed_build(tmp_path: Path) -> None:
    output = build(SOURCE, tmp_path / "overlay")
    assert output == build(SOURCE, tmp_path / "overlay")
    output.write_text("foreign overlay")
    with pytest.raises(ValueError, match="replace different"):
        build(SOURCE, tmp_path / "overlay")
    with pytest.raises(ValueError, match="outside"):
        build(SOURCE, SOURCE / "PF1-must-not-write")
    source = tmp_path / "wrong-source" / INDEXER_PATH
    source.parent.mkdir(parents=True)
    source.write_text("wrong engine revision")
    with pytest.raises(ValueError, match="base SHA mismatch"):
        build(tmp_path / "wrong-source", tmp_path / "other-output")


def test_host_bounds(patched: str) -> None:
    scope = definitions(patched, {"_prefill_workspace_rows"})
    rows = cast(Callable[[int, list[int]], int], scope["_prefill_workspace_rows"])
    assert rows(4 * 327680, [32000, 32000, 17000]) == 32000
    assert rows(4 * 524288, [32000, 32000, 17000]) == 32000
    assert rows(4 * 524288, [450000, 500000]) == 500000
    assert rows(4 * 524288, [524288] * 4) == 524288
    assert rows(4 * 524288, [4 * 524288]) == 4 * 524288
    assert rows(32, []) == 0
    for reserved, sizes in ((4, [5]), (4, [-1]), (-1, [])):
        with pytest.raises(ValueError):
            rows(reserved, sizes)


class Workspace:
    """Run the image's real aligned view constructor over a CPU byte buffer."""

    def __init__(self, scope: dict[str, object]) -> None:
        workspace_source = (SOURCE / "vllm/v1/worker/workspace.py").read_text()
        tree = ast.parse(workspace_source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "WorkspaceManager")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_simultaneous")
        method_scope = definitions(ast.unparse(method), {"get_simultaneous"})
        size_scope = definitions(workspace_source, {"_compute_bytes"})
        method_scope["_compute_bytes"] = size_scope["_compute_bytes"]
        self.get_views = cast(Callable[..., list[torch.Tensor]], method_scope["get_simultaneous"])
        self.storage = torch.empty(1, dtype=torch.uint8)
        self.requested: list[int] = []
        self.scope = scope

    def _ensure_workspace_size(self, size: int) -> torch.Tensor:
        self.requested.append(size)
        # Poison every unfilled byte; accidental tail reads must affect parity.
        self.storage = torch.full((size,), 0xD3, dtype=torch.uint8)
        return self.storage

    def get_simultaneous(self, *specs: tuple[tuple[int, ...], torch.dtype]) -> list[torch.Tensor]:
        return self.get_views(self, *specs)


class Metadata(SimpleNamespace):
    pass


class NoRowShard:
    @staticmethod
    def from_metadata(metadata: object, decode_tokens: int) -> None:
        return None


class CpuOps:
    def __init__(self, fp4: bool) -> None:
        self.fp4 = fp4
        self.gathers = 0
        self.seen_values: list[torch.Tensor] = []
        self.seen_scales: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []

    def cp_gather_indexer_k_quant_cache(
        self, cache: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
        table: torch.Tensor, cu_lens: torch.Tensor,
    ) -> None:
        self.gathers += 1
        # Model request-local paged gather: order and duplicates come from table.
        count = int(cu_lens[-1])
        gathered = cache[table[:count].long()]
        values[:count].view(torch.uint8).copy_(gathered[:, :values.shape[1]])
        if self.fp4:
            scales[:count].fill_(127)
        else:
            scales[:count].view(torch.float32).fill_(1.0)

    def mqa(
        self, query: tuple[torch.Tensor, torch.Tensor | None],
        keys: tuple[torch.Tensor, torch.Tensor], weights: torch.Tensor,
        starts: torch.Tensor, ends: torch.Tensor, *, clean_logits: bool,
    ) -> torch.Tensor:
        values, scales = keys
        self.seen_values.append(values.view(torch.uint8).clone())
        self.seen_scales.append(scales.view(torch.uint8).clone())
        # Deterministic CPU surrogate at the native boundary. Source buffers
        # are exact bytes; CUDA quantization/top-k arithmetic remains untested.
        raw = values.view(torch.uint8).float()
        q = query[0].float()
        result = q @ raw.T
        result *= weights[:, :1]
        valid = (torch.arange(raw.shape[0])[None, :] >= starts[:, None]) & (
            torch.arange(raw.shape[0])[None, :] < ends[:, None])
        result.masked_fill_(~valid, float("-inf"))
        self.logits.append(result.clone())
        return result

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
        out: torch.Tensor, rows: int, stride0: int, stride1: int, topk: int,
    ) -> None:
        assert logits.shape[0] == rows
        assert logits.stride() == (stride0, stride1)
        order = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :topk]
        selected = logits.gather(1, order)
        out[:, :order.shape[1]] = torch.where(selected.isfinite(), order, -1).to(torch.int32)


def execute(
    text: str, enabled: bool, reserve: int, fp4: bool,
    use_pcp: bool = False, dcp_world_size: int = 1,
) -> tuple[torch.Tensor, CpuOps, Workspace]:
    names = {"_gather_workspace_shapes", "sparse_attn_indexer"}
    if "def _prefill_workspace_rows" in text:
        names.add("_prefill_workspace_rows")
    scope = definitions(text, names)
    workspace = Workspace(scope)
    ops = CpuOps(fp4)
    # Two requests in one gather, query continuation without another gather,
    # then a shorter request reusing the same workspace. Nonzero token offset
    # preserves existing decode-prefix rows (mixed-batch prefill segment).
    chunks = []
    for start, stop, count, skip in ((2, 4, 19, False), (4, 6, 19, True), (6, 9, 7, False)):
        chunks.append(SimpleNamespace(
            token_start=start, token_end=stop, max_local_total_seq_lens=count,
            local_total_seq_lens=count, local_cu_seq_lens=torch.tensor([0, count]),
            block_table=(torch.arange(count) * 3) % 23, skip_kv_gather=skip,
            cu_seqlen_ks=torch.zeros(stop - start, dtype=torch.int32),
            cu_seqlen_ke=torch.arange(count - (stop - start) + 1, count + 1, dtype=torch.int32),
            pcp_deinterleave_idx=None,
        ))
    metadata = Metadata(
        slot_mapping=torch.arange(9), num_decodes=0, num_prefills=3,
        num_decode_tokens=2, prefill=SimpleNamespace(chunks=chunks),
    )
    scope.update({
        "_DSV41_INDEXER_REQUEST_WORKSPACE": enabled,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={"index": metadata}, cudagraph_runtime_mode=0),
        "current_platform": SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn, is_xpu=lambda: False),
        "_resolve_layer_name": lambda name: name,
        "DeepseekV32IndexerMetadata": Metadata,
        "CUDAGraphMode": SimpleNamespace(FULL=1),
        "current_workspace_manager": lambda: workspace,
        "PrefillRowShard": NoRowShard,
        "ops": ops, "fp8_fp4_mqa_logits": ops.mqa,
        "_merge_dcp_topk_global": lambda *args, **kwargs: None,
    })
    width = 64 if fp4 else 128
    q = torch.arange(9 * width, dtype=torch.float32).reshape(9, width) % 5
    if fp4:
        q = q.to(torch.uint8)
    cache = (torch.arange(23 * width).reshape(23, width) % 113).to(torch.uint8)
    out = torch.full((12, 4), 999, dtype=torch.int32)
    function = cast(Callable[..., torch.Tensor], scope["sparse_attn_indexer"])
    function(torch.empty(9, 1), "index", cache, q,
             torch.ones(9, 4, dtype=torch.uint8) if fp4 else None,
             None, torch.ones(9, 1), 128, "ue8m0", 4, 128,
             reserve // 4, reserve, out, True, use_pcp, False, "", use_fp4_cache=fp4,
             dcp_world_size=dcp_world_size)
    return out, ops, workspace


@pytest.mark.parametrize("fp4", [False, True])
def test_full_prefill_control_flow_and_bytes_equal(patched: str, fp4: bool) -> None:
    base = (SOURCE / INDEXER_PATH).read_text()
    reference, ref_ops, legacy = execute(base, False, 256, fp4)
    default, _, default_workspace = execute(patched, False, 256, fp4)
    low, low_ops, low_workspace = execute(patched, True, 256, fp4)
    high, high_ops, high_workspace = execute(patched, True, 512, fp4)
    assert torch.equal(reference, default)
    assert torch.equal(reference, low) and torch.equal(low, high)
    assert ref_ops.gathers == low_ops.gathers == high_ops.gathers == 2
    assert default_workspace.requested == legacy.requested
    assert low_workspace.requested == high_workspace.requested
    assert low_workspace.requested[0] < legacy.requested[0]
    for attribute in ("seen_values", "seen_scales", "logits"):
        expected = cast(list[torch.Tensor], getattr(ref_ops, attribute))
        for actual_ops in (low_ops, high_ops):
            actual = cast(list[torch.Tensor], getattr(actual_ops, attribute))
            assert len(expected) == len(actual)
            assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    assert (high[:2] == -1).all()  # prefill clearing behavior retained
    assert (high[9:] == 999).all()  # spare output rows untouched


@pytest.mark.parametrize("pcp,dcp", [(True, 1), (False, 2), (True, 2)])
def test_parallel_context_modes_keep_legacy_layout(patched: str, pcp: bool, dcp: int) -> None:
    base = (SOURCE / INDEXER_PATH).read_text()
    reference, _, legacy = execute(base, False, 256, False, pcp, dcp)
    actual, _, current = execute(patched, True, 256, False, pcp, dcp)
    assert torch.equal(reference, actual)
    assert legacy.requested == current.requested


def test_real_chunker_uses_request_lengths_at_both_caps() -> None:
    text = (SOURCE / "vllm/v1/attention/backends/mla/indexer.py").read_text()
    tree = ast.parse(text)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "_split_indexer_prefill_chunks")
    scope = definitions(ast.unparse(method), {"_split_indexer_prefill_chunks"})
    split = cast(Callable[..., list[tuple[slice, slice]]], scope["_split_indexer_prefill_chunks"])
    for length in (4096, 32000, 150000, 327680):
        for seats in (1, 4):
            for ratio in (1, 2):
                seq = torch.full((seats,), length // ratio, dtype=torch.int32)
                query = torch.full((seats,), 4096 // seats, dtype=torch.int32)
                low = split(seq, query, 4 * 327680 // ratio, 128 * 1024**2)
                high = split(seq, query, 4 * 524288 // ratio, 128 * 1024**2)
                assert low == high
    # At 500K, factor 1 can hold every individual request; the real chunker
    # handles multiple requests by grouping/splitting, never truncating keys.
    long = split(torch.tensor([500000] * 4), torch.tensor([1024] * 4), 524288, 128 * 1024**2)
    assert all(0 <= request.start < request.stop <= 4 for request, _ in long)
    assert {i for request, _ in long for i in range(request.start, request.stop)} == set(range(4))


def test_only_live_tp_prefill_view_layout_changes(patched: str) -> None:
    """Profile and decode code must remain literally unchanged, not just similar."""
    base = (SOURCE / INDEXER_PATH).read_text()
    original = definitions(base, {"sparse_attn_indexer"})
    updated = definitions(patched, {"sparse_attn_indexer"})
    # AST diff outside the one prefill span: preserve dummy reservations,
    # decode operations, kernel calls and loop contents exactly.
    before = ast.parse(base)
    after = ast.parse(patched)
    def fn(tree: ast.Module) -> ast.FunctionDef:
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "sparse_attn_indexer")
    old_fn, new_fn = fn(before), fn(after)
    old_prefill = next(node for node in old_fn.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "has_prefill")
    new_prefill = next(node for node in new_fn.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "has_prefill")
    old_fn.body.remove(old_prefill)
    new_fn.body.remove(new_prefill)
    assert ast.dump(old_fn, include_attributes=False) == ast.dump(new_fn, include_attributes=False)
    assert original.keys() == updated.keys()
