"""Build R-SP: prompt-only sequence-parallel prefill on the composed K1 model.py (UPS1 #3, spark-ds41f 0014-0018 idea).

Inputs (sha-pinned in BASES.json, never modified): K1's `nvidia/model.py` (= UP1's copy, the kvpack MANIFEST overlay
bytes). Output: that file with narrow anchored edits (each anchor must match exactly once) plus one new module,
`vllm/models/deepseek_v41/sp_prefill.py`. Runtime switch VLLM_DSV41_PREFILL_SP (default 0): with it off every SP branch
is skipped and the model runs K1's statements (CPU tests run both on a fake TP2 group). `build(out, False)` emits the
K1 bytes unchanged. CPU only: nothing installs or restarts a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
K1 = REPO / 'offload/kvpack/overlay'
MODEL = 'vllm/models/deepseek_v41/nvidia/model.py'
FILES = (MODEL,)
HELPER = 'vllm/models/deepseek_v41/sp_prefill.py'
GENERATED = HERE / 'generated'
CONTAINER_PREFIX = '/usr/local/lib/python3.12/dist-packages/'

# (old, new): every `old` must occur exactly once in the K1 base.
EDITS: tuple[tuple[str, str], ...] = (
    # 1. import the helper
    ('from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask\n',
     'from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask\n'
     'from .. import sp_prefill as _sp  # dsv41 R-SP: prompt-only sequence parallelism\n'),
    # 2. decoder layer: the forward's split rows
    ('        capture_previous_aux: bool = False,\n'
     '        mega_gate_metadata: MegaGateRoutingMetadata | None = None,\n'
     '    ) -> tuple[\n',
     '        capture_previous_aux: bool = False,\n'
     '        mega_gate_metadata: MegaGateRoutingMetadata | None = None,\n'
     '        sp_rows: "_sp.SPRows | None" = None,\n'
     '    ) -> tuple[\n'),
    # 3. Engram: lookup and wkv on every row, the gate on this rank's rows
    ('                residual = self.engram(\n'
     '                    previous_post,\n'
     '                    engram_hashes[:, self.engram.layer_hash_index],\n'
     '                    engram_mask,\n'
     '                )\n',
     '                if sp_rows is not None:\n'
     '                    # dsv41 R-SP: lookup and wkv on every row, the gate on this rank\'s rows\n'
     '                    residual = _sp.engram_local(\n'
     '                        self.engram,\n'
     '                        sp_rows,\n'
     '                        previous_post,\n'
     '                        engram_hashes[:, self.engram.layer_hash_index],\n'
     '                        engram_mask,\n'
     '                    )\n'
     '                else:\n'
     '                    residual = self.engram(\n'
     '                        previous_post,\n'
     '                        engram_hashes[:, self.engram.layer_hash_index],\n'
     '                        engram_mask,\n'
     '                    )\n'),
    # 4. attention: gather to T rows, wo_b unreduced, reduce-scatter
    ('        if self.use_sequence_parallel:\n'
     '            x = sp_all_gather(x)[: positions.shape[0]]\n'
     '\n'
     '        with _memprobe.phase(self.layer_idx, "attn", positions.shape[0]) if _memprobe.ENABLED else _NULLCTX:\n'
     '            x = self.attn(positions, x, None)\n'
     '        # With GEMM-RS bound, the attention output is already this rank\'s\n'
     '        # sequence-parallel shard (see DeepseekV4Attention._wo_b_proj).\n'
     '        if self.use_sequence_parallel and self.attn.gemm_rs is None:\n'
     '            x = sp_reduce_scatter(x)\n',
     '        if self.use_sequence_parallel:\n'
     '            x = sp_all_gather(x)[: positions.shape[0]]\n'
     '        elif sp_rows is not None:\n'
     '            # dsv41 R-SP: attention runs on every row; wo_b\'s partial is reduce-scattered below\n'
     '            x = sp_rows.gather(x)\n'
     '\n'
     '        with _memprobe.phase(self.layer_idx, "attn", positions.shape[0]) if _memprobe.ENABLED else _NULLCTX:\n'
     '            if sp_rows is not None:\n'
     '                with _sp.unreduced(self.attn.wo_b):\n'
     '                    x = self.attn(positions, x, None)\n'
     '            else:\n'
     '                x = self.attn(positions, x, None)\n'
     '        # With GEMM-RS bound, the attention output is already this rank\'s\n'
     '        # sequence-parallel shard (see DeepseekV4Attention._wo_b_proj).\n'
     '        if self.use_sequence_parallel and self.attn.gemm_rs is None:\n'
     '            x = sp_reduce_scatter(x)\n'
     '        elif sp_rows is not None:\n'
     '            x = sp_rows.reduce_scatter(x)\n'),
    # 5. MoE: every row in, this rank's rows of the sum out
    ('                x = self.ffn.forward_unfinalized(x, input_ids)\n'
     '            else:\n'
     '                x = self.ffn(x, input_ids, mega_gate_metadata)\n',
     '                x = self.ffn.forward_unfinalized(x, input_ids)\n'
     '            elif sp_rows is not None:\n'
     '                # dsv41 R-SP: gather, run the MoE on every row, reduce-scatter its sum\n'
     '                x = _sp.moe(self.ffn, sp_rows, x, input_ids)\n'
     '            else:\n'
     '                x = self.ffn(x, input_ids, mega_gate_metadata)\n'),
    # 6. replay cut layer's KV write: its attention input on every row
    ('        residual: torch.Tensor,\n'
     '    ) -> None:\n'
     '        """Write the KV ``forward`` would for the replay layers\' inputs, skipping\n',
     '        residual: torch.Tensor,\n'
     '        sp_rows: "_sp.SPRows | None" = None,\n'
     '    ) -> None:\n'
     '        """Write the KV ``forward`` would for the replay layers\' inputs, skipping\n'),
    ('        self.attn.forward_kv(positions, x)\n',
     '        if sp_rows is not None:\n'
     '            x = sp_rows.gather(x)  # dsv41 R-SP: the KV of every row\n'
     '        self.attn.forward_kv(positions, x)\n'),
    # 7. model: the threshold, once
    ('        else:\n'
     '            self._mtp_hidden_buffer = None\n'
     '\n'
     '    def embed_input_ids(',
     '        else:\n'
     '            self._mtp_hidden_buffer = None\n'
     '\n'
     '        # dsv41 R-SP: smallest prompt forward that runs sequence-parallel (0 = never;\n'
     '        # VLLM_DSV41_PREFILL_SP=1 enables it).\n'
     '        self.prefill_sp_min_rows = _sp.resolve_min_rows(\n'
     '            vllm_config,\n'
     '            static_sequence_parallel=self.use_sequence_parallel,\n'
     '            fuse_mhc_all_reduce=self.fuse_mhc_all_reduce,\n'
     '            native_mega_moe=self.use_native_mega_moe,\n'
     '            gemm_rs=self.run_gemm_rs,\n'
     '        )\n'
     '\n'
     '    def embed_input_ids('),
    # 8. model forward: split this forward's rows
    ('            hidden_states = sp_shard(hidden_states)\n'
     '            input_ids = sp_shard(input_ids)\n'
     '\n',
     '            hidden_states = sp_shard(hidden_states)\n'
     '            input_ids = sp_shard(input_ids)\n'
     '        # dsv41 R-SP: a prompt forward keeps this rank\'s rows between collectives;\n'
     '        # input_ids stay whole for the MoE (it runs on every row).\n'
     '        sp_rows = (\n'
     '            _sp.plan(\n'
     '                full_num_tokens,\n'
     '                self.prefill_sp_min_rows,\n'
     '                get_tensor_model_parallel_world_size(),\n'
     '                get_tensor_model_parallel_rank(),\n'
     '                in_piecewise_cudagraph() or torch.cuda.is_current_stream_capturing(),\n'
     '            )\n'
     '            if self.prefill_sp_min_rows\n'
     '            else None\n'
     '        )\n'
     '        if sp_rows is not None:\n'
     '            hidden_states = sp_rows.local_slice(hidden_states)\n'
     '\n'),
    # 9. model forward: layers, replay (leaves SP on a trimming step), collapse
    ('            aux_hidden_by_layer,\n'
     '            engram_hashes,\n'
     '            engram_mask,\n'
     '        )\n'
     '        late_aux: list[torch.Tensor] = []\n'
     '        if self.decoder_replay_layers is not None:\n',
     '            aux_hidden_by_layer,\n'
     '            engram_hashes,\n'
     '            engram_mask,\n'
     '            sp_rows=sp_rows,\n'
     '        )\n'
     '        late_aux: list[torch.Tensor] = []\n'
     '        if self.decoder_replay_layers is not None and sp_rows is not None:\n'
     '            # dsv41 R-SP: a trimming step leaves SP before the replay layers\n'
     '            assert isinstance(hidden_states, torch.Tensor)\n'
     '            assert residual is not None and post_mix is not None and res_mix is not None\n'
     '            assert pre_mix is not None\n'
     '            sp_rows, (hidden_states, pre_mix, *late_aux) = _sp.replay(\n'
     '                self.decoder_replay_layers,\n'
     '                self._run_replay_layers,\n'
     '                sp_rows,\n'
     '                hidden_states,\n'
     '                positions,\n'
     '                input_ids,\n'
     '                pre_mix,\n'
     '                post_mix,\n'
     '                res_mix,\n'
     '                residual,\n'
     '            )\n'
     '        elif self.decoder_replay_layers is not None:\n'),
    ('                aux_hidden_by_layer,\n'
     '                full_num_tokens,\n'
     '            )\n'
     '\n'
     '        aux_hidden_states = [\n',
     '                aux_hidden_by_layer,\n'
     '                full_num_tokens,\n'
     '                sp_rows=sp_rows,\n'
     '            )\n'
     '\n'
     '        aux_hidden_states = [\n'),
    # 10. model forward: gather the split rows back (MTP buffer, final hidden states)
    ('                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]\n',
     '                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]\n'
     '            elif sp_rows is not None:\n'
     '                hidden_states = sp_rows.gather(hidden_states)\n'
     '                pre_mix = sp_rows.gather(pre_mix)\n'
     '                sp_rows = None\n'),
    ('            # Without MTP, gather only the collapsed and normalized hidden states.\n'
     '            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]\n',
     '            # Without MTP, gather only the collapsed and normalized hidden states.\n'
     '            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]\n'
     '        elif sp_rows is not None:\n'
     '            hidden_states = sp_rows.gather(hidden_states)\n'),
    # 11. _run_layers: pass the rows, gather aux
    ('        engram_hashes: torch.Tensor | None = None,\n'
     '        engram_mask: torch.Tensor | None = None,\n'
     '    ) -> tuple[\n'
     '        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor\n'
     '    ]:\n',
     '        engram_hashes: torch.Tensor | None = None,\n'
     '        engram_mask: torch.Tensor | None = None,\n'
     '        sp_rows: "_sp.SPRows | None" = None,\n'
     '    ) -> tuple[\n'
     '        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor\n'
     '    ]:\n'),
    ('                    mega_gate_metadata=mega_gate_metadata,\n'
     '                )\n'
     '            )\n',
     '                    mega_gate_metadata=mega_gate_metadata,\n'
     '                    sp_rows=sp_rows,\n'
     '                )\n'
     '            )\n'),
    ('                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]\n',
     '                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]\n'
     '                elif sp_rows is not None:\n'
     '                    previous_aux = sp_rows.gather(previous_aux)\n'),
    # 12. _collapse: gather the final aux
    ('        aux_hidden_by_layer: dict[int, torch.Tensor],\n'
     '        full_num_tokens: int,\n'
     '    ) -> torch.Tensor:\n',
     '        aux_hidden_by_layer: dict[int, torch.Tensor],\n'
     '        full_num_tokens: int,\n'
     '        sp_rows: "_sp.SPRows | None" = None,\n'
     '    ) -> torch.Tensor:\n'),
    ('                final_aux = sp_all_gather(final_aux)[:full_num_tokens]\n',
     '                final_aux = sp_all_gather(final_aux)[:full_num_tokens]\n'
     '            elif sp_rows is not None:\n'
     '                final_aux = sp_rows.gather(final_aux)\n'),
    # 13. _run_replay_layers: continue split when the step has no replay batch
    ('        res_mix: torch.Tensor,\n'
     '        residual: torch.Tensor,\n'
     '    ) -> tuple[torch.Tensor, ...]:\n',
     '        res_mix: torch.Tensor,\n'
     '        residual: torch.Tensor,\n'
     '        sp_rows: "_sp.SPRows | None" = None,\n'
     '    ) -> tuple[torch.Tensor, ...]:\n'),
    ('            residual,\n'
     '            aux_hidden_by_layer,\n'
     '        )\n'
     '        hidden_states = self._collapse(\n'
     '            hidden_states,\n'
     '            residual,\n'
     '            post_mix,\n'
     '            res_mix,\n'
     '            aux_hidden_by_layer,\n'
     '            positions.shape[0],\n'
     '        )\n',
     '            residual,\n'
     '            aux_hidden_by_layer,\n'
     '            sp_rows=sp_rows,\n'
     '        )\n'
     '        hidden_states = self._collapse(\n'
     '            hidden_states,\n'
     '            residual,\n'
     '            post_mix,\n'
     '            res_mix,\n'
     '            aux_hidden_by_layer,\n'
     '            positions.shape[0],\n'
     '            sp_rows=sp_rows,\n'
     '        )\n'),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_bases(root: Path = K1) -> dict[str, str]:
    pins: dict[str, str] = json.loads((HERE / 'BASES.json').read_text())
    if set(pins) != set(FILES):
        raise ValueError(f'R-SP BASES.json must pin exactly {FILES}')
    for path, expected in pins.items():
        actual = sha((root / path).read_bytes())
        if actual != expected:
            raise ValueError(f'R-SP base drift: {path}: {actual} != {expected}')
    return pins


def overlay_model(source: str) -> str:
    for old, new in EDITS:
        count = source.count(old)
        if count != 1:
            raise ValueError(f'R-SP anchor found {count} times (need 1): {old[:90]!r}')
        source = source.replace(old, new, 1)
    compile(source, MODEL, 'exec')
    return source


def build(output: Path, enabled: bool = True, root: Path = K1) -> dict[str, str]:
    """Write the overlay tree; returns {path: sha256}. OFF writes the K1 bytes unchanged."""
    verify_bases(root)
    output = output.resolve()
    if output != GENERATED.resolve():  # the window config's default source root (gitignored)
        for forbidden in (K1.resolve(), HERE.resolve()):
            if output == forbidden or forbidden in output.parents:
                raise ValueError(f'R-SP build must not modify {forbidden}')
    if output.exists():
        raise ValueError(f'R-SP build destination must not exist: {output}')
    base = (root / MODEL).read_text()
    files = {MODEL: overlay_model(base) if enabled else base}
    if enabled:
        helper = (HERE / 'overlay' / HELPER).read_text()
        compile(helper, HELPER, 'exec')
        files[HELPER] = helper
    output.mkdir(parents=True)
    digests: dict[str, str] = {}
    for path, body in files.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        digests[path] = sha(target.read_bytes())
    return digests


def overlays_arg(output: Path, digests: dict[str, str]) -> str:
    """The OVERLAYS entries (start_exp.sh `src=dst` pairs) that stage this build into the container."""
    return ' '.join(f'{output / path}={CONTAINER_PREFIX}{path}' for path in sorted(digests))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--off', action='store_true', help='emit the K1 bytes unchanged')
    args = parser.parse_args()
    digests = build(args.output, not args.off)
    print(json.dumps({'qualification': 'CPU only; VLLM_DSV41_PREFILL_SP default 0; not deployed',
                      'files': digests, 'overlays': overlays_arg(args.output, digests)}, indent=2))


if __name__ == '__main__':
    main()
