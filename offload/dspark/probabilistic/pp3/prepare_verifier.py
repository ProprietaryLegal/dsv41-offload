"""Prepare pinned image overlays offline. Writes only to an explicit empty output
directory; never deploys or edits the image snapshot. Run before any coordinator
window. The generated standard verifier accepts actual FP32 q, not log(q)/T.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

UTILS = 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py'
RUNNER = 'vllm/v1/worker/gpu/model_runner.py'
PINS = {UTILS: 'ef1d40bf31140eb644a8f3b7a003373209d6a8e297f059e70ea66cf4972d8d90',
        RUNNER: 'af9be025ff8a17a5aabd452fc3e4b987abe64fb28d852c569e1372d6f1e4d3a6'}
ABI = 'pp3-dyadic24-standard-v1'


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'probability overlay source mismatch: {old[:80]!r}')
    return text.replace(old, new)


def utilities(text: str) -> str:
    text = 'import os\nREMOTE_DRAFT_PROBS_ABI = ' + repr(ABI) + '\n' + text
    old = '    if HAS_DRAFT_LOGITS:\n        # draft_logits is stored pre-temperature, so apply scale first.'
    new = '''    if HAS_DRAFT_LOGITS == 2:
        q = tl.load(draft_logits_ptr + req_state_idx * draft_logits_stride_0
                    + draft_step * draft_logits_stride_1 + token,
                    mask=mask, other=0.).to(tl.float32)
        draft_log_prob = tl.log(q)
        draft_lse = 0.
    elif HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.'''
    text = replace_once(text, old, new)
    text = replace_once(text, '        if HAS_DRAFT_LOGITS:\n            # Get local draft max',
                        '        if HAS_DRAFT_LOGITS == 1:\n            # Get local draft max')
    text = replace_once(text, '        draft_log_probs = draft_logits - draft_lse', '''        draft_log_probs = draft_logits - draft_lse
        if HAS_DRAFT_LOGITS == 2:
            q = tl.load(draft_logits_ptr + req_state_idx * draft_logits_stride_0
                        + resample_idx * draft_logits_stride_1 + block,
                        mask=mask, other=0.).to(tl.float32)
            draft_log_probs = tl.log(q)''')
    text = replace_once(text, '    # Compute the per-vocab-block logits stats,', '''    if os.environ.get("VLLM_DSPARK_SAMPLED_TOP32", "0") == "1":
        if use_block_verification or synthetic_conditional_rates is not None or watermark:
            raise ValueError("PP3 q ABI supports standard verification only, without watermarking")
        if draft_logits is None or draft_logits.dtype != torch.float32 or draft_logits.shape[-1] != target_logits.shape[-1]:
            raise ValueError("PP3 standard verification needs full-vocab FP32 probabilities")
        has_draft_logits = 2

    # Compute the per-vocab-block logits stats,''')
    return text


def runner(text: str) -> str:
    text = 'REMOTE_DRAFT_PROBS_ABI = ' + repr(ABI) + '\n' + text
    return replace_once(text, '        sampler_output: SamplerOutput | None\n', '''        # PP3 failed/missing-q rounds are target-only. Mask verification, never
        # reinterpret sampled IDs as point-mass proposals. Tensor values stay on GPU.
        pp3_valid = getattr(self.speculator, "sampled_valid", None)
        if pp3_valid is not None and input_batch.num_draft_tokens > 0:
            bad = ((input_batch.expanded_local_pos > 0)
                   & ~pp3_valid[input_batch.expanded_idx_mapping.long()])
            invalid_drafts = bad if invalid_drafts is None else invalid_drafts | bad

        sampler_output: SamplerOutput | None
''')


def prepare(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError('output must not exist; refuse overwriting any prepared overlay')
    texts = {}
    for name, pin in PINS.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise ValueError(f'image snapshot differs from pinned PP3 source: {name}')
        texts[name] = utilities(data.decode()) if name == UTILS else runner(data.decode())
        ast.parse(texts[name])
        compile(texts[name], name, 'exec')
    result: dict[str, object] = {'abi': ABI, 'source_sha256': PINS,
                               'output_sha256': {name: hashlib.sha256(text.encode()).hexdigest()
                                                 for name, text in texts.items()}}
    output.mkdir(parents=True)
    for name, text in texts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (output / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), indent=2))


if __name__ == '__main__':
    main()
