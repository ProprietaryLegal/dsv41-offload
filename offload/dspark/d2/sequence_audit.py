"""Text-free sequence provenance, token-ID spans and explicit label policy.

This is additive sidecar metadata (own schema), not a tensor-schema revision.
No decoding, model call, source text or target tensor is needed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from .split_audit import field
from .assembly_paths import AssemblyPaths

SCHEMA = 'd2.sequence-audit.v1'
MARKERS = ('think_start', 'think_end', 'assistant', 'user', 'eos', 'dsml')


def object_map(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a metadata object')
    return cast(Mapping[str, object], value)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def hash_field(row: Mapping[str, object], key: str) -> str:
    value = field(row, key)
    if not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{key} must be a SHA256 identity')
    return value


def validate_context(context: Mapping[str, object], prompt_manifest: Mapping[str, object],
                     *, require_chat_match: bool = True, paths: AssemblyPaths | None = None,
                     mode: str = 'prefill_replay') -> None:
    if context.get('schema') != 'd2.sequence-context.v1':
        raise ValueError('expected d2.sequence-context.v1')
    target = object_map(context.get('target'), 'target')
    key = 'decode_capture_target' if mode == 'decode_capture' else 'replay_target'
    if mode == 'decode_capture' and context.get('mode') != 'decode_capture':
        raise ValueError('decode capture requires an explicit decode-capture context')
    replay = object_map(context.get(key), key)
    for identity in (target, replay):
        for key in ('weights_manifest_sha256', 'quantization_manifest_sha256', 'receipt_sha256'):
            hash_field(identity, key)
        for key in ('model_root', 'quantization', 'engine_revision'):
            field(identity, key)
    identity_keys = ('model_root', 'quantization', 'engine_revision',
                     'weights_manifest_sha256', 'quantization_manifest_sha256')
    same = (all(target[k] == replay[k] for k in identity_keys) if mode == 'decode_capture' else
            {k: v for k, v in target.items() if k != 'receipt_sha256'} ==
            {k: v for k, v in replay.items() if k != 'receipt_sha256'})
    if not same:
        raise ValueError('generation and replay target/quantization/engine identity differ')
    inputs = object_map(prompt_manifest.get('inputs'), 'prompt manifest inputs')
    for key in ('encoder_sha256', 'tokenizer_sha256'):
        if hash_field(context, key) != inputs.get(key):
            raise ValueError(f'{key} differs from prompt manifest')
    hash_field(context, 'chat_template_sha256')
    markers = object_map(context.get('markers'), 'markers')
    values: set[int] = set()
    for key in MARKERS:
        value = markers.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 129280 or value in values:
            raise ValueError('markers must be distinct tokenizer-verified token IDs')
        values.add(value)
    # Verify marker IDs from the hashed tokenizer, never trust an arbitrary map.
    from tokenizers import Tokenizer
    for label in ('tokenizer', 'encoder', 'chat_template'):
        original = field(context, label + '_path')
        path = paths.resolve(original) if paths else Path(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != context[label + '_sha256']:
            raise ValueError(f'{label} file hash differs from context')
    tokenizer_path = field(context, 'tokenizer_path')
    tokenizer = Tokenizer.from_file(str(paths.resolve(tokenizer_path)) if paths else tokenizer_path)
    strings = ('<think>', '</think>', '<｜Assistant｜>', '<｜User｜>', '<｜end▁of▁sentence｜>', '｜DSML｜')
    if any(tokenizer.token_to_id(text) != markers[key] for key, text in zip(MARKERS, strings)):
        raise ValueError('marker IDs differ from hashed tokenizer')
    sampling = object_map(context.get('sampling'), 'sampling')
    if sampling.get('temperature') != 1.0 or sampling.get('top_p') != 1.0:
        raise ValueError('D2 sampling must preserve target T=1 / top_p=1')
    for key in ('top_k', 'min_p', 'structured_outputs', 'stop', 'stop_token_ids', 'seed'):
        if key not in sampling:
            raise ValueError(f'sampling must explicitly record {key}, including null/default')
    formatting = object_map(context.get('formatting'), 'formatting')
    if require_chat_match and formatting.get('chat_template_kwargs') != prompt_manifest.get('chat_template_kwargs'):
        raise ValueError('chat template kwargs differ from prompt manifest')
    if 'tools' not in formatting or 'tool_choice' not in formatting:
        raise ValueError('formatting must explicitly record tools and tool_choice')


def token_spans(tokens: Sequence[int], gen_start: int, markers: Mapping[str, int]) -> list[dict[str, object]]:
    """Half-open absolute intervals; token controls are separate from content.

    Carry reasoning state from the raw prompt across every cancellation boundary.
    A response cut before </think> stays reasoning, never an invented answer.
    DSML generation is marked tool_call until EOS; raw IDs remain authoritative.
    """
    if not 1 <= gen_start <= len(tokens):
        raise ValueError('invalid generated span')
    controls = {value: key for key, value in markers.items()}
    phase = 'answer'
    spans: list[dict[str, object]] = []
    for pos, token in enumerate(tokens):
        control = controls.get(token)
        kind = 'format' if control else phase
        if pos >= gen_start:
            if spans and spans[-1]['kind'] == kind:
                spans[-1]['end'] = pos + 1
            else:
                spans.append({'start': pos, 'end': pos + 1, 'kind': kind})
        if control == 'think_start':
            phase = 'reasoning'
        elif control in ('think_end', 'assistant', 'user', 'eos'):
            phase = 'answer'
        elif control == 'dsml':
            phase = 'tool_call'
    return spans


def segment_coverage(segments: Sequence[Mapping[str, object]], length: int) -> dict[str, object]:
    """Generated-token offsets; gaps and explicit unrecorded segments carry no inferred revision."""
    cursor = recorded = 0
    unknown: list[list[int]] = []
    for segment in segments:
        start, n = segment.get('start'), segment.get('n')
        if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(n, int)
                or isinstance(n, bool) or start < cursor or n < 0 or start + n > length):
            raise ValueError('invalid or overlapping generation segment coverage')
        if start > cursor:
            unknown.append([cursor, start])
        if segment.get('unrecorded') is True:
            if n:
                unknown.append([start, start + n])
        else:
            recorded += n
        cursor = start + n
    if cursor < length:
        unknown.append([cursor, length])
    return {'offset_origin': 'gen_start', 'recorded_tokens': recorded,
            'unrecorded_tokens': length - recorded, 'unrecorded_intervals': unknown}


def sequence_audit(tokens: Sequence[int], gen_start: int, context: Mapping[str, object],
                   source: Mapping[str, object], kind: str) -> dict[str, object]:
    markers = cast(Mapping[str, int], object_map(context['markers'], 'markers'))
    spans = token_spans(tokens, gen_start, markers)
    segments = source.get('segments', [])
    if kind == 'gen' and not isinstance(segments, list):
        raise ValueError('generation segments must be a list')
    for segment in cast(list[Mapping[str, object]], segments if isinstance(segments, list) else []):
        recorded = segment.get('sampling')
        if recorded is not None:
            sampling = object_map(recorded, 'segment sampling')
            expected = object_map(context['sampling'], 'context sampling')
            if any(key in expected and expected[key] != value for key, value in sampling.items()):
                raise ValueError('segment sampling differs from context receipt')
    return {'schema': SCHEMA, 'context_sha256': digest(context), 'identity': dict(context),
            'absolute_positions': {'origin': 0, 'length': len(tokens), 'gen_start': gen_start,
                                   'tap_start': max(0, gen_start - 128)},
            'response_spans': spans,
            'label_policy': {'scope': 'response_tokens_only_including_reasoning_and_format',
                'prompt_labels': False, 'response_interval': [gen_start, len(tokens)],
                'eligible_draft_targets': [gen_start + 1, len(tokens)],
                'first_response_token': 'anchor/correction, not a draft-label position',
                'tail': 'P+2+i < L; other labels=-1, position_mask=false'},
            'alignment': {'tap_last': 'P', 'anchor': 'P+1', 'query_positions': 'P+1..P+5',
                          'target': 'P+2+i', 'teacher_forced_previous': 'P+1+i',
                          'teacher_logprob': 'p(tokens[t] | tokens[:t])', 'future_taps_allowed': False},
            'source_state_schema': source.get('schema'), 'source_kind': kind,
            'local_prompt_ids_match': source.get('local_ids_match'),
            'segments': source.get('segments'),
            'segment_coverage': segment_coverage(cast(list[Mapping[str, object]], segments), len(tokens) - gen_start)
                                if kind == 'gen' else None,
            'legacy_segment_policy': 'recorded segments missing priority/revision = historical e2b2186 priority5; '
                                     'gaps and unrecorded segments have unknown provenance; never relabel'}
