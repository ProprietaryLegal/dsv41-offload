from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models

from d2.sequence_audit import MARKERS, digest, segment_coverage, sequence_audit, token_spans, validate_context


def test_token_id_spans_preserve_partial_reasoning_and_tool_format() -> None:
    m = dict(zip(MARKERS, range(10, 16)))
    # Prompt already opened reasoning; cancellation may happen before the close.
    assert token_spans([12, 10, 1, 2], 2, m) == [{'start': 2, 'end': 4, 'kind': 'reasoning'}]
    assert token_spans([12, 10, 1, 11, 2, 15, 3, 14], 2, m) == [
        {'start': 2, 'end': 3, 'kind': 'reasoning'}, {'start': 3, 'end': 4, 'kind': 'format'},
        {'start': 4, 'end': 5, 'kind': 'answer'}, {'start': 5, 'end': 6, 'kind': 'format'},
        {'start': 6, 'end': 7, 'kind': 'tool_call'}, {'start': 7, 'end': 8, 'kind': 'format'}]


def context_fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    import hashlib
    strings = ('<think>', '</think>', '<｜Assistant｜>', '<｜User｜>', '<｜end▁of▁sentence｜>', '｜DSML｜')
    tok = Tokenizer(models.WordLevel({s: i for i, s in enumerate(strings)}))
    tp = root / 'tokenizer.json'
    tok.save(str(tp))
    ep = root / 'encoder.py'
    ep.write_text('# synthetic encoder identity\n')
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    target = {'weights_manifest_sha256': 'a' * 64, 'quantization_manifest_sha256': 'b' * 64,
              'receipt_sha256': 'c' * 64, 'quantization': 'synthetic', 'engine_revision': 'synthetic',
              'model_root': 'synthetic'}
    ctk = {'thinking': True, 'reasoning_effort': 'high'}
    c: dict[str, object] = {'schema': 'd2.sequence-context.v1', 'target': target, 'replay_target': dict(target),
        'tokenizer_path': str(tp), 'tokenizer_sha256': sha(tp), 'encoder_path': str(ep), 'encoder_sha256': sha(ep),
        'chat_template_path': str(ep), 'chat_template_sha256': sha(ep),
        'markers': dict(zip(MARKERS, range(6))),
        'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': None, 'min_p': None,
                     'structured_outputs': None, 'stop': None, 'stop_token_ids': None, 'seed': None},
        'formatting': {'chat_template_kwargs': ctk, 'tools': None, 'tool_choice': None}}
    pm: dict[str, object] = {'inputs': {'tokenizer_sha256': sha(tp), 'encoder_sha256': sha(ep)},
                             'chat_template_kwargs': ctk}
    return c, pm


def test_context_binds_encoder_target_quantization_and_actual_marker_ids(tmp_path: Path) -> None:
    c, pm = context_fixture(tmp_path)
    validate_context(c, pm)
    for key, value in [('tokenizer_sha256', 'f' * 64), ('encoder_sha256', 'f' * 64),
                       ('replay_target', {}), ('markers', dict(zip(MARKERS, range(20, 26))))]:
        bad = deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_context(bad, pm)
    a = sequence_audit([2, 0, 7, 1, 8, 4], 2, c, {'schema': 'synthetic', 'segments': []}, 'gen')
    assert a['context_sha256'] == digest(c)
    assert a['label_policy'] == {'scope': 'response_tokens_only_including_reasoning_and_format',
        'prompt_labels': False, 'response_interval': [2, 6], 'eligible_draft_targets': [3, 6],
        'first_response_token': 'anchor/correction, not a draft-label position',
        'tail': 'P+2+i < L; other labels=-1, position_mask=false'}


def test_segment_coverage_distinguishes_legacy_from_unrecorded(tmp_path: Path) -> None:
    c, _ = context_fixture(tmp_path)
    segments: list[dict[str, object]] = [
        {'start': 0, 'n': 1},  # Actual legacy segment; preserve its absent revision.
        {'start': 1, 'n': 2, 'unrecorded': True},
        {'start': 3, 'n': 1, 'priority': 20, 'generator_revision': 'd2-priority20-v1'}]
    a = sequence_audit(list(range(6)), 2, c, {'segments': segments}, 'gen')
    assert a['segments'] == segments
    assert a['segment_coverage'] == {'offset_origin': 'gen_start', 'recorded_tokens': 2,
                                    'unrecorded_tokens': 2, 'unrecorded_intervals': [[1, 3]]}
    # Pre-hardening files with a gap must also report unknown coverage without a state load.
    assert segment_coverage([segments[0], segments[2]], 5) == {
        'offset_origin': 'gen_start', 'recorded_tokens': 2,
        'unrecorded_tokens': 3, 'unrecorded_intervals': [[1, 3], [4, 5]]}
    with pytest.raises(ValueError, match='coverage'):
        segment_coverage([{'start': 0, 'n': 3}, {'start': 2, 'n': 2}], 4)
