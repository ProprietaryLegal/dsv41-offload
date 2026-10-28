"""Original request identity must survive omitted historical sampling defaults."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from d2.decode_dataset import digest
from d2.decode_smoke import checked_spec
from d2.decode_smoke_dry import DrySmoke
from d2.smoke_prepare import historical_request

REPO = Path(__file__).resolve().parents[4]


def test_w5_ast_recovers_body_without_importing_driver(tmp_path):
    source = tmp_path/'w5.py'
    source.write_text("PROMPTS = ['synthetic']\n"
        "raise RuntimeError('must never execute')\n"
        "def run():\n"
        " for p in PROMPTS:\n"
        "  body = {'model':'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2',"
        "'messages':[{'role':'user','content':p}], 'temperature':0, 'max_tokens':300,"
        "'chat_template_kwargs':{'thinking':False}}\n")
    capture = tmp_path/'capture.jsonl'
    capture.write_text(json.dumps({'i':0,'text':'synthetic'})+'\n')
    body = historical_request(source, capture, 0)
    assert 'top_p' not in body and 'priority' not in body
    assert body['messages'] == [{'role':'user','content':'synthetic'}]
    capture.write_text(json.dumps({'i':1,'text':'synthetic'})+'\n')
    with pytest.raises(ValueError, match='ordering'):
        historical_request(source, capture, 0)


def test_omitted_top_p_allowed_only_on_exact_historical_control(tmp_path):
    io = DrySmoke(REPO, tmp_path)
    inputs = io.inputs()
    inputs['requests'][2]['body'].pop('top_p')
    path = Path(inputs['greedy_reference']['request_receipt'])
    proof = json.loads(path.read_text())
    proof['request'].pop('top_p')
    path.write_text(json.dumps(proof))
    inputs['greedy_reference']['request_receipt_sha256'] = digest(path)
    checked_spec(inputs)
    inputs['requests'][2]['body']['top_p'] = 1
    with pytest.raises(ValueError, match='identity'):
        checked_spec(inputs)
    inputs['requests'][2]['body'].pop('top_p')
    inputs['requests'][0]['body'].pop('top_p')
    with pytest.raises(ValueError, match='sampling'):
        checked_spec(inputs)


def test_original_control_without_request_id_is_untagged(tmp_path):
    inputs = DrySmoke(REPO, tmp_path).inputs()
    inputs['requests'][2]['body'].pop('request_id')
    checked_spec(inputs)
    inputs['requests'][2]['body']['request_id'] = 'd2c.forbidden.0.0.8'
    with pytest.raises(ValueError, match='untagged'):
        checked_spec(inputs)
