"""Synthetic runner gates; no test may start an SSH/HTTP process."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from d2 import decode_apply as A, decode_smoke as M
from d2.decode_smoke_dry import DrySmoke

REPO = Path(__file__).resolve().parents[4]


def dry(tmp_path):
    out = tmp_path/'out'
    out.mkdir()
    io = DrySmoke(REPO, tmp_path)
    return io, io.inputs(), out


def test_checked_three_prompt_smoke_cannot_contact_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run attempted an external process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    io, inputs, out = dry(tmp_path)
    report = M.run(io, inputs, REPO, out, 'synthetic')
    assert report['synthetic'] is True and report['pass'] is True
    assert report['option_ii'] is True and report['ledger_complete'] is True
    assert [c for c in io.calls if c.startswith('prompt-')] == ['prompt-0','prompt-1']


@pytest.mark.parametrize('failure', ['busy','image','hash','target','reference','prompt_ids','zero','missing_row','extra_seq'])
def test_smoke_refuses_realistic_faults(tmp_path, failure):
    io, inputs, out = dry(tmp_path)
    post, call, metrics, freeze = io.post, io.call, io.metrics, io.freeze
    if failure == 'busy':
        io.metrics = lambda: 'vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n'
    elif failure in ('image','hash','target'):
        def corrupt_call(name, *args, **kwargs):
            result = call(name, *args, **kwargs)
            if name.startswith('identity-'):
                rows = [json.loads(line) for line in result.splitlines()]
                if failure == 'image': rows[0]['image'] = 'wrong'
                elif failure == 'target': rows[1]['files'] = {}
                else: rows[1]['mounted_code_sha256'] = {}
                return '\n'.join(json.dumps(r) for r in rows)
            return result
        io.call = corrupt_call
    elif failure in ('reference','prompt_ids'):
        def corrupt_post(*args, **kwargs):
            result = post(*args, **kwargs)
            if failure == 'reference': result['choices'][0]['text'] = 'different'
            else: result['choices'][0]['prompt_token_ids'] = [999]
            return result
        io.post = corrupt_post
    else:
        def bad_freeze():
            if failure == 'extra_seq': io.captured['untagged'] = [1,2,3]
            freeze()
            root = tmp_path/'remote'
            if failure == 'zero':
                p = root/'smoke0/ledger.jsonl'
                row = json.loads(p.read_text()); row['zero_rows'] = 1
                p.write_text(json.dumps(row)+'\n')
                f = json.loads((root/'FLUSHED.json').read_text())
                f['sequences']['smoke0']['ledger_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                (root/'FLUSHED.json').write_text(json.dumps(f))
            if failure == 'missing_row':
                p = root/'smoke0/taps.bin'; p.write_bytes(p.read_bytes()[:-30720])
        io.freeze = bad_freeze
    with pytest.raises(ValueError):
        M.run(io, inputs, REPO, out, 'synthetic')
    assert not (out/'report.json').exists()


@pytest.mark.parametrize('mode', ['missing','http400','no_drafting'])
def test_option_ii_negative_selects_replay_without_claiming_support(tmp_path, mode):
    io, inputs, out = dry(tmp_path)
    post = io.post
    def unsupported(name, endpoint, body):
        if name == 'prompt-1' and mode == 'http400':
            io.calls.append(name)
            return {'error':'unsupported', 'http_status':400}
        response = post(name, endpoint, body)
        if name == 'prompt-1':
            if mode == 'missing': response['choices'][0].pop('logprobs')
            else: response['metrics']['speculative_decoding']['num_draft_tokens'] = 0
        return response
    io.post = unsupported
    result = M.run(io, inputs, REPO, out, 'synthetic')
    assert result['pass'] is True and result['option_ii'] is False
    assert result['top20_source'] == 'prefill_replay_top20_only'


def chat_logprobs(rows):
    """rows: list of (sampled token, listed [(token, logprob)], committed logprob)."""
    return {'choices': [{'finish_reason': 'length', 'token_ids': [t for t, _, _ in rows], 'logprobs': {'content': [
        {'token': f'token_id:{t}', 'logprob': lp, 'top_logprobs': [{'token': f'token_id:{k}', 'logprob': v} for k, v in listed]}
        for t, listed, lp in rows]}}]}


TOP = [(100+i, -3.5-0.25*i) for i in range(20)]  # sorted top20, mass ~0.14


@pytest.mark.parametrize('case,ok', [
    ('sorted', True),
    ('sampled_first_not_argmax', True),     # R5/R6: chat lists the sampled token first
    ('sampled_is_20th', False),             # rank 20 is indistinguishable from below-20 (true 20th lost): refuse
    ('sampled_tied_with_19th', True),       # a tie at the edge loses nothing: every listed value is a top20 value
    ('sampled_strictly_lowest', False),     # sample + ranks 1-19: the true 20th may be lost (S29 finding 2)
    ('21_entries', False),                  # chat never returns more than top_logprobs
    ('committed_absent', False),            # chat always includes the sampled token
    ('19_entries', False),
    ('duplicate', False),
    ('mass_above_one', False),
    ('committed_lp_mismatch', False),
    ('positive_lp', False),
])
def test_top20_ok_ranks_by_value_and_accepts_the_sampled_entry(case, ok):
    listed, tok, lp = list(TOP), 100, TOP[0][1]
    if case == 'sampled_first_not_argmax':
        tok, lp = 104, TOP[4][1]
        listed = [TOP[4]] + TOP[:4] + TOP[5:]
    elif case == 'sampled_is_20th':
        tok, lp = 119, TOP[19][1]
        listed = [TOP[19]] + TOP[:19]
    elif case == 'sampled_tied_with_19th':
        tok, lp = 999, TOP[18][1]
        listed = [(999, TOP[18][1])] + TOP[:19]
    elif case == 'sampled_strictly_lowest':
        tok, lp = 999, -9.0
        listed = [(999, -9.0)] + TOP[:19]
    elif case == '21_entries':
        tok, lp = 999, -9.0
        listed = [(999, -9.0)] + TOP
    elif case == 'committed_absent':
        tok, lp = 999, -9.0
    elif case == '19_entries':
        listed = TOP[:19]
    elif case == 'duplicate':
        listed = TOP[:19] + [TOP[0]]
    elif case == 'mass_above_one':
        listed = [(100+i, -0.01) for i in range(20)]
        lp = -0.01
    elif case == 'committed_lp_mismatch':
        lp = TOP[0][1] - 0.1
    elif case == 'positive_lp':
        listed = [(100, 0.1)] + TOP[1:]
    assert M.top20_ok(chat_logprobs([(tok, listed, lp), (101, list(TOP), TOP[1][1])])) is ok


def test_window_gate_requires_continuous_real_drain_evidence():
    good = {'source':'a'*40, 'coordinator':'Deepseek/coordinator', 'created_unix':1000,
        'drain_samples':[{'t':t,'metrics':'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'}
                         for t in range(0,301,10)]}
    M.window_gate(good, 'a'*40, 1001)
    for fault in ('busy','blind','short','stale'):
        bad = deepcopy(good)
        if fault == 'busy': bad['drain_samples'][8]['metrics'] = 'vllm:num_requests_running{} 1\nvllm:num_requests_waiting{} 0\n'
        if fault == 'blind': bad['drain_samples'][8]['metrics'] = ''
        if fault == 'short': bad['drain_samples'].pop()
        if fault == 'stale': bad['created_unix'] = 0
        with pytest.raises(ValueError): M.window_gate(bad, 'a'*40, 4000 if fault == 'stale' else 1001)


def base_fixture(tmp_path):
    data = subprocess.check_output(['git','-C',str(REPO),'show',A.BASE_COMMIT+':dsv41.env'])
    start = subprocess.check_output(['git','-C',str(REPO),'show',A.BASE_COMMIT+':start.sh'])
    (tmp_path/'dsv41.env').write_bytes(data)
    (tmp_path/'start.sh').write_bytes(start)
    p = tmp_path/'offload/kvpack'; p.mkdir(parents=True)
    (p/'MANIFEST.json').write_bytes((REPO/'offload/kvpack/MANIFEST.json').read_bytes())
    log = tmp_path/'logs/k1-promotion'; log.mkdir(parents=True)
    record = {'profile':'327680','original':{'dsv41.env':'original','start.sh':'original'},
              'promoted':{'dsv41.env':A.BASE_ENV,'start.sh':A.BASE_START}}
    (log/'files.json').write_text(json.dumps(record, indent=2))
    review = tmp_path/'review.json'
    after = A.render(data, 'a'*40, 'run1')
    review.write_text(json.dumps({'schema':'d2.decode-apply-review.v1','approved':True,
        'coordinator':'Deepseek/coordinator','source':'a'*40,'base_commit':A.BASE_COMMIT,
        'base_env_sha256':A.BASE_ENV,'capture_env_sha256':hashlib.sha256(after).hexdigest()}))
    return data, start, (log/'files.json').read_bytes(), review


def test_apply_preserves_k1_lineage_and_rolls_back_byte_identically(tmp_path):
    data, start, record, review = base_fixture(tmp_path)
    A.apply(tmp_path, 'a'*40, 'run1', review)
    changed = json.loads((tmp_path/'logs/k1-promotion/files.json').read_text())
    assert changed['d2_decode_derivative']['base_env_sha256'] == A.BASE_ENV
    assert changed['promoted']['dsv41.env'] == A.digest(tmp_path/'dsv41.env')
    assert (tmp_path/'dsv41.env').read_bytes().startswith(data)
    assert (tmp_path/'start.sh').read_bytes() == start
    # Shell only: appended args/env parse and retain all base K1 values.
    script = 'source "$1"; test "$PROD_MAX_LEN" = 327680; test "$PROD_KV_BYTES" = 6442450944; [[ "$OFFLOAD_DOCKER_ENV" == *VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode* ]]'
    subprocess.run(['bash','-c',script,'test',str(tmp_path/'dsv41.env')],check=True)
    A.rollback(tmp_path, 'run1')
    assert (tmp_path/'dsv41.env').read_bytes() == data
    assert (tmp_path/'logs/k1-promotion/files.json').read_bytes() == record


def test_apply_and_rollback_refuse_unreviewed_or_unrelated_edits(tmp_path):
    data, _, _, review = base_fixture(tmp_path)
    bad = json.loads(review.read_text()); bad['approved'] = False; review.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='review'): A.apply(tmp_path,'a'*40,'run1',review)
    assert (tmp_path/'dsv41.env').read_bytes() == data
    bad['approved'] = True; review.write_text(json.dumps(bad))
    A.apply(tmp_path,'a'*40,'run1',review)
    with (tmp_path/'dsv41.env').open('ab') as fh: fh.write(b'# unrelated edit\n')
    with pytest.raises(ValueError,match='unrelated'): A.rollback(tmp_path,'run1')


def test_capture_config_matches_the_production_k1_d_profile():
    script = ('set -eu; export D2_DECODE_RUN=synthetic; . offload/launch/configs/D-decode-capture.env; '
              'export EXP_DOCKER_ENV OVERLAYS MAX_LEN_OVERRIDE KV_BYTES_OVERRIDE; '
              'python3 -c "import os,json; print(json.dumps({k:os.environ[k] for k in '
              "['EXP_DOCKER_ENV','OVERLAYS','MAX_LEN_OVERRIDE','KV_BYTES_OVERRIDE']}))\"")
    values = json.loads(subprocess.check_output(['bash','-c',script],cwd=REPO,text=True))
    env = dict(pair.split('=',1) for pair in values['EXP_DOCKER_ENV'].split())
    profiles = json.loads((REPO/'offload/operations/profiles.v1.json').read_text())
    profile = next(p for p in profiles['profiles'] if p['name'] == 'K1-327680')
    for key, expected in profile['match']['env'].items():
        assert env.get(key) == expected
    assert values['MAX_LEN_OVERRIDE'] == '327680' and values['KV_BYTES_OVERRIDE'] == '6442450944'
    assert env['VLLM_DSPARK_DECODE_CAPTURE_DIR'] == '/d2-decode'
    assert 'capture/decode_capture.py=' in values['OVERLAYS']


def test_tagged_smoke_must_include_actual_decode_rows(tmp_path):
    io, inputs, out = dry(tmp_path)
    post = io.post
    def too_short(*args, **kwargs):
        response = post(*args, **kwargs)
        response['choices'][0]['token_ids'] = [20]
        return response
    io.post = too_short
    with pytest.raises(ValueError, match='too short'):
        M.run(io, inputs, REPO, out, 'synthetic')
    assert not (out/'report.json').exists()


def test_k1_control_can_preserve_a_historical_chat_request_and_tracing_defaults(tmp_path):
    io, inputs, out = dry(tmp_path)
    control = inputs['requests'][2]
    control['endpoint'] = '/v1/chat/completions'
    control['body'].pop('prompt')
    control['body']['messages'] = [{'role':'user','content':'synthetic'}]
    control['body']['max_tokens'] = 300
    control['body']['priority'] = 0
    proof_path = Path(inputs['greedy_reference']['request_receipt'])
    proof = json.loads(proof_path.read_text())
    proof['request'] = dict(control['body'])
    for key in ('request_id','return_token_ids','stream'):
        proof['request'].pop(key)
    proof['endpoint'] = control['endpoint']
    proof_path.write_text(json.dumps(proof))
    inputs['greedy_reference']['request_receipt_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    assert M.run(io, inputs, REPO, out, 'synthetic')['untagged_greedy_identical'] is True


@pytest.mark.parametrize('kind', ['ttft','step','missing','reset','foreign','cache'])
def test_smoke_refuses_latency_regression_or_unusable_reference(tmp_path,kind):
    io,inputs,out = dry(tmp_path)
    post = io.post
    def disturbed(name, endpoint, body):
        result = post(name, endpoint, body)
        if name.startswith('control-'):
            if kind == 'ttft':
                io.latency_metrics = lambda n: (f'vllm:time_to_first_token_seconds_count{{}} {n}\n'
                    f'vllm:time_to_first_token_seconds_sum{{}} {n*.12}\n'
                    f'vllm:inter_token_latency_seconds_count{{}} {n*2}\n'
                    f'vllm:inter_token_latency_seconds_sum{{}} {n*.08}\n')
            if kind == 'step':
                io.latency_metrics = lambda n: (f'vllm:time_to_first_token_seconds_count{{}} {n}\n'
                    f'vllm:time_to_first_token_seconds_sum{{}} {n*.1}\n'
                    f'vllm:inter_token_latency_seconds_count{{}} {n*2}\n'
                    f'vllm:inter_token_latency_seconds_sum{{}} {n*.12}\n')
            if kind == 'foreign': io.completed += 1
            if kind == 'reset': io.completed = 0
        return result
    io.post = disturbed
    if kind == 'missing':
        proof_path = Path(inputs['greedy_reference']['request_receipt'])
        proof = json.loads(proof_path.read_text())
        proof.pop('samples')
        proof_path.write_text(json.dumps(proof))
        inputs['greedy_reference']['request_receipt_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    if kind == 'cache':
        saved_post = io.post
        def cached(*args,**kwargs):
            result = saved_post(*args,**kwargs)
            result['usage']['prompt_tokens_details']['cached_tokens'] = 1
            return result
        io.post = cached
    with pytest.raises(ValueError):
        M.run(io,inputs,REPO,out,'synthetic')
    assert not (out/'report.json').exists()


def test_latency_bound_is_five_percent_and_reports_engine_steps():
    baseline = {'ttft_s':1.0,'step_mean_s':.1,'step_samples':10.0}
    assert M.latency_gate([baseline]*5,[{'ttft_s':1.05,'step_mean_s':.105,'step_samples':10.0}]*5)['pass']
    for measured in ({'ttft_s':1.051,'step_mean_s':.1}, {'ttft_s':1.,'step_mean_s':.1051}):
        with pytest.raises(ValueError,match='above 5%'):
            M.latency_gate([baseline]*5,[measured]*5)


def test_smoke_records_baseline_and_measured_untagged_timings(tmp_path):
    io,inputs,out = dry(tmp_path)
    report = M.run(io,inputs,REPO,out,'synthetic')
    assert report['untagged_latency']['max_regression_fraction'] == .05
    assert len(report['untagged_latency']['measured_samples']) == 5
    assert all(s['step_samples'] == 2 for s in report['untagged_latency']['measured_samples'])
    assert (out/'control-timed-5-before.prom').is_file()
    assert (out/'control-timed-5-after.prom').is_file()


def test_runtime_and_hold_rechecked_before_each_request(tmp_path):
    io,inputs,out=dry(tmp_path)
    call=io.call
    checked=0
    def restart(name,*args,**kwargs):
        nonlocal checked
        result=call(name,*args,**kwargs)
        if name == 'identity-sparkhead':
            checked+=1
            if checked == 3:
                rows=[json.loads(line) for line in result.splitlines()]
                rows[0]['id']='b'*64
                return '\n'.join(json.dumps(r) for r in rows)
        return result
    io.call=restart
    with pytest.raises(ValueError,match='container changed'):
        M.run(io,inputs,REPO,out,'synthetic')
    assert io.calls.count('hold') == 3 and io.calls.count('identity-sparkhead') == 3
    assert [c for c in io.calls if c.startswith('prompt-')] == []
    assert not (out/'report.json').exists()


def test_latency_uses_five_medians_and_retains_five_percent_bound():
    baseline = [{'ttft_s':1.,'step_mean_s':.1}]*5
    noisy = [{'ttft_s':1.,'step_mean_s':.1}]*4 + [{'ttft_s':10.,'step_mean_s':1.}]
    result = M.latency_gate(baseline, noisy)
    assert result['measured'] == {'ttft_s':1.,'step_mean_s':.1}
    assert result['statistic'] == 'median'
    for count in (1,4,6):
        with pytest.raises(ValueError,match='exactly 5'):
            M.latency_gate(baseline, noisy[:1]*count)


def test_controls_are_warmed_and_timed_only_after_durable_flush(tmp_path):
    io, inputs, out = dry(tmp_path)
    report = M.run(io, inputs, REPO, out, 'synthetic')
    assert io.calls.index('flush') < io.calls.index('control-warmup')
    assert io.calls.index('control-warmup') < io.calls.index('control-timed-1')
    assert [c for c in io.calls if c.startswith('control-timed-')] == [f'control-timed-{i}' for i in range(1,6)]
    assert report['informational_active_writer']['gated'] is False
    assert report['informational_repeats'] == 3


@pytest.mark.parametrize('sent', [0, 1, 2])
def test_informational_shortfall_is_recorded_not_refused(tmp_path, sent):
    # R5 (2026-10-24): two controls were SENT, so the active interval held 2 tagged + 2 completions, not + 3.
    io, inputs, out = dry(tmp_path)
    io.informational_sent = sent
    report = M.run(io, inputs, REPO, out, 'synthetic')
    assert report['informational_repeats'] == sent and report['http_requests'] == 8 + sent
    assert report['informational_active_writer']['informational_status'].startswith('incomplete')
    assert report['untagged_greedy_identical'] is True and report['ledger_complete'] is True


@pytest.mark.parametrize('fault', ['status','late'])
def test_informational_controls_need_overlap(tmp_path, fault):
    io, inputs, out = dry(tmp_path)
    active = io.active
    def corrupt(*args):
        result = active(*args)
        if fault == 'status': del result['informational_status']
        else: result['informational'][1]['both_tagged_inflight_at_start'] = False
        return result
    io.active = corrupt
    with pytest.raises(ValueError): M.run(io, inputs, REPO, out, 'synthetic')
    assert 'control-warmup' not in io.calls


def set_text(reply, edit):
    c = reply['choices'][0]
    if 'text' in c: c['text'] = edit(c['text'])
    else: c['message'] = {'content': edit(c['message']['content'])}


def test_informational_output_drift_is_recorded_not_gated(tmp_path):
    # R4 (2026-10-24): controls batched beside the tagged pair diverged from K1 at char 52; batch-shape numerics.
    io, inputs, out = dry(tmp_path)
    active = io.active
    def drift(*args):
        result = active(*args)
        set_text(result['informational'][1]['response'], lambda old: old[:2]+'#drift')
        return result
    io.active = drift
    report = M.run(io, inputs, REPO, out, 'synthetic')
    rows = report['informational_active_writer']['outputs_vs_k1']
    assert report['informational_active_writer']['outputs_gated'] is False
    assert [r['text_identical'] for r in rows] == [True, False, True]
    assert rows[1]['first_divergence_char'] == 2 and rows[0]['first_divergence_char'] is None
    assert all(r['prompt_echo_identical'] for r in rows)
    assert json.loads((out/'informational-outputs.json').read_text()) == rows
    assert report['untagged_greedy_identical'] is True and report['pass'] is True


def test_isolated_control_drift_still_refuses(tmp_path):
    io, inputs, out = dry(tmp_path)
    post = io.post
    def drift(name, endpoint, body):
        reply = post(name, endpoint, body)
        if name == 'control-timed-3':
            set_text(reply, lambda old: 'drift')
        return reply
    io.post = drift
    with pytest.raises(ValueError, match='untagged greedy output differs'): M.run(io, inputs, REPO, out, 'synthetic')


@pytest.mark.parametrize('fault',['duplicate','limit'])
def test_receipt_requires_distinct_baseline_attempts_within_ten(tmp_path,fault):
    io,inputs,out = dry(tmp_path)
    path = Path(inputs['greedy_reference']['request_receipt'])
    proof = json.loads(path.read_text())
    proof['samples'][2]['attempt'] = proof['samples'][1]['attempt'] if fault == 'duplicate' else 11
    path.write_text(json.dumps(proof))
    inputs['greedy_reference']['request_receipt_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='distinct or exceed'):
        M.checked_spec(inputs)


def test_chat_drafting_proof_uses_decode_ledger_when_api_counters_disabled(tmp_path):
    ledger = tmp_path/'ledger.jsonl'
    ledger.write_text(json.dumps({'first':0,'rows':128})+'\n')
    assert M.request_drafting({},ledger,128) is False  # prefill is not drafting
    ledger.write_text(json.dumps({'first':128,'rows':1})+'\n')
    assert M.request_drafting({},ledger,128) is False
    ledger.write_text(json.dumps({'first':128,'rows':3})+'\n')
    assert M.request_drafting({},ledger,128) is True
    assert M.request_drafting({'metrics':{'speculative_decoding':{'num_draft_tokens':0}}},ledger,128) is False
    assert M.request_drafting({'metrics':{'speculative_decoding':{'num_draft_tokens':2}}},ledger,128) is True
