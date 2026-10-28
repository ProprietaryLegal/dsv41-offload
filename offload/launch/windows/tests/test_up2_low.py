"""Offline rehearsal: no Spark, x86host, GPU or listening HTTP server."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.launch.windows import up2_checks as ck, up2_low as low, up2_proxy_stats as proxy


class Fake:
    def __init__(self, out: Path, *, slow: bool = False, fail: str = ''):
        self.out,self.slow,self.fail = out,slow,fail
        self.events: list[dict[str,Json]] = []
        self.starts: list[tuple[str,int,int]] = []
        self.probes: list[tuple[str,str,int,int]] = []
        self.number = 0

    def start(self, arm: str, replay: int, audit: int) -> None:
        self.starts.append((arm,replay,audit))
        if self.fail == arm: raise ValueError('injected start failure')

    def request(self, arm: str, name: str, target: int, budget: int, *, numeric: bool = False,
                force_length: bool = False, rep: int = 0, cold: bool = False) -> dict[str,Json]:
        self.probes.append((arm,name,target,budget))
        self.number += 1
        ident = str(self.number)
        self.events.extend([{'event':'send_intent','id':ident},{'event':'http_opened','id':ident}])
        if name == self.fail:
            self.events.append({'event':'failed','id':ident})
            raise ValueError('injected partial request failure')
        self.events.append({'event':'completed','id':ident})
        return {'window_id':str(self.out.resolve()), 'kind':'cold' if cold else name,'rep':rep,'cap':327680,'target_ctx':target,
            'cached_tokens':0,'uncached_prompt_tokens':target,
            'prompt_sha256':hashlib.sha256(str(target).encode()).hexdigest(),
            'cache_salt_sha256':hashlib.sha256(f'{name}:{target}:{rep}'.encode()).hexdigest(),
            'usage':{'prompt_tokens':target,'completion_tokens':budget},'finish':'length',
            'sampling':{'temperature':1,'top_p':1,'seed':42,'priority':10,'max_tokens':budget},
            'ttft_s':10000 if self.slow and arm == 'on' else 100+rep,'total_s':10001,
            'probabilities':{'token-a':.999,'token-b':.0005},'text_sha256':'synthetic-same',
            'batch_numeric_delta_INFORMATIONAL':1000000,'batch_text_identical_INFORMATIONAL':False}

    def native(self, arm: str, *, complete: bool) -> int:
        if self.fail == 'native': raise ValueError('NOT RUN/HOLD')
        return 4092

    def needles(self, arm: str) -> None:
        for n in (150000,300000): self.request(arm,'needle',n,300)

    def clean(self, arm: str, audit: int) -> None:
        if self.fail == 'clean': raise ValueError('injected health failure')


def run_fake(tmp_path: Path, *, slow: bool = False, fail: str = '') -> tuple[Fake,dict[str,Json],list[dict[str,Json]]]:
    fake = Fake(tmp_path,slow=slow,fail=fail)
    checkpoints: list[dict[str,Json]] = []
    result = low.sequence(fake,'all',checkpoints.append)
    return fake,result,checkpoints


def test_entire_off_on_off_rehearsal_and_info_never_gates(tmp_path: Path) -> None:
    fake,result,checkpoints = run_fake(tmp_path,slow=True)
    assert fake.starts == low.ARMS
    assert result['completed_cells'] == 18 and result['diagnostics'] == 'PASS'
    assert len(checkpoints) == 19
    assert ck.accounting(fake.events)['completed'] == 49
    assert [p[2] for p in fake.probes if p[1].startswith('tail')] == [147313,147439,147440,147441]*2
    assert [(p[2],p[3]) for p in fake.probes if p[1] == 'near-cap'] == [(327675,5)]*2
    points = ck.rows(result['points'])
    assert points[0]['ratio_median'] is not None and cast(float,points[0]['ratio_median']) > 90
    # A giant timing loss and batch text/numeric differences remain observations.
    assert result['decision'] == 'HOLD; no automatic adoption'


def test_partial_cells_account_for_only_actual_submissions(tmp_path: Path) -> None:
    fake = Fake(tmp_path,fail='cold150000-1')
    checkpoints: list[dict[str,Json]] = []
    with pytest.raises(ValueError,match='partial'):
        low.sequence(fake,'all',checkpoints.append)
    counts = ck.accounting(fake.events)
    assert counts['transport_attempts'] == 31 and counts['completed'] == 30 and counts['failed'] == 1
    assert checkpoints[-1]['completed_cells'] == 1


@pytest.mark.parametrize('sent', [0,1,2])
def test_best_effort_shortfalls_have_no_assumed_three_request_count(sent: int) -> None:
    events: list[dict[str,Json]] = []
    for i in range(sent):
        for event in ('send_intent','http_opened','completed'): events.append({'event':event,'id':str(i)})
    events.append({'event':'not_sent','id':'optional-ended-before-admission','reason':'active interval ended'})
    assert ck.accounting(events)['http_opened'] == sent
    assert ck.accounting(events)['completed'] == sent


def test_unresolved_dispatch_is_not_claimed_complete() -> None:
    assert ck.accounting([{'event':'send_intent','id':'a'}]) == {
        'transport_attempts':1,'http_opened':0,'completed':0,'failed':0,'canceled':0,'unresolved':1,
        'dispatch_uncertain':1,'planned_unsent_not_counted':True}


def stats() -> dict[str,Json]:
    return {'counters':{'rounds_ok':20},'policy':'capacity','async':True,'async_proxy':{'mode':'process'},
        'async_handoff':{'mlock':'ok'},'async_watchdog':{'fires':0,'running':True},'proxy_torch_loaded':False}


@pytest.mark.parametrize('counter', ck.STOP_COUNTERS)
def test_every_hard_counter_stops(counter: str) -> None:
    data = stats()
    record(data['counters'])[counter] = 1
    with pytest.raises(ValueError,match=counter): ck.counters(data)


@pytest.mark.parametrize('field,value', [('proxy_torch_loaded',True),('policy','resilience'),('async',False),
    ('async_dead',True),('async_watchdog',{'fires':0,'running':False}),('async_proxy',{'mode':'thread'}),
    ('async_handoff',{'mlock':'failed'})])
def test_runtime_proxy_hard_gates(field: str, value: Json) -> None:
    data = stats(); data[field] = value
    with pytest.raises(ValueError): ck.counters(data)


def test_expected_ordinary_faults_are_scoped_and_new_errors_still_stop() -> None:
    before = stats(); record(before['counters'])['fallback_internal_error'] = 2
    after = copy.deepcopy(before)
    ck.counters(after,before=before)
    record(after['counters'])['fallback_internal_error'] = 3
    with pytest.raises(ValueError,match='internal_error'): ck.counters(after,before=before)


@pytest.mark.parametrize('line', ['PARTIAL_WINDOW','0075 rank provenance mismatch','0075 admitted successor reservation invalidated',
    'replay audit final suffix incomplete','EngineDeadError','CUDA error','Traceback (most recent call last)',
    'NaN','identity mismatch','stale-generation usable','invalid replay tap provenance','replay skip would remove taps'])
def test_error_logs_stop(line: str) -> None:
    with pytest.raises(ValueError): ck.clean_logs(line)


def test_good_logs_numeric_observations_do_not_stop() -> None:
    ck.clean_logs('remote DSpark capacity: 1024 rounds, ok1024; batch text differs; coarse delta1.2 timing+999%; financial')


def test_memory_floor_and_missing_fields() -> None:
    assert ck.memory('MemAvailable: 3145728 kB\nSwapFree: 10 kB') == (3145728,10)
    for text in ('MemAvailable: 3145727 kB\nSwapFree: 10 kB','MemAvailable: 9999999 kB',''):
        with pytest.raises(ValueError): ck.memory(text)


def test_deadline_comes_from_hold_and_never_renews() -> None:
    hold = '2026-10-24T21:00:00Z DSV41-OFFLOAD (session coordinator): UP2'
    import datetime
    t = datetime.datetime.fromisoformat('2026-10-24T21:00:00+00:00').timestamp()
    assert ck.window_deadlines(hold,t+3600) == (t,t+4500,t+5400)
    for now in (t-1,t+4500,t+5400):
        with pytest.raises(ValueError): ck.window_deadlines(hold,now)


def packet(tmp_path: Path) -> dict[str,Json]:
    evidence = tmp_path/'native-evidence.txt'; evidence.write_text('SYNTHETIC CPU ONLY\n')
    return {'schema':'up2-native-v1','classification':'MEASURED','arm':'diag-off','container_ids':['a','b'],
        'manifest_sha256':'manifest','q':4092,'tail_sizes':[147313,147439,147440,147441],
        'audited_rounds':20,'checks':{key:'PASS' for key in ck.NATIVE_CHECKS},
        'request_events':[{'event':event,'id':'native-probe'} for event in ('send_intent','http_opened','completed')],
        'evidence':[{'path':evidence.name,'sha256':ck.digest(evidence)}]}


@pytest.mark.parametrize('name',ck.NATIVE_CHECKS)
def test_each_native_safety_gate_missing_retains_hold(tmp_path: Path,name: str) -> None:
    data = packet(tmp_path)
    record(data['checks'])[name] = 'NOT RUN'
    with pytest.raises(ValueError,match='HOLD'): ck.native(data,'diag-off',['a','b'],'manifest',tmp_path,complete=True)


@pytest.mark.parametrize('field,value', [('container_ids',['old','b']),('manifest_sha256','drift'),('q',128),
    ('tail_sizes',[147457,147583,147584,147585]),('audited_rounds',0)])
def test_native_identity_q_tails_audit_fail_closed(tmp_path: Path,field: str,value: Json) -> None:
    data = packet(tmp_path); data[field] = value
    with pytest.raises(ValueError): ck.native(data,'diag-off',['a','b'],'manifest',tmp_path,complete=True)


def test_native_artifact_hash_tampering_stops(tmp_path: Path) -> None:
    data = packet(tmp_path)
    assert ck.native(data,'diag-off',['a','b'],'manifest',tmp_path,complete=True) == 4092
    (tmp_path/'native-evidence.txt').write_text('tampered')
    with pytest.raises(ValueError,match='hash'): ck.native(data,'diag-off',['a','b'],'manifest',tmp_path,complete=True)


def test_numerical_floor_mass_and_sampled_first_format() -> None:
    a: dict[str,Json] = {'probabilities':{'a':.999,'b':.0005}}
    b: dict[str,Json] = {'probabilities':{'b':.0005,'a':.999}}
    assert ck.numerical(a,a,b)['on_off_kl'] == 0
    with pytest.raises(ValueError,match='mass'): ck.numerical(a,a,{'probabilities':{'x':.9}})
    with pytest.raises(ValueError,match='floor'): ck.numerical(
        {'probabilities':{'a':.99,'b':.01}}, {'probabilities':{'a':.99,'b':.01}},
        {'probabilities':{'a':.01,'b':.99}})


def runtime_fixture() -> tuple[dict[str,Json],dict[str,Json]]:
    env: dict[str,Json] = {'VLLM_DSV41_KV_PAIR':'3','VLLM_DSV41_DROP_MTP_BUFFER':'1',
        'VLLM_EXL3_TF_PROMPT_TRANSIENT':'1','VLLM_EXL3_TF_PROMPT_ALIAS':'1','VLLM_INDEXER_PREFILL_BUFFER_FACTOR':'4',
        'VLLM_DSPARK_REMOTE_POLICY':'capacity','VLLM_DSPARK_REMOTE_ASYNC':'1',
        'VLLM_DSPARK_REMOTE_ASYNC_PROXY':'process','VLLM_ENGRAM_REMOTE_REQUIRED':'0'}
    prod: dict[str,Json] = {'image':'sha256:v5','files':{'model':'sha256:legal-v2a'},'engine':'vllm-pinned',
        'engram_identity':{'jit':'sha256:engram'},
        'env':env,'argv':'vllm serve /srv/dsv41/legal-v2a --port 8000 --max-model-len 327680 --max-num-seqs 4 '
        '--max-num-batched-tokens 4096 --kv-cache-memory-bytes 6442450944 --long-prefill-token-threshold 1024'}
    live = copy.deepcopy(prod); live['state'] = 'running'
    live['argv'] = str(prod['argv']).replace('--port 8000','--port 8001')
    record(live['env']).update({'VLLM_DSV41_REPLAY_SKIP':'1','VLLM_DSV41_REPLAY_AUDIT':'0',
        'VLLM_EXL3_MOE_SPLIT_MAX_TOKENS':'0','VLLM_DSV41_VOR_FUSED':'0','VLLM_ENGRAM_REMOTE_REQUIRED':'1'})
    return prod,live


@pytest.mark.parametrize('field,value', [('image','wrong'),('files',{'model':'wrong'}),('engine','wrong'),
    ('state','dead'),('argv','vllm serve wrong'),('engram_identity',{'jit':'wrong'})])
def test_runtime_identity_and_argv_gates(field: str,value: Json) -> None:
    prod,live = runtime_fixture(); live[field] = value
    with pytest.raises((ValueError,IndexError)): ck.runtime(prod,live,1,0)


def test_runtime_exact_flags_and_k1() -> None:
    prod,live = runtime_fixture(); ck.runtime(prod,live,1,0)
    record(live['env'])['VLLM_DSV41_KV_PAIR'] = '1'
    with pytest.raises(ValueError): ck.runtime(prod,live,1,0)


def test_cold_identity_and_pairing_failures(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    row = fake.request('off-a','cold150000-0',150000,1500,cold=True)
    ck.cold(row,150000,0)
    for key,value in [('cached_tokens',384),('cap',524288),('uncached_prompt_tokens',149616)]:
        wrong = {**row,key:value}
        with pytest.raises(ValueError): ck.cold(wrong,150000,0)
    wrong = {**row,'prompt_sha256':'f'*64}
    with pytest.raises(ValueError,match='hash'): ck.summarize({'off-a':[row],'on':[wrong],'off-b':[row]})


def test_proxy_stats_flush_only_control_word_and_ack(tmp_path: Path) -> None:
    fd = os.memfd_create('up2-test')
    os.ftruncate(fd,proxy.PAGE+65536)
    for offset,value in ((0,proxy.MAGIC),(4,1),(192,1),(256,0)):
        os.pwrite(fd,struct.pack('<I',value),proxy.PAGE+offset)
    def housekeeping() -> None:
        end = time.monotonic()+2
        while time.monotonic() < end:
            value = os.pread(fd,4,proxy.PAGE+proxy.FLUSH_REQ)
            if value != b'\0'*4:
                os.pwrite(fd,value,proxy.PAGE+proxy.FLUSH_ACK)
                return
            time.sleep(.001)
    worker = threading.Thread(target=housekeeping); worker.start()
    try:
        proxy.flush(fd)
        assert os.pread(fd,4,proxy.PAGE+proxy.FLUSH_ACK) == struct.pack('<I',1)
    finally:
        worker.join(timeout=3); os.close(fd)


def test_proxy_stats_layout_and_flush_timeout_fail(tmp_path: Path) -> None:
    fd = os.memfd_create('up2-test'); os.ftruncate(fd,proxy.PAGE+65536)
    try:
        with pytest.raises(ValueError): proxy.flush(fd)
        for offset,value in ((0,proxy.MAGIC),(4,1),(192,1)):
            os.pwrite(fd,struct.pack('<I',value),proxy.PAGE+offset)
        with pytest.raises(TimeoutError): proxy.flush(fd,timeout=.02)
    finally: os.close(fd)


def test_proxy_control_offsets_are_pinned_to_actual_reviewed_source() -> None:
    import ast
    constants: dict[str,int] = {}
    for name in ('hostflag','shmregion'):
        tree = ast.parse((low.ROOT/f'offload/dspark/spark/remote_dspark/{name}.py').read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,int):
                for target in node.targets:
                    if isinstance(target,ast.Name): constants[target.id] = node.value.value
    assert constants['PAGE_BYTES'] == proxy.PAGE and constants['MAGIC'] == proxy.MAGIC
    assert constants['C_FLUSH_REQ'] == proxy.FLUSH_REQ and constants['C_FLUSH_ACK'] == proxy.FLUSH_ACK


def test_emitted_metadata_probes_compile_without_model_imports() -> None:
    import ast
    for source in (low.ENGRAM_PROBE,low.X86HOST_PROBE):
        compile(source,'coordinator-metadata-probe','exec')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Import):
                assert all(not name.name.startswith(('torch','vllm')) for name in node.names)
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('torch','vllm'))


def test_dry_and_abort_always_print_restore_without_remote_calls(tmp_path: Path) -> None:
    for args in (['--dry-run'],['--execute','--out',str(tmp_path/'forbidden')]):
        env = {k:v for k,v in os.environ.items() if k not in ('D2_COORDINATOR','WINDOW_COORDINATOR')}
        result = subprocess.run([sys.executable,'-m','offload.launch.windows.up2_low',*args],
            text=True,capture_output=True,env=env,check=False)
        assert result.returncode == (0 if args[0] == '--dry-run' else 1)
        for step in ('start','check','close'): assert f'W8/restore.sh {step}' in result.stdout
    assert not (tmp_path/'forbidden').exists()


def test_supervisor_external_stop_kills_only_owned_child(tmp_path: Path) -> None:
    supervisor = low.Supervisor(tmp_path,time.time()+60)
    timer = threading.Timer(.1,lambda:(tmp_path/'STOP').write_text('operator stop'))
    timer.start()
    try:
        with pytest.raises(ValueError,match='STOP'):
            supervisor.run('owned',[sys.executable,'-c','import time; time.sleep(10)'])
        assert supervisor.active is None
    finally:
        timer.join(timeout=1); supervisor.close()


def test_missing_or_invalid_timing_is_recorded_without_stopping(tmp_path: Path) -> None:
    row = Fake(tmp_path).request('off-a','cold150000-0',150000,1500,cold=True)
    for value in (None,-1,'unavailable'):
        sample = {**row,'ttft_s':value,'total_s':None}
        ck.cold(sample,150000,0)
        result = ck.summarize({'off-a':[sample],'on':[sample],'off-b':[sample]})
        assert ck.rows(result['points'])[0]['ratio_median'] is None


class FakeResponse:
    def __init__(self, data: bytes, lines: list[bytes] | None = None):
        self.data,self.lines = data,lines or []
    def __enter__(self) -> FakeResponse: return self
    def __exit__(self, *args: object) -> None: pass
    def read(self) -> bytes: return self.data
    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


@pytest.mark.parametrize('screen',[False,True])
@pytest.mark.parametrize('numeric,broken',[(False,False),(True,False),(False,True)])
def test_real_request_driver_http_fake_and_failure_ledger(tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
                                                        numeric: bool,broken: bool,screen: bool) -> None:
    from offload.launch.windows import up2_request as request
    import urllib.request
    monkeypatch.setattr(request,'corpus',lambda _root: ('public materials '*1000,{'public':'sha'}))
    def open_fake(req: str | urllib.request.Request, timeout: float = 0) -> FakeResponse:
        url = req if isinstance(req,str) else req.full_url
        if url.endswith('/metrics'):
            return FakeResponse(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        if url.endswith('/v1/models'):
            return FakeResponse(json.dumps({'data':[{'id':'target','root':'/srv/dsv41/legal-v2a','max_model_len':327680}]}).encode())
        assert isinstance(req,urllib.request.Request) and isinstance(req.data,bytes)
        body = record(cast(Json,json.loads(req.data)))
        if url.endswith('/tokenize'):
            text = str(record(cast(list[Json],body['messages'])[0])['content'])
            return FakeResponse(json.dumps({'count':len(text)}).encode())
        assert url.endswith('/v1/chat/completions')
        assert body['temperature'] == (0 if numeric else 1) and body['priority'] == 10
        usage = {'prompt_tokens':512,'completion_tokens':8,'prompt_tokens_details':{'cached_tokens':0}}
        if numeric:
            return FakeResponse(json.dumps({'usage':usage,'choices':[{'finish_reason':'stop',
                'message':{'content':'PRIVATE_FAKE_GENERATED_TEXT'},'logprobs':{'content':[
                    {'top_logprobs':[{'token':'hidden-token','logprob':-.001},
                                     {'token':'another-token','logprob':-10}]}]}}]}).encode())
        lines = [b'data: '+json.dumps({'id':'request','choices':[{'delta':{'reasoning_content':'PRIVATE_FAKE_GENERATED_TEXT'},
            'finish_reason':'length'}],'usage':usage}).encode()]
        if not broken: lines.append(b'data: [DONE]')
        return FakeResponse(b'',lines)
    monkeypatch.setattr(urllib.request,'urlopen',open_fake)
    output,ledger = tmp_path/'result.json',tmp_path/'ledger.jsonl'
    if broken:
        with pytest.raises(RuntimeError,match='incomplete'):
            request.execute(512,8,'paired',output,ledger,time.time()+60,tmp_path,screen=screen)
        assert not output.exists()
    else:
        row = request.execute(512,8,'paired',output,ledger,time.time()+60,tmp_path,
                              numeric=numeric,force_length=not numeric,screen=screen)
        assert row['cached_tokens'] == 0 and row['target_ctx'] == 512
        assert row.get('screen',False) is screen
        if screen: assert ck.read(output)['adoptable'] is False
        assert 'PRIVATE_FAKE' not in output.read_text() and 'hidden-token' not in output.read_text()
    events = [record(cast(Json,json.loads(line))) for line in ledger.read_text().splitlines()]
    assert ck.accounting(events)['http_opened'] == 1
    assert ck.accounting(events)['completed'] == int(not broken)
    assert ck.accounting(events)['failed'] == int(broken)
    assert all(e.get('screen',False) is screen for e in events)


class StartFake(low.Live):
    """Run the real startup/health/worker/hash/argv logic with raw-command fakes."""
    def __init__(self, out: Path, fail: str):
        self.out,self.fail = out,fail
        self.remote,self.expected = 'reviewed','a'*40
        self.submit = time.time()+60
        self.ids = []
        self.previous_ids = set()
        self.production = [runtime_fixture()[0],runtime_fixture()[0]]
        self.verifier = 'FAKE VERIFIER'
        self.proxy_pid = 0
        self.replay,self.audit = 0,1
        self.launched = False

    def ssh(self, host: str, command: str, stdin: str | None = None) -> str:
        if command.endswith('/health'):
            if self.fail == 'health': raise ValueError('health failure')
            return ''
        if command.endswith('/v1/models'):
            return json.dumps({'data':[{'root':'/srv/dsv41/legal-v2a',
                'max_model_len':524288 if self.fail == 'cap' else 327680}]})
        if command.endswith('/metrics'):
            return 'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'
        if host == 'x86host':
            import datetime
            return json.dumps({'revision':'wrong' if self.fail == 'worker' else ck.WORKER,
                'current':'/prod/fb68c52','cwd':'/prod/fb68c52/offload/dspark',
                'cmd':['python','--tp','2','--max-model-len','1048576'],
                'gpus':'GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc',
                'status':{'tp':2,'tp_comm_errors':0,'listeners':['10.202.28.1:18620'],
                    'time':datetime.datetime.now(datetime.timezone.utc).isoformat()}})
        if 'start_exp.sh' in command:
            self.launched = True
            return ''
        if 'docker exec -i' in command:
            return 'wrong hashes' if self.fail == 'hash' else 'UP1 runtime hashes verified: 42 files;'
        if 'docker logs' in command:
            return ('GPU KV cache size: '+('1,604,926' if self.fail == 'kv' else '2,502,600')+
                '\nEngram table rows from the remote table server\nremote DSpark async mode on\n'
                'remote DSpark through proxy process pid 123\n'+('PARTIAL_WINDOW' if self.fail == 'engine' else ''))
        if 'docker inspect' in command: return 'rank0' if host == 'sparkhead' else 'rank1'
        return ''

    def capture(self, host: str) -> dict[str,Json]:
        _prod,live = runtime_fixture()
        live['id'] = 'rank0' if host == 'sparkhead' else 'rank1'
        record(live['env']).update({'VLLM_DSV41_REPLAY_SKIP':'0','VLLM_DSV41_REPLAY_AUDIT':'1'})
        if self.fail == 'identity': live['files'] = {'model':'wrong'}
        return live


@pytest.mark.parametrize('screen',[False,True])
@pytest.mark.parametrize('gate',['health','cap','worker','hash','kv','engine','identity'])
def test_real_startup_route_hard_failures_stop(tmp_path: Path,gate: str,screen: bool) -> None:
    backend = StartFake(tmp_path,gate)
    backend.screen = screen
    with pytest.raises(ValueError): backend.start('diag-off',0,1)
    if gate == 'worker': assert not backend.launched


def test_real_startup_route_with_raw_command_fake(tmp_path: Path) -> None:
    backend = StartFake(tmp_path,'')
    backend.start('diag-off',0,1)
    assert backend.ids == ['rank0','rank1'] and backend.proxy_pid == 123
    assert (tmp_path/'diag-off-identity.json').is_file()


@pytest.mark.parametrize('case',['low','EOF','stale'])
def test_live_memory_supervisor_with_local_only_monitor_fake(tmp_path: Path,monkeypatch: pytest.MonkeyPatch,case: str) -> None:
    script = tmp_path/'ssh-file-read-fake.py'
    text = ('import time,sys\n'
        f'print("MemAvailable: {3145727 if case == "low" else 5000000} kB;SwapFree: 100 kB",flush=True)\n')
    if case == 'swap': text += 'time.sleep(.2)\nprint("MemAvailable: 5000000 kB;SwapFree: 99 kB",flush=True)\n'
    if case != 'EOF': text += 'time.sleep(10)\n'
    script.write_text(text)
    monkeypatch.setattr(low,'SSH',[sys.executable,str(script)])
    supervisor = low.Supervisor(tmp_path,time.time()+60)
    try:
        with pytest.raises(ValueError):
            supervisor.begin()
            if case == 'stale':
                supervisor.last = {host:time.monotonic()-13 for host in supervisor.last}
            supervisor.run('owned',[sys.executable,'-c','import time; time.sleep(2)'])
    finally: supervisor.close()
    assert all(p.poll() is not None for p in supervisor.monitors)


def test_owned_command_timeout_and_failure_stop(tmp_path: Path) -> None:
    supervisor = low.Supervisor(tmp_path,time.time()+60)
    try:
        with pytest.raises(ValueError,match='timeout'):
            supervisor.run('timeout',[sys.executable,'-c','import time; time.sleep(5)'],timeout=.05)
        with pytest.raises(ValueError,match='failed rc'):
            supervisor.run('failed',[sys.executable,'-c','raise SystemExit(3)'])
    finally: supervisor.close()


def test_invalid_cli_and_conflicting_coordinator_print_restore(tmp_path: Path) -> None:
    for args,env in [(['--bad-option'],os.environ),(['--execute'],{**os.environ,
        'WINDOW_COORDINATOR':'Deepseek/coordinator','D2_COORDINATOR':'other'})]:
        result = subprocess.run([sys.executable,'-m','offload.launch.windows.up2_low',*args],
            text=True,capture_output=True,env=env,check=False)
        assert result.returncode != 0
        for step in ('start','check','close'): assert f'W8/restore.sh {step}' in result.stdout


@pytest.mark.parametrize('screen',[False,True])
@pytest.mark.parametrize('outcome',['pass','miss','foreign'])
def test_needle_driver_exact_recall_and_foreign_gate_with_http_fake(tmp_path: Path,monkeypatch: pytest.MonkeyPatch,outcome: str,screen: bool) -> None:
    from offload.launch.windows import up2_request as request
    from offload.kvpack import needle as legacy
    import urllib.request
    codes = {name:str(100000+i) for i,name in enumerate(('Abbeville','Beaufort','Calhoun','Darlington','Edgefield','Fairfield'))}
    class Builder:
        def __init__(self,*args: object): pass
        def build(self,ctx: int,n: int,seed: int) -> tuple[str,dict[str,str]]: return 'synthetic public prompt',codes
    monkeypatch.setattr(legacy,'Builder',Builder)
    def open_fake(req: str | urllib.request.Request,timeout: float = 0) -> FakeResponse:
        url = req if isinstance(req,str) else req.full_url
        if url.endswith('/metrics'): return FakeResponse(b'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        if url.endswith('/v1/models'): return FakeResponse(b'{"data":[{"id":"target","max_model_len":327680}]}')
        selected = list(codes.items())[:5 if outcome == 'miss' else 6]
        text = '\n'.join(f'{key}: {value}' for key,value in selected)
        if outcome == 'foreign': text += '\n999999'
        return FakeResponse(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':text}}],
            'usage':{'prompt_tokens':150000,'prompt_tokens_details':{'cached_tokens':0}}}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',open_fake)
    output,ledger = tmp_path/'needle.json',tmp_path/'ledger.jsonl'
    if outcome == 'pass':
        request.needle(150000,'salt',output,ledger,time.time()+60,tmp_path,screen=screen)
        assert ck.read(output)['recall'] == 6
        assert ck.read(output).get('screen',False) is screen
        if screen: assert ck.read(output)['adoptable'] is False
        assert '100000' not in output.read_text()
    else:
        with pytest.raises(ValueError,match='needle'): request.needle(150000,'salt',output,ledger,time.time()+60,tmp_path,screen=screen)
        assert not output.exists()
    events = [record(cast(Json,json.loads(x))) for x in ledger.read_text().splitlines()]
    assert ck.accounting(events)['completed'] == int(outcome == 'pass')


@pytest.mark.parametrize('arrives',[False,True])
def test_native_checkpoint_fail_closed_or_bounded_wait(tmp_path: Path,arrives: bool) -> None:
    native_root = tmp_path/'native'; native_root.mkdir()
    arm_root = native_root/'diag-off'; arm_root.mkdir()
    backend = StartFake(tmp_path,'')
    backend.native_root,backend.manifest,backend.ids = native_root,'manifest',['a','b']
    backend.native_wait,backend.tail_q = .5 if arrives else 0,{}
    backend.s = low.Supervisor(tmp_path,time.time()+60)
    data = packet(native_root)
    def publish() -> None:
        (arm_root/'q.json').write_text(json.dumps(data))
    timer = threading.Timer(.05,publish) if arrives else None
    if timer: timer.start()
    try:
        if arrives:
            assert backend.native('diag-off',complete=False) == 4092
            assert backend.tail_q['diag-off'] == 4092
        else:
            with pytest.raises(ValueError,match='NOT RUN/HOLD'): backend.native('diag-off',complete=False)
            assert (tmp_path/'diag-off-native-NOT-RUN.json').is_file()
        assert (tmp_path/'diag-off-await-q.json').is_file()
    finally:
        if timer: timer.join(timeout=1)
        backend.s.close()


class ScreenFake(Fake):
    """Exercise Live.native's missing-evidence path inside the complete fake sequence."""
    def __init__(self, out: Path, *, fail: str = ''):
        super().__init__(out,fail=fail)
        self.native_backend = StartFake(out,'')
        live = self.native_backend
        live.screen = True
        live.native_root,live.manifest = out/'absent-native','manifest'
        live.ids = ['a','b']
        live.native_wait,live.tail_q,live.last_stats = 0,{},stats()
        live.s = low.Supervisor(out,time.time()+60)

    @property
    def native_missing(self) -> bool:
        return self.native_backend.native_missing

    def native(self, arm: str, *, complete: bool) -> int:
        return self.native_backend.native(arm,complete=complete)


def test_screen_missing_native_continues_all_cells_and_labels(tmp_path: Path) -> None:
    fake = ScreenFake(tmp_path)
    checkpoints: list[dict[str,Json]] = []
    try:
        result = low.sequence(fake,'all',checkpoints.append,screen=True)
    finally:
        fake.native_backend.s.close()
    assert fake.starts == low.ARMS and result['completed_cells'] == 18
    assert result['diagnostics'] == 'PASS-SCREEN' and result['native'] == 'NOT RUN'
    assert result['screen'] is True and result['adoptable'] is False
    assert len(checkpoints) == 19 and all(c['screen'] is True and c['adoptable'] is False for c in checkpoints)
    for arm in ('diag-off','diag-on'):
        missing = ck.read(tmp_path/f'{arm}-native-NOT-RUN.json')
        assert missing['checks'] == {key:'NOT RUN' for key in ck.NATIVE_CHECKS}
        tails = ck.read(tmp_path/f'{arm}-tails.json')
        assert tails['classification'] == 'ESTIMATED' and tails['q'] == 4092
        assert tails['tails_verified'] is False and 'CPU' in str(tails['source'])
    assert all(ck.read(p)['screen'] is True and ck.read(p)['adoptable'] is False for p in tmp_path.glob('*.json'))
    assert ck.accounting(fake.events)['completed'] == 49


@pytest.mark.parametrize('broken',['schema','check','hash','q','tail'])
def test_screen_supplied_bad_packet_still_stops(tmp_path: Path,broken: str) -> None:
    fake = ScreenFake(tmp_path)
    live = fake.native_backend
    root = live.native_root; (root/'diag-off').mkdir(parents=True)
    data = packet(root)
    if broken == 'schema': data['schema'] = 'fake'
    if broken == 'check': record(data['checks'])[ck.NATIVE_CHECKS[0]] = 'NOT RUN'
    if broken == 'hash': (root/'native-evidence.txt').write_text('tampered')
    if broken == 'q': data['q'] = 128
    if broken == 'tail': data['tail_sizes'] = [1,127,128,129]
    (root/'diag-off'/'qualification.json').write_text(json.dumps(data))
    live.tail_q['diag-off'] = 4092
    try:
        with pytest.raises(ValueError): live.native('diag-off',complete=True)
    finally: live.s.close()
    assert not (tmp_path/'diag-off-native-complete.json').exists()


def test_screen_supplied_measured_q_used_without_complete_packet(tmp_path: Path) -> None:
    fake = ScreenFake(tmp_path); live = fake.native_backend
    root = live.native_root; (root/'diag-off').mkdir(parents=True)
    data = packet(root); data['q'] = 4000
    (root/'diag-off'/'q.json').write_text(json.dumps(data))
    try:
        assert live.native('diag-off',complete=False) == 4000
        assert live.native('diag-off',complete=True) == 4000
    finally: live.s.close()
    tails = ck.read(tmp_path/'diag-off-tails.json')
    assert tails['q'] == 4000 and tails['classification'] == 'MEASURED'
    assert tails['tail_sizes'] == [144001,144127,144128,144129]
    assert tails['tails_verified'] is False and str(tails['source']).endswith('/q.json')


def screen_qualification(root: Path) -> None:
    for arm in ('diag-off','diag-on'):
        low.save(root/f'{arm}-identity.json',{'ranks':[{'id':'a'},{'id':'b'}]},screen=True)
        low.save(root/f'{arm}-native-NOT-RUN.json',{'status':'NOT RUN',
            'checks':{key:'NOT RUN' for key in ck.NATIVE_CHECKS}},screen=True)
    low.save(root/'qualification.json',{'diagnostics':'PASS-SCREEN','native':'NOT RUN',
        'manifest_sha256':ck.digest(low.ROOT/'offload/upstream_adopt/MANIFEST.json'),
        'config_sha256':ck.digest(low.ROOT/ck.CONFIG),'worker_source':ck.WORKER,
        'experiment_commit':'a'*40,'native_evidence_root':str(root/'absent'),
        'artifact_sha256':{p.name:ck.digest(p) for p in root.glob('*.json')}},screen=True)


def test_pass_screen_refused_without_screen_and_bound_artifacts_rechecked(tmp_path: Path) -> None:
    screen_qualification(tmp_path)
    with pytest.raises(ValueError,match='prior diagnostics'): low.validate_qualification(tmp_path,'a'*40)
    low.validate_qualification(tmp_path,'a'*40,screen=True)
    with pytest.raises(ValueError,match='prior diagnostics'): low.validate_qualification(tmp_path,'b'*40,screen=True)
    (tmp_path/'diag-on-native-NOT-RUN.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'): low.validate_qualification(tmp_path,'a'*40,screen=True)


def test_screen_prior_supplied_invalid_receipt_cannot_hide_behind_not_run(tmp_path: Path) -> None:
    screen_qualification(tmp_path)
    (tmp_path/'diag-off-native-complete.json').write_text('{"schema":"bad"}')
    with pytest.raises(ValueError,match='native qualification'): low.validate_qualification(tmp_path,'a'*40,screen=True)


def test_screen_mode_is_explicit_and_measured_checkpoints_labelled(tmp_path: Path) -> None:
    assert 'SCREEN' not in low.plan('diagnostics')
    assert 'SCREEN ONLY' in low.plan('diagnostics',screen=True)
    fake = Fake(tmp_path)
    checkpoints: list[dict[str,Json]] = []
    result = low.sequence(fake,'measured',checkpoints.append,screen=True)
    assert result['screen'] is True and result['adoptable'] is False
    assert all(c['screen'] is True and c['adoptable'] is False for c in checkpoints)


@pytest.mark.parametrize('fail',['warm32','near-cap','num512-0','needle','clean','diag-on','cold150000-1'])
def test_screen_existing_sequence_failures_still_stop(tmp_path: Path,fail: str) -> None:
    fake = ScreenFake(tmp_path,fail=fail)
    try:
        with pytest.raises(ValueError): low.sequence(fake,'all',lambda _: None,screen=True)
    finally: fake.native_backend.s.close()


@pytest.mark.parametrize('screen',[False,True])
def test_fully_native_qualification_still_validated(tmp_path: Path,screen: bool) -> None:
    root = tmp_path/'native'; root.mkdir()
    for arm in ('diag-off','diag-on'):
        data = packet(root); data['arm'] = arm
        data['manifest_sha256'] = ck.digest(low.ROOT/'offload/upstream_adopt/MANIFEST.json')
        low.save(tmp_path/f'{arm}-identity.json',{'ranks':[{'id':'a'},{'id':'b'}]},screen=screen)
        for stage in ('q','complete'):
            low.save(tmp_path/f'{arm}-native-{stage}.json',data,screen=screen)
    low.save(tmp_path/'qualification.json',{'diagnostics':'PASS-SCREEN' if screen else 'PASS',
        'native':'PASS','manifest_sha256':ck.digest(low.ROOT/'offload/upstream_adopt/MANIFEST.json'),
        'config_sha256':ck.digest(low.ROOT/ck.CONFIG),'worker_source':ck.WORKER,
        'experiment_commit':'a'*40,'native_evidence_root':str(root),
        'artifact_sha256':{p.name:ck.digest(p) for p in tmp_path.glob('*.json')}},screen=screen)
    low.validate_qualification(tmp_path,'a'*40,screen=screen)
    (root/'native-evidence.txt').write_text('tampered')
    with pytest.raises(ValueError,match='hash'): low.validate_qualification(tmp_path,'a'*40,screen=screen)


@pytest.mark.parametrize('contexts', [(300000,), (150000,), (150000,300000)])
def test_measured_context_subset(tmp_path: Path, contexts: tuple[int, ...]) -> None:
    fake = Fake(tmp_path)
    result = low.sequence(fake,'measured',lambda _: None,screen=True,contexts=contexts)
    assert fake.starts == low.ARMS[2:]
    assert {p[2] for p in fake.probes if p[1].startswith('cold')} == set(contexts)
    assert result['completed_cells'] == result['expected_cells'] == 9*len(contexts)
    assert [p['context'] for p in ck.rows(result['points'])] == list(contexts)
    assert record(result['stage_b'])['status'] == ('FAIL' if 300000 in contexts else 'INCOMPLETE')
    assert result['adoptable'] is False
    assert len(fake.probes) == 3 + 9*len(contexts)
    assert sum(p[2] for p in fake.probes) == 96000 + 9*sum(contexts)
    assert sum(p[3] for p in fake.probes) == 96 + 13500*len(contexts)


@pytest.mark.parametrize('contexts', [(), (512,), (300000,300000)])
def test_bad_measured_contexts_refuse(contexts: tuple[int, ...], tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    with pytest.raises(ValueError,match='contexts'):
        low.sequence(fake,'measured',lambda _: None,contexts=contexts)
    assert not fake.starts


def test_summary_refuses_spliced_arms_and_contexts(tmp_path: Path) -> None:
    row = Fake(tmp_path).request('off-a','cold150000-0',150000,1500,cold=True)
    other = {**row,'window_id':str(tmp_path/'different-window')}
    for cells in ({'off-a':[row],'on':[other]},
                  {'off-a':[row,{**other,'target_ctx':300000}]}):
        with pytest.raises(ValueError,match='window identity'): ck.summarize(cells)
    with pytest.raises(ValueError,match='contexts'):
        ck.summarize({'off-a':[row]},(300000,))
    with pytest.raises(ValueError,match='duplicate'):
        ck.summarize({'off-a':[row,row]})
    with pytest.raises(ValueError,match='window identity'):
        ck.summarize({'off-a':[{k:v for k,v in row.items() if k != 'window_id'}]})


def test_stage_b_requires_three_usable_300k_pairs(tmp_path: Path) -> None:
    cells: dict[str,list[dict[str,Json]]] = {}
    for arm in ('off-a','on','off-b'):
        cells[arm] = [Fake(tmp_path).request(arm,f'cold300000-{rep}',300000,1500,cold=True,rep=rep)
                      for rep in range(3)]
        for row in cells[arm]: row['ttft_s'] = 80 if arm == 'on' else 100
    result = ck.summarize(cells,(300000,))
    assert record(result['stage_b'])['status'] == 'PASS'
    for arm, value in (('off-a',90),('on',95),('off-b',110)):
        for row in cells[arm]: row['ttft_s'] = value
    stage = record(ck.summarize(cells,(300000,))['stage_b'])
    assert stage['status'] == 'FAIL' and cast(float,stage['gain']) > 0
    cells['off-b'][2]['ttft_s'] = None
    assert record(ck.summarize(cells,(300000,))['stage_b'])['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('label', ['submission', 'needle submission'])
def test_child_deadline_reason_survives_supervision(tmp_path: Path, label: str) -> None:
    supervisor = low.Supervisor(tmp_path,time.time()+30)
    try:
        code = f'raise ValueError("STOP: {label} deadline 1234.000000 reached")'
        with pytest.raises(ValueError,match=f'owned: {label} deadline 1234.000000 reached'):
            supervisor.run('owned',[sys.executable,'-c',code])
        assert supervisor.active is None
    finally: supervisor.close()


def test_hard_deadline_names_bound(tmp_path: Path) -> None:
    supervisor = low.Supervisor(tmp_path,1234)
    try:
        with pytest.raises(ValueError,match='90-minute hard deadline 1234.000000 reached'):
            supervisor.run('unsent',[sys.executable,'-c','raise AssertionError("never started")'])
        assert not (tmp_path/'unsent.log').exists()
    finally: supervisor.close()


def test_unrelated_child_failure_is_not_labelled_deadline(tmp_path: Path) -> None:
    supervisor = low.Supervisor(tmp_path,time.time()+30)
    try:
        with pytest.raises(ValueError,match='owned failed rc1'):
            supervisor.run('owned',[sys.executable,'-c','raise ValueError("unrelated")'])
    finally: supervisor.close()


def test_request_driver_deadline_refuses_before_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.launch.windows import up2_request as request
    import urllib.request
    monkeypatch.setattr(request,'idle',lambda _: None)
    monkeypatch.setattr(request,'corpus',lambda _: ('synthetic public corpus',{}))
    monkeypatch.setattr(request,'exact_prompt',lambda _materials,_target,_count: 'synthetic')
    calls: list[str] = []
    def open_fake(url: str, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        assert url.endswith('/v1/models')
        return FakeResponse(json.dumps({'data':[{'id':'target','root':'/srv/dsv41/legal-v2a',
                                                  'max_model_len':327680}]}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',open_fake)
    with pytest.raises(ValueError,match='submission deadline 1234.000000 reached'):
        request.execute(512,8,'salt',tmp_path/'result.json',tmp_path/'ledger.jsonl',1234,tmp_path)
    assert len(calls) == 1 and not (tmp_path/'ledger.jsonl').exists()


@pytest.mark.parametrize('args,expected,output', [
    (['--phase','measured','--contexts','300000'],0,'cold exact300000 x3'),
    (['--phase','diagnostics','--contexts','300000'],1,'--contexts requires --phase measured'),
    (['--phase','measured','--contexts','300000','300000'],1,'unique nonempty subset'),
])
def test_context_cli_dry_validation(args: list[str], expected: int, output: str,
                                    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys,'argv',['up2_low','--dry-run',*args])
    assert low.main() == expected
    captured = capsys.readouterr()
    assert output in captured.out+captured.err
    assert 'Coordinator restoration commands' in captured.out


class ShortTextFake(Fake):
    """W13: the ON arm's 512-token greedy text differs from OFF (first-token KL 0); optionally OFF A/A differs too (r2)."""
    def __init__(self, out: Path, *, aa_differs: bool = False):
        super().__init__(out)
        self.aa_differs = aa_differs

    def request(self, arm: str, name: str, target: int, budget: int, *, numeric: bool = False,
                force_length: bool = False, rep: int = 0, cold: bool = False) -> dict[str,Json]:
        row = super().request(arm,name,target,budget,numeric=numeric,force_length=force_length,rep=rep,cold=cold)
        if name.startswith('num512') and (arm == 'diag-on' or (self.aa_differs and rep == 1)):
            row['text_sha256'] = f'differs-{arm}-{rep}'
        return row


def test_w13_short_text_on_vs_off_and_aa_are_records(tmp_path: Path) -> None:
    for name,aa_differs in (('on-differs',False),('aa-differs',True)):
        fake = ShortTextFake(tmp_path/name,aa_differs=aa_differs); (tmp_path/name).mkdir()
        checkpoints: list[dict[str,Json]] = []
        result = low.sequence(fake,'all',checkpoints.append)
        numerical = cast(list[dict[str,Json]],[c for c in checkpoints if 'numerical' in c][0]['numerical'])
        short = [c for c in numerical if c['context'] == 512][0]
        assert short['text_on_equals_off_INFORMATIONAL'] is False
        assert short['text_aa_equal_INFORMATIONAL'] is (not aa_differs)
        assert all(c['text_aa_equal_INFORMATIONAL'] is True for c in numerical if c['context'] != 512)
        assert result.get('diagnostics') in ('PASS','PASS-SCREEN')


def test_w13_numerical_floor_stays_hard_when_text_is_record(tmp_path: Path) -> None:
    class FloorFake(ShortTextFake):
        def request(self, arm: str, name: str, target: int, budget: int, *, numeric: bool = False,
                    force_length: bool = False, rep: int = 0, cold: bool = False) -> dict[str,Json]:
            row = super().request(arm,name,target,budget,numeric=numeric,force_length=force_length,rep=rep,cold=cold)
            if name.startswith('num512') and arm == 'diag-on':
                probabilities = cast(dict[str,Json],row['probabilities'])
                first,second = sorted(probabilities,key=lambda k: -cast(float,probabilities[k]))[:2]
                probabilities[first],probabilities[second] = probabilities[second],probabilities[first]
            return row
    fake = FloorFake(tmp_path/'floor',aa_differs=True); (tmp_path/'floor').mkdir()
    with pytest.raises(ValueError, match='numerical floor exceeded'):
        low.sequence(fake,'all',lambda _: None)
