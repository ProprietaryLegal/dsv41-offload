"""Offline synthetic smoke IO; it has no network/subprocess implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import cast

from .decode_dataset import SCHEMA, digest
from .decode_smoke import BASE, run
from .replay_window import CAMPAIGN, IMAGE, expected_hashes


class DrySmoke:
    synthetic = True

    def __init__(self, repo: Path, root: Path) -> None:
        self.repo, self.root = repo, root
        self.calls: list[str] = []
        self.drafts = 0
        self.completed = 0
        self.informational_sent = 3  # a live run may send fewer (best-effort); each one sent is a real request
        self.captured: dict[str, list[int]] = {}
        self.prompts = [[100+i for i in range(8)], [200+i for i in range(8)], [300+i for i in range(8)]]

    def latency_metrics(self, completed: int) -> str:
        return (f'vllm:time_to_first_token_seconds_count{{}} {completed}\n'
                f'vllm:time_to_first_token_seconds_sum{{}} {completed * .1}\n'
                f'vllm:inter_token_latency_seconds_count{{}} {completed * 2}\n'
                f'vllm:inter_token_latency_seconds_sum{{}} {completed * .08}\n')

    def metrics(self) -> str:
        return self.latency_metrics(self.completed)+ 'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n'+f'vllm:spec_decode_num_drafts_total{{}} {self.drafts}\n'

    def post(self, name: str, endpoint: str, body: dict[str, object]) -> dict[str, object]:
        i = int(name[-1]) if name.startswith('prompt-') else 2
        self.calls.append(name)
        self.drafts += 2
        self.completed += 1
        ids = [20, 21, 22]
        c: dict[str, object] = {'finish_reason': 'length', 'token_ids': ids,
            'prompt_token_ids': self.prompts[i], 'text': 'synthetic-greedy'}
        if i < 2:
            self.captured['smoke'+str(i)] = self.prompts[i] + ids
        if i == 1:
            c['logprobs'] = {'content': [{'token': f'token_id:{t}', 'logprob': -5.0,
                'top_logprobs': [{'token': f'token_id:{k}', 'logprob': -5.0} for k in range(10,30)]} for t in ids]}
        if i == 2 and endpoint == '/v1/chat/completions':
            c['message'] = {'content': c.pop('text')}
        return {'metrics': {'speculative_decoding': {'num_draft_tokens': 2}}, 'choices': [c], 'prompt_token_ids': self.prompts[i],
                'usage': {'prompt_tokens_details': {'cached_tokens': 0}}}

    def active(self, requests: list[dict[str, object]], remote: str) -> dict[str, object]:
        tagged = [self.post('prompt-'+str(i), str(r['endpoint']), cast(dict[str, object], r['body']))
                  for i, r in enumerate(requests[:2])]
        info = [{'response': self.post('informational-'+str(i), str(requests[2]['endpoint']),
                                      cast(dict[str, object], requests[2]['body'])),
                 'both_tagged_inflight_at_start': True,
                 'latency': {'ttft_s': .1, 'delivery_chunk_mean_s': .04}}
                for i in range(self.informational_sent)]
        return {'tagged': tagged, 'informational': info, 'writer_activity': [{'synthetic': True}],
                'informational_status': 'complete' if self.informational_sent == 3 else
                    'incomplete: tagged decoding ended before active-writer evidence',
                'informational_medians': {'ttft_s': .1, 'delivery_chunk_mean_s': .04},
                'timing_source': 'synthetic client streaming token-ID chunks; informational only', 'gated': False}

    def call(self, name: str, host: str, command: str, timeout: int = 60, supervised: bool = False) -> str:
        self.calls.append(name)
        if name.startswith('identity-'):
            env = ['VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode', 'VLLM_DSPARK_REMOTE_POLICY=capacity',
                   'VLLM_DSPARK_REMOTE_ASYNC=1', 'VLLM_DSPARK_REMOTE_ASYNC_PROXY=process', 'VLLM_DSV41_KV_PAIR=3',
                   'VLLM_DSV41_DROP_MTP_BUFFER=1', 'VLLM_EXL3_TF_PROMPT_TRANSIENT=1',
                   'VLLM_EXL3_TF_PROMPT_ALIAS=1', 'VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4']
            actual = {BASE+remote: digest(self.repo/'offload/dspark'/local) for local, remote in (
                ('capture/speculator_replay.py','/dspark/speculator.py'),
                ('capture/decode_capture.py','/dspark/decode_capture.py'),
                ('spark/remote_dspark/speculator.py','/remote_dspark/speculator.py'))}
            profiles = json.loads((self.repo/'offload/operations/profiles.v1.json').read_text())
            k1 = next(p for p in profiles['profiles'] if p['name'] == 'K1-327680')
            for key, value in k1['match']['env'].items():
                if value is not None and not any(e.startswith(key+'=') for e in env):
                    env.append(key+'='+value)
            for mount in k1['match']['mounts']:
                dest = mount['destination']
                if any(dest.endswith(path) for path in profiles['shared']['k1_paths']):
                    actual[dest] = digest(self.repo/'offload/kvpack/overlay'/dest.split('/dist-packages/')[-1])
            return json.dumps({'id':'a'*64, 'image':IMAGE, 'state':'running', 'env':env,
                               'mounts':[{'Destination':'/d2-decode','RW':True}]})+'\n'+json.dumps({
                                   'engine':'0.31.1.dev0+g6e517b15.dsv41gb10', 'files':expected_hashes(self.repo), 'mounted_code_sha256':actual})
        if name == 'models':
            return json.dumps({'data':[{'root':'/srv/dsv41/legal-v2a','max_model_len':327680}]})
        if name == 'flush':
            self.freeze()
            return (self.root/'remote/FLUSHED.json').read_text()
        return 'synthetic check passed\n'

    def freeze(self) -> None:
        root = self.root/'remote'
        root.mkdir()
        records: dict[str, object] = {}
        for sid, ids in self.captured.items():
            d = root/sid
            d.mkdir()
            # Synthetic exact BF16-one rows; no model, proprietary data or GPU.
            tb = struct.pack('<H', 0x3f80)*15360*len(ids)
            ib = struct.pack('<'+'i'*len(ids), *ids)
            (d/'taps.bin').write_bytes(tb)
            (d/'ids.bin').write_bytes(ib)
            (d/'CAPTURE.json').write_text(json.dumps({'schema':SCHEMA, 'taps_source':'decode_capture',
                'seq':sid, 'from':0, 'row_bytes':30720, 'capture_sha256':digest(self.repo/'offload/dspark/capture/decode_capture.py')}))
            (d/'ledger.jsonl').write_text(json.dumps({'schema':SCHEMA, 'seq':sid, 'seg':0, 'tag_r0':0,
                'prompt_length':len(self.prompts[int(sid[-1])]), 'replaced_rows':[],
                'row_sha256_taps':[hashlib.sha256(tb[j*30720:(j+1)*30720]).hexdigest() for j in range(len(ids))],
                'row_sha256_ids':[hashlib.sha256(ib[j*4:(j+1)*4]).hexdigest() for j in range(len(ids))],
                'round':1, 'first':0, 'rows':len(ids), 'dup_rows':0, 'dup_rows_differing':0, 'zero_rows':0,
                'sha256_taps':hashlib.sha256(tb).hexdigest(), 'sha256_ids':hashlib.sha256(ib).hexdigest()})+'\n')
            records[sid] = {'ledger_sha256':digest(d/'ledger.jsonl'), 'taps_bytes':len(tb), 'ids_bytes':len(ib)}
        (root/'FLUSH.json').write_text(json.dumps({'nonce':'smoke-synthetic'}))
        (root/'FLUSHED.json').write_text(json.dumps({'schema':SCHEMA, 'nonce':'smoke-synthetic',
                                                   'errors':0, 'sequences':records}))

    def copy(self, remote: str, local: Path) -> None:
        self.calls.append('copy-'+local.name)
        parts = remote.split('/d2-decode-synthetic/')[1]
        shutil.copyfile(self.root/'remote'/parts, local)

    def inputs(self) -> dict[str, object]:
        reqs = []
        for i in range(3):
            body: dict[str, object] = {'model':'synthetic', 'prompt':self.prompts[i], 'max_tokens':3,
                'temperature':0 if i == 2 else 1, 'top_p':1, 'priority':10 if i == 2 else 20,
                'return_token_ids':True, 'stream':False,
                'request_id':f'd2c.smoke{i}.0.0.{len(self.prompts[i])}' if i < 2 else 'untagged-control'}
            if i == 1:
                body.pop('prompt')
                body.update(messages=[{'role':'user','content':'synthetic'}], logprobs=True, top_logprobs=20,
                            return_tokens_as_token_ids=True)
            reqs.append({'body':body, 'prompt_ids':self.prompts[i],
                         'endpoint':'/v1/chat/completions' if i == 1 else '/v1/completions'})
        baseline = self.root/'synthetic-k1.jsonl'
        baseline.write_text(json.dumps({'text':'synthetic-greedy'})+'\n')
        proof = self.root/'synthetic-k1-request.json'
        request = dict(cast(dict[str, object], reqs[2]['body']))
        request.pop('request_id')
        proof.write_text(json.dumps({'schema':'d2.k1-greedy-reference.v2', 'timing_qualified':True,
            'warmup':{'attempt':1,'accepted':True,'output_identical':True},
            'samples':[{'attempt':i+2,'before':self.latency_metrics(i)+'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n',
                        'after':self.latency_metrics(i+1)+'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n',
                        'accepted':True,'output_identical':True,'cached_prompt_tokens':0} for i in range(5)],
            'profile':'K1-327680', 'capture_sha256':digest(baseline), 'row':0,
                                    'request':request, 'verified':True, 'cached_prompt_tokens':0,
            'latency_metrics':{'before':self.latency_metrics(0), 'after':self.latency_metrics(1)}}))
        return {'schema':'d2.decode-smoke-input.v1','requests':reqs, 'greedy_reference':{
            'path':str(baseline), 'sha256':digest(baseline), 'row':0,
            'request_receipt':str(proof), 'request_receipt_sha256':digest(proof)}}


def dry_run(repo: Path) -> None:
    scratch = CAMPAIGN/'scratch/D2-SMOKE-R2'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='decode-smoke-dry-', dir=scratch) as path:
        root = Path(path)
        out = root/'out'
        out.mkdir()
        io = DrySmoke(repo, root)
        print('SYNTHETIC / DRY-RUN: no SSH, HTTP, services or model calls')
        run(io, io.inputs(), repo, out, 'synthetic')
        assert [c for c in io.calls if c.startswith('prompt-')] == ['prompt-0','prompt-1']
        assert len([c for c in io.calls if c.startswith('control-timed-')]) == 5
