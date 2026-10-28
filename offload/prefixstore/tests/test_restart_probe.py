from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from restart_probe import ask,bucket_kl,compare,digest


class ProbeTests(unittest.TestCase):
    def first(self,p=.999):
        return {'token':'A','bytes':[65], 'top_logprobs':[
            {'token':'A','bytes':[65],'logprob':math.log(p)},
            {'token':'B','bytes':[66],'logprob':math.log(1-p)}]}

    def fixture(self):
        packet={'model':'test','cases':[{'prompt':'fake public fixture', 'salt':'s', 'codes':{'Abbeville':'123456'}}]}
        def run(cached):
            return {'packet_sha256':digest(packet),'results':[{'case_sha256':digest(packet['cases'][0]),
                'cached_tokens':cached,'text':'Abbeville: 123456','ttft_s':1.0,'first':self.first()}]}
        return packet,run(0),run(384),run(0)

    def test_honest_then_foreign_or_cached_control_refused(self):
        packet,cold,reload,fresh=self.fixture()
        self.assertTrue(compare(packet,cold,reload,fresh,384)[0]['pass'])
        fresh['results'][0]['cached_tokens']=384
        self.assertFalse(compare(packet,cold,reload,fresh,384)[0]['pass'])
        fresh['results'][0]['cached_tokens']=0
        reload['results'][0]['text']='Abbeville: 654321'
        self.assertFalse(compare(packet,cold,reload,fresh,384)[0]['pass'])

    def test_low_mass_or_missing_reload_refused(self):
        packet,cold,reload,fresh=self.fixture()
        reload['results'][0]['cached_tokens']=0
        self.assertFalse(compare(packet,cold,reload,fresh,384)[0]['pass'])
        reload['results'][0]['cached_tokens']=384
        reload['results'][0]['first']=self.first(.1)
        self.assertFalse(compare(packet,cold,reload,fresh,384)[0]['pass'])
        fresh['packet_sha256']='bad'
        with self.assertRaises(ValueError): compare(packet,cold,reload,fresh,384)

    def test_kl_known_distributions(self):
        kl,mass=bucket_kl(self.first(.9),self.first(.8))
        self.assertAlmostEqual(kl,.9*math.log(.9/.8)+.1*math.log(.1/.2))
        self.assertAlmostEqual(mass,1)

    def test_stream_usage_and_ttft(self):
        events=[{'choices':[{'delta':{'role':'assistant'}}]},
                {'choices':[{'delta':{'content':'Abbeville: 123456'},'logprobs':{'content':[self.first()]}}]},
                {'choices':[],'usage':{'prompt_tokens':24000,'prompt_tokens_details':{'cached_tokens':384}}}]
        stream=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n'
        with patch('urllib.request.urlopen',return_value=io.BytesIO(stream)):
            got=ask('http://unused','model',{'prompt':'fake','salt':'a'*64},False,1)
        self.assertEqual(got['cached_tokens'],384)
        self.assertGreaterEqual(got['ttft_s'],0)
        self.assertEqual(got['text'],'Abbeville: 123456')
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'data: [DONE]\n')):
            with self.assertRaises(ValueError): ask('http://unused','model',{'prompt':'fake','salt':'a'*64},False,1)

    def test_matter_tags(self):
        # MC1: saved runs persist under the case salt as their synthetic namespace; fresh controls never do.
        events=[{'choices':[{'delta':{'content':'x'},'logprobs':{'content':[self.first()]}}]},
                {'choices':[],'usage':{'prompt_tokens':10,'prompt_tokens_details':{'cached_tokens':0}}}]
        stream=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n'
        def body(**kw):
            seen={}
            def fake(req,timeout):
                seen.update(json.loads(req.data)); return io.BytesIO(stream)
            with patch('urllib.request.urlopen',side_effect=fake):
                ask('http://unused','model',{'prompt':'p','salt':'b'*64},timeout=1,**kw)
            return seen
        tagged=body(fresh=False)
        self.assertEqual(tagged['kv_transfer_params'],{'matter_ns':'b'*64,'persist':True})
        self.assertEqual(tagged['cache_salt'],'b'*64)
        fresh=body(fresh=True)
        self.assertNotIn('kv_transfer_params',fresh); self.assertNotEqual(fresh['cache_salt'],'b'*64)
        self.assertEqual(len(fresh['cache_salt']),64)
        self.assertNotIn('kv_transfer_params',body(fresh=False,persist=False))
        self.assertEqual(body(fresh=False,matter_ns='c'*64)['kv_transfer_params']['matter_ns'],'c'*64)
        with self.assertRaises(ValueError): body(fresh=False,matter_ns='short')


if __name__=='__main__': unittest.main()
