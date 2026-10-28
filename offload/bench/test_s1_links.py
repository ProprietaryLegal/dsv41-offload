"""Safety controls have consequences: fail closed and clean up all started peers."""
from __future__ import annotations
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import json
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s1_links as links


class SafetyTest(unittest.TestCase):
    def metrics(self, body: str) -> dict[str,float]:
        with patch('s1_links.urllib.request.urlopen',return_value=io.BytesIO(body.encode())):
            return links.idle()

    def test_metrics_missing_busy_negative_nan_refuse(self) -> None:
        for value in ['1','-1','NaN','inf']:
            with self.subTest(value=value),self.assertRaises(RuntimeError):
                self.metrics(f'vllm:num_requests_running{{engine="0"}} {value}\nvllm:num_requests_waiting 0\n')
        with self.assertRaises(RuntimeError): self.metrics('vllm:num_requests_running 0\n')
        self.assertEqual(self.metrics('vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting 0\n')['num_requests_running'],0)

    def test_unreachable_metrics_refuses_with_distinct_guard_error(self) -> None:
        with patch('s1_links.urllib.request.urlopen',side_effect=ConnectionRefusedError('connection refused')):
            with self.assertRaisesRegex(links.MetricsUnavailable,'ConnectionRefusedError'):
                links.idle()

    def test_one_busy_engine_refuses_even_when_sum_zero(self) -> None:
        with self.assertRaises(RuntimeError):
            self.metrics('vllm:num_requests_running{engine="0"} -1\nvllm:num_requests_running{engine="1"} 1\nvllm:num_requests_waiting 0\n')

    def test_listeners_bound_and_spark_connects_out(self) -> None:
        for link in links.LINKS.values():
            server,command,client,client_command=links.rdma_pair(link,'ib_write_bw',77824,True)
            self.assertEqual(server,'x86host')
            self.assertIn('--bind_source_ip='+link.x86host_ip,command)
            self.assertEqual(client,link.host)
            self.assertEqual(client_command[-1],link.x86host_ip)
            self.assertIn('--bind_source_ip='+link.spark_ip,client_command)
            self.assertIn('--reversed',command)

    def test_busy_after_launch_closes_all_peers_before_wait(self) -> None:
        created: list[Peer] = []
        class Process:
            def __init__(self) -> None: self.stdin=io.StringIO()
            def poll(self) -> None: return None
        class Peer:
            def __init__(self,*args: object) -> None:
                self.process=Process();created.append(self)
            def close(self) -> dict[str,object]:
                assert all(p.process.stdin.closed for p in created)
                return dict(returncode=-15)
        with patch.object(links,'Remote',Peer),patch('s1_links.time.sleep'),patch.object(links,'idle',side_effect=[{}, {}, RuntimeError('Serve busy')]):
            result=links.run_pairs([('x86host',[],'sparkworker',[])])
        self.assertFalse(result['ok'])
        self.assertIn('Serve busy',result['error'])
        self.assertEqual(len(created),2)

    def test_spark_a_without_copy_marker_starts_nothing(self) -> None:
        missing: subprocess.CompletedProcess[str]=subprocess.CompletedProcess(['ssh'],1)
        with patch('s1_links.subprocess.run',return_value=missing),patch.object(links,'Remote') as remote:
            with self.assertRaisesRegex(RuntimeError,'ENGRAM_COPY_DONE'):
                links.matrix(['a'],Path('unused'))
            remote.assert_not_called()

    def test_empty_interrupted_checkpoint_resumes_but_corruption_fails(self) -> None:
        artifacts=Path(__file__).resolve().parents[1]/'results/s1/.artifacts'
        artifacts.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts)as folder:
            root=Path(folder)
            raw=root/'example.jsonl'
            raw.write_text('')
            self.assertFalse(links.completed(root,'example',True))
            raw.write_text('{"ok":false}\n')
            self.assertFalse(links.completed(root,'example',True))
            raw.write_text('{"ok":true}\n')
            self.assertTrue(links.completed(root,'example',True))
            raw.write_text('{invalid')
            with self.assertRaises(json.JSONDecodeError):links.completed(root,'example',True)

    def test_coordinator_metrics_override_checks_selected_serve(self) -> None:
        endpoint='http://HEAD_IP:8001/metrics'
        with patch.object(links,'idle',return_value={}) as metrics:
            result=links.run_pairs([],metrics_url=endpoint)
        self.assertEqual(result['metrics_url'],endpoint)
        self.assertTrue(metrics.call_args_list)
        self.assertTrue(all(call.args==(endpoint,) for call in metrics.call_args_list))

    def test_initial_busy_starts_nothing(self) -> None:
        with patch.object(links,'idle',side_effect=RuntimeError('busy')),patch.object(links,'Remote') as remote:
            with self.assertRaises(RuntimeError):links.run_pairs([('x86host',[],'sparkworker',[])])
            remote.assert_not_called()

if __name__=='__main__':unittest.main()
