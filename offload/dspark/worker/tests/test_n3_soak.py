"""Exercise the protocol runner and coordinator-only refusal without CUDA."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from worker.server import DraftCore, FakeBackend, expected_hello
from offload.dspark.native.tools.n3.soak import rounds


def test_protocol_soak_exercises_all_output_counts_and_margins():
    exp = expected_hello(bytes.fromhex('ab'*32),bytes.fromhex('cd'*32))
    # Small rows keep this a fast wire/runner regression, not a model test.
    exp.hidden, exp.row_bytes = 4,24
    def draft(win,tok,apos,N,vocab):
        return np.arange(N,dtype=np.int32)+1,np.ones(N,dtype=np.float32)*.5
    backend = FakeBackend(128,24,4,draft,129280)
    core = DraftCore(exp,backend)
    taps, tokens = np.zeros((1600,24),dtype=np.uint8),np.arange(1600)
    a = rounds(core,taps,tokens,80,123)
    b = rounds(core,taps,tokens,80,124)
    assert len(a['cases']) == 40 and all(v==2 for v in a['cases'].values())
    assert a['digest']==b['digest']
    assert a['stats']['identity_mismatch']==2 and a['stats']['config_mismatch']==1


def test_real_tcp_fake_client_completes_2000_rounds_without_cuda(monkeypatch):
    import socket
    from worker.server import TcpStubServer
    from offload.dspark.native.tools.n3.fake_client import RemoteCore
    monkeypatch.setenv('DSPARK_WORKER_ALLOW_LOOPBACK','1')
    exp = expected_hello(bytes.fromhex('ab'*32),bytes.fromhex('cd'*32))
    exp.hidden,exp.row_bytes=4,24
    def draft(win,tok,apos,N,vocab):
        return np.arange(N,dtype=np.int32)+1,np.ones(N,dtype=np.float32)*.5
    core=DraftCore(exp,FakeBackend(128,24,4,draft,129280))
    server=TcpStubServer(core,'127.0.0.1',0)
    try:
        with socket.create_connection(('127.0.0.1',server.port),timeout=5) as connection:
            connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            remote=RemoteCore(connection,'ab'*32,'cd'*32)
            remote.expected=exp
            result=rounds(remote,np.zeros((1600,24),np.uint8),np.arange(1600),2000,123)
        assert result['rounds']==2000 and len(result['cases'])==40
        assert result['stats']['steps']==2000
        assert result['stats'].get('internal_errors',0)==0
    finally:
        server.stop()
