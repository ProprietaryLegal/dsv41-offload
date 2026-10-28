#!/usr/bin/env python3
"""Tiny direct-IP TCP echo/stream probe. Run via SSH -c, no Spark file writes."""
from __future__ import annotations
import argparse
import json
import socket
import statistics
import time


def receive(sock: socket.socket, buf: bytearray) -> None:
    view = memoryview(buf)
    while view:
        n = sock.recv_into(view)
        if not n:
            raise EOFError('Peer closed before full payload')
        view = view[n:]


def probe(role: str, bind: str, peer: str, port: int, size: int, count: int, mode: str, start_at: float = 0) -> dict[str, object]:
    payload = bytes([137])*size
    buf = bytearray(size)
    listener: socket.socket | None = None
    sock = socket.socket()
    sock.settimeout(20)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY,1)
    if role == 'server':
        listener = sock
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        listener.bind((bind,port))
        listener.listen(1)
        print(json.dumps({'listening': listener.getsockname()}),flush=True)
        sock,_ = listener.accept()
        sock.settimeout(20)
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    else:
        sock.bind((bind,0))
        sock.connect((peer,port))
    samples: list[float] = []
    with sock:
        # One untimed handshake; reports application timings, excludes connect/setup.
        if role == 'client':
            if start_at: time.sleep(max(0,start_at-time.time()))
            sock.sendall(b'!')
            if sock.recv(1)!=b'!': raise ValueError('Handshake mismatch')
        else:
            if sock.recv(1)!=b'!': raise ValueError('Handshake mismatch')
            sock.sendall(b'!')
        start=time.perf_counter()
        for _ in range(count):
            t=time.perf_counter()
            if mode=='echo':
                if role=='client':
                    sock.sendall(payload); receive(sock,buf)
                else:
                    receive(sock,buf); sock.sendall(buf)
            elif (mode=='forward' and role=='client') or (mode=='reverse' and role=='server'):
                sock.sendall(payload)
            else:
                receive(sock,buf)
            samples.append((time.perf_counter()-t)*1e6)
        # Receiver checks last payload. Echo client checks its roundtrip.
        receives = mode=='echo' or (mode=='forward' and role=='server') or (mode=='reverse' and role=='client')
        if receives and buf!=payload: raise ValueError('TCP payload mismatch')
        if role=='server': sock.sendall(b'!')
        else:
            if sock.recv(1)!=b'!': raise ValueError('Final acknowledgement mismatch')
        elapsed=time.perf_counter()-start
    if listener: listener.close()
    return dict(role=role,bind=bind,peer=peer,size=size,count=count,mode=mode,seconds=elapsed,
                payload_gbps=size*count*(2 if mode=='echo' else 1)*8/elapsed/1e9,
                rtt_mean_us=statistics.mean(samples) if mode=='echo' and role=='client' else None,
                rtt_median_us=statistics.median(samples) if mode=='echo' and role=='client' else None,
                rtt_p99_us=sorted(samples)[min(len(samples)-1,int(len(samples)*.99))] if mode=='echo' and role=='client' else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['client','server'])
    p.add_argument('bind');p.add_argument('peer')
    p.add_argument('port',type=int);p.add_argument('size',type=int);p.add_argument('count',type=int)
    p.add_argument('mode',choices=['echo','forward','reverse'])
    p.add_argument('start_at',type=float,nargs='?',default=0)
    a=p.parse_args()
    print(json.dumps(probe(a.role,a.bind,a.peer,a.port,a.size,a.count,a.mode,a.start_at)),flush=True)
