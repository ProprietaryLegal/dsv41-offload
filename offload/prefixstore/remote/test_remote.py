#!/usr/bin/env python3
"""Synthetic CPU tests. --network runs x86host-local ofl clients only, never Spark traffic."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import logging
import secrets
import time
import unittest
from unittest.mock import patch
import zlib

from store import (Entry, GET, HEADER, MAX_CHUNK, PUT, Key, Namespace, Packet, PrefixStore,
                   Refused, digest, pack, request, response, unpack)

SECRET = bytes(range(32))
NS = Namespace(digest(b'synthetic-model'), digest(b'layout-pack3-replicated-v1'), digest(b'fake-matter'), 0)


def key(data: bytes, label: bytes = b'prefix') -> Key:
    return Key(NS, 0, digest(label), digest(data), len(data))


class StoreTests(unittest.TestCase):
    def test_roundtrip_immutable_and_crc(self) -> None:
        store = PrefixStore(frozenset((NS,)))
        data = bytes(range(256)) * 128
        k = key(data)
        store.put(k, data)
        store.put(k, data)
        self.assertEqual(store.used, len(data))
        self.assertEqual(store.get(k), data)
        with self.assertRaisesRegex(Refused, 'hash/size'):
            store.put(k, b'x' * len(data))
        store._entries[k] = Entry(b'x' * len(data), zlib.crc32(data))
        with self.assertRaisesRegex(Refused, 'corruption'):
            store.get(k)
        self.assertIsNone(store.get(k))
        self.assertEqual(store.corruptions, 1)
        self.assertEqual(store.used, 0)

    def test_rank_model_layout_scope_isolation(self) -> None:
        store = PrefixStore(frozenset((NS,)))
        data = b'rank-zero bytes'
        k = key(data)
        store.put(k, data)
        for ns in (replace(NS, rank=1), replace(NS, model=digest(b'wrong-model')),
                   replace(NS, layout=digest(b'wrong-layout')), replace(NS, scope=digest(b'other-matter')),
                   replace(NS, world_size=3)):
            with self.assertRaisesRegex(Refused, 'namespace'):
                store.get(replace(k, namespace=ns))
        self.assertEqual(store.get(k), data)

    def test_lru_bytes_and_entry_count(self) -> None:
        store = PrefixStore(frozenset((NS,)), capacity=8, max_entries=2)
        a, b, c = key(b'aaaa'), key(b'bbbb'), key(b'cccc')
        store.put(a, b'aaaa')
        store.put(b, b'bbbb')
        self.assertEqual(store.get(a), b'aaaa')
        store.put(c, b'cccc')
        self.assertIsNone(store.get(b))
        self.assertEqual(store.used, 8)
        self.assertEqual(store.evictions, 1)
        small = PrefixStore(frozenset((NS,)), capacity=8, max_entries=1)
        small.put(key(b'a'), b'a')
        small.put(key(b'b'), b'b')
        self.assertEqual(small.used, 1)
        self.assertEqual(small.evictions, 1)

    def test_wire_auth_bounds_and_crc(self) -> None:
        nonce, encoded = request(SECRET, PUT, key(b'abc'), b'abc')
        packet = unpack(SECRET, b'RP1Q', encoded)
        self.assertEqual(packet.nonce, nonce)
        for malformed in (encoded[:-1], encoded + b'x', encoded[:-33] + b'x' + encoded[-32:]):
            with self.assertRaises(Refused):
                unpack(SECRET, b'RP1Q', malformed)
        with self.assertRaisesRegex(Refused, 'authentication'):
            unpack(b'x' * 32, b'RP1Q', encoded)
        # Re-sign a bad CRC to exercise corruption independently of authentication.
        import hmac
        damaged = bytearray(encoded[:-32])
        damaged[HEADER.size - 1] ^= 1
        with self.assertRaisesRegex(Refused, 'CRC'):
            unpack(SECRET, b'RP1Q', bytes(damaged) + hmac.digest(SECRET, damaged, 'sha256'))
        with self.assertRaises(Refused):
            pack(SECRET, b'RP1Q', Packet(GET, nonce, b'x' * 2049, b''))
        with self.assertRaises(Refused):
            key(b'x' * (MAX_CHUNK + 1))

    def test_request_validation_and_replay_response(self) -> None:
        store = PrefixStore(frozenset((NS,)))
        k = key(b'abc')
        nonce, encoded = request(SECRET, PUT, k, b'abc')
        reply = store.handle(SECRET, encoded)
        self.assertEqual(response(SECRET, reply, PUT, nonce, k), ('stored', b''))
        with self.assertRaisesRegex(Refused, 'stale'):
            response(SECRET, reply, PUT, secrets.token_bytes(16), k)
        with self.assertRaisesRegex(Refused, 'stale'):
            response(SECRET, reply, PUT, nonce, key(b'xyz'))
        for bad in (b'{}', b'{"namespace": null}', b'\xff'):
            with self.assertRaises(Refused):
                Key.decode(bad)
        nonce, bad_get = request(SECRET, GET, k, b'illegal')
        with self.assertRaisesRegex(Refused, 'operation'):
            store.handle(SECRET, bad_get)

    def test_restart_is_miss(self) -> None:
        store = PrefixStore(frozenset((NS,)))
        store.put(key(b'abc'), b'abc')
        restarted = PrefixStore(frozenset((NS,)))
        self.assertIsNone(restarted.get(key(b'abc')))

    def test_batch_total_deadline_and_resource_refusals(self) -> None:
        from ofl_backend import Load, PrefixClient, PrefixServer, memory_guard
        with self.assertRaisesRegex(Refused, '64 GiB'):
            memory_guard(64 << 30, 1)
        with self.assertRaisesRegex(Refused, 'direct-link'):
            PrefixServer('0.0.0.0', 18640, SECRET, frozenset((NS,)))
        client = PrefixClient('10.202.28.1', 18640, '10.202.28.1', SECRET)
        def delayed_load(k: Key, *, budget: float | None = None) -> Load:
            time.sleep(0.015)
            return Load(b'abc', 'hit')
        with patch.object(client, 'load', delayed_load):
            loaded = client.load_many([key(b'abc')] * 3, total_deadline_s=0.02)
        self.assertIsNone(loaded.data, 'partial prefix survived total deadline')
        self.assertIn('deadline', loaded.reason)


class NetworkTests(unittest.TestCase):
    ip = '10.202.28.1'
    backend = 1
    port = 18640

    def setUp(self) -> None:
        from ofl_backend import PrefixClient, PrefixServer
        self.server = PrefixServer(self.ip, self.port, SECRET, frozenset((NS,)),
                                   capacity=4 << 20, max_entries=64, rdma=self.backend == 2)
        self.server.start()
        self.client = PrefixClient(self.ip, self.port, self.ip, SECRET, backend=self.backend,
                                   timeout_s=0.5, cooldown_s=0)

    def tearDown(self) -> None:
        self.client.close()
        self.server.stop()

    def test_roundtrip_batch_partial_miss_and_no_stale(self) -> None:
        data = bytes(range(256)) * 4096
        k = key(data)
        self.assertTrue(self.client.put(k, data))
        self.assertEqual(self.client.load(k).data, data)
        self.assertEqual(self.client.load_many([k, k]).data, data + data)
        self.assertIsNone(self.client.load_many([k, key(b'absent')]).data)
        self.assertIsNone(self.client.load(key(b'absent')).data)
        self.assertEqual(self.client.load(k).data, data)
        self.assertIsNone(self.client.load_many([k] * 17).data)

    def test_bad_namespace_auth_and_corruption_fall_back(self) -> None:
        from ofl_backend import PrefixClient
        data = b'never stale'
        k = key(data)
        self.assertTrue(self.client.put(k, data))
        self.assertIsNone(self.client.load(replace(k, namespace=replace(NS, rank=1))).data)
        bad = PrefixClient(self.ip, self.port, self.ip, b'x' * 32,
                           backend=self.backend, timeout_s=0.5)
        try:
            self.assertIsNone(bad.load(k).data)
            self.assertIn('OflError', bad.last_error)
        finally:
            bad.close()
        self.server.store._entries[k] = Entry(b'x' * len(data), zlib.crc32(data))
        self.assertIsNone(self.client.load(k).data)
        self.assertEqual(self.server.store.corruptions, 1)
        self.assertGreaterEqual(self.server.refusals, 3)

    def test_multiple_fake_clients(self) -> None:
        from ofl_backend import PrefixClient
        def work(index: int) -> bool:
            client = PrefixClient(self.ip, self.port, self.ip, SECRET,
                                  backend=self.backend, timeout_s=0.5)
            data = bytes((index,)) * (128 << 10)
            k = key(data, str(index).encode())
            try:
                return client.put(k, data) and client.load(k).data == data
            finally:
                client.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(work, range(4))), [True] * 4)

    def test_timeout_then_compute_and_late_reply(self) -> None:
        original = self.server.store.handle
        def delayed(secret: bytes, raw: bytes) -> bytes:
            time.sleep(0.1)
            return original(secret, raw)
        with patch.object(self.server.store, 'handle', delayed):
            self.client.timeout_s = 0.02
            start = time.monotonic()
            result = self.client.load(key(b'absent'))
            elapsed = time.monotonic() - start
            self.assertIsNone(result.data)
            self.assertIn('OflError', result.reason)
            self.assertLess(elapsed, 0.2)
            snapshot = bytes(self.client._buffer)
            time.sleep(0.15)
            self.assertEqual(bytes(self.client._buffer), snapshot, 'late reply wrote into client buffer')
        self.client.timeout_s = 0.5
        self.assertIsNone(self.client.load(key(b'absent')).data)
        print(json.dumps({'event': 'timeout', 'backend': self.backend, 'elapsed_s': elapsed}))

    def test_server_link_down_falls_back(self) -> None:
        self.assertTrue(self.client.put(key(b'cached'), b'cached'))
        self.server.stop()
        start = time.monotonic()
        self.assertIsNone(self.client.load(key(b'cached')).data)
        self.assertLess(time.monotonic() - start, 0.6)
        # Fresh connect to closed listener must also return compute, never cached buffer.
        self.assertIsNone(self.client.load(key(b'cached')).data)

    def test_synthetic_throughput(self) -> None:
        data = bytes(range(256)) * 4096
        k = key(data)
        self.assertTrue(self.client.put(k, data))
        times: list[float] = []
        for _ in range(16):
            start = time.monotonic()
            self.assertEqual(self.client.load(k).data, data)
            times.append(time.monotonic() - start)
        times.sort()
        print(json.dumps({'event': 'x86host-local-synthetic-get', 'backend': self.backend,
                          'bytes': len(data), 'reps': len(times),
                          'median_ms': times[len(times) // 2] * 1000,
                          'MiB_s': 16 / sum(times), 'max_ms': max(times) * 1000}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--ip', choices=('10.202.28.1', '10.202.29.1'), default='10.202.28.1')
    parser.add_argument('--backend', choices=('tcp', 'rdma'), default='tcp')
    parser.add_argument('--port', type=int, default=18640)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StoreTests)
    if args.network:
        NetworkTests.ip, NetworkTests.backend, NetworkTests.port = args.ip, {'tcp': 1, 'rdma': 2}[args.backend], args.port
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NetworkTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
