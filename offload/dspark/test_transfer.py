"""Focused checks for exact extraction and fail-closed traffic protection."""
from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import copy_to_x86host as transfer
from inventory_source import direct_read


class TransferTests(unittest.TestCase):
    def test_extract_preserves_names_dtypes_shapes_and_packed_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"ignored-header"+bytes(range(128)))
            rows: list[transfer.TensorRow] = [
                {"name": "mtp.0.attn.wo_a.weight", "shape": [2, 4], "dtype": "F8_E4M3",
                 "resident_packed_bytes": 8, "source_byte_range": [14, 22]},
                {"name": "mtp.0.attn.wo_a.scale", "shape": [2], "dtype": "U8",
                 "resident_packed_bytes": 2, "source_byte_range": [100, 102]},
            ]
            destination = root / "extracted.safetensors"
            transfer.extract(source, destination, rows)
            with destination.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
                payload = stream.read()
            self.assertEqual(set(header), {row["name"] for row in rows})
            self.assertEqual(header[rows[0]["name"]]["dtype"], "F8_E4M3")
            self.assertEqual(header[rows[0]["name"]]["shape"], [2, 4])
            self.assertEqual(payload, bytes(range(8))+bytes(range(86, 88)))
            for row in rows:
                start, end = row["source_byte_range"]
                self.assertEqual(row["source_byte_range_sha256"], hashlib.sha256(source.read_bytes()[start:end]).hexdigest())

    def test_short_source_never_publishes_destination(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"short")
            destination = root / "extracted.safetensors"
            with self.assertRaises(EOFError):
                transfer.extract(source, destination, [{"name": "mtp.0.test", "shape": [12], "dtype": "U8",
                    "resident_packed_bytes": 12, "source_byte_range": [0, 12]}])
            self.assertFalse(destination.exists())

    def test_idle_guard_rejects_busy_missing_and_bad_metrics(self) -> None:
        idle = 'vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\n'
        with patch("copy_to_x86host.urllib.request.urlopen", return_value=io.BytesIO(idle.encode())):
            transfer.idle()
        for body in (idle.replace('0.0', '1.0', 1), idle.replace('waiting', 'missing'), 'invalid'):
            with self.subTest(body=body), patch("copy_to_x86host.urllib.request.urlopen", return_value=io.BytesIO(body.encode())):
                with self.assertRaises(RuntimeError):
                    transfer.idle()

    def test_existing_bad_file_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "wrong"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "refusing unverified"):
                transfer.copy("unused", path, "0"*64, 5)

    def test_bad_range_rejected_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            destination = root / "invalid.safetensors"
            with self.assertRaisesRegex(ValueError, "invalid source range"):
                transfer.extract(root/"unused", destination, [{"name": "mtp.bad", "shape": [4],
                    "dtype": "U8", "resident_packed_bytes": 4, "source_byte_range": [12, 17]}])
            self.assertFalse(destination.exists())

    def test_resume_uses_direct_aligned_read_and_exact_overlap(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = (root/"original").resolve()
            data = bytes(range(256))*32+b"tail!"
            source.write_bytes(data)
            destination = root/"resumed"
            partial = destination.with_name(destination.name+".s2-partial")
            partial.write_bytes(data[:4123])
            with patch.object(transfer, "SSH", ["bash", "-c"]), patch.object(transfer, "idle"):
                receipt = transfer.copy(str(source), destination, hashlib.sha256(data).hexdigest(), len(data), resume=True)
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(receipt["resumed_from_bytes"], 4123)
            self.assertEqual(receipt["bytes_transferred"], len(data)-4123)

    def test_bad_checkpoint_prefix_fails_complete_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = (root/"original").resolve()
            data = bytes(range(256))*32
            source.write_bytes(data)
            destination = root/"bad-resume"
            destination.with_name(destination.name+".s2-partial").write_bytes(b"corrupt")
            with patch.object(transfer, "SSH", ["bash", "-c"]), patch.object(transfer, "idle"):
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    transfer.copy(str(source), destination, hashlib.sha256(data).hexdigest(), len(data), resume=True)
            self.assertFalse(destination.exists())

    def test_complete_checkpoint_verified_without_source_traffic(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            destination = Path(directory)/"finished"
            data = b"completed before disconnect"
            destination.with_name(destination.name+".s2-partial").write_bytes(data)
            with patch.object(transfer, "idle"), patch("copy_to_x86host.subprocess.Popen") as process:
                result = transfer.copy("unused", destination, hashlib.sha256(data).hexdigest(), len(data), resume=True)
            process.assert_not_called()
            self.assertTrue(result["complete_checkpoint_verified"])
            self.assertEqual(destination.read_bytes(), data)

    def test_direct_read_alignment_and_short_read(self) -> None:
        import os
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "direct"
            data = bytes(range(256))*32
            path.write_bytes(data)
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                self.assertEqual(direct_read(fd, 4091, 50), data[4091:4141])
                with self.assertRaises(EOFError):
                    direct_read(fd, 8000, 500)
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
