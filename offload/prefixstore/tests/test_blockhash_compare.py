"""blockhash_compare.py on synthetic dumps: pass, differ, missing, unlabeled, unstable dump, empty."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import blockhash_compare as bc


def line(ev: str, key: str | None, g: int, sub: int, sha: str, seg: str = '') -> str:
    return json.dumps({'ev': ev, 'rank': 0, 'job': 1, 'g': g, 'lb': sub, 'key': key, 'sub': sub, 'sha': sha,
                       'seg': seg, 'n': 4096})


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name: str, *lines: str) -> str:
        path = self.dir / name
        path.write_text('\n'.join(lines) + '\n')
        return str(path)

    def run_main(self, *args: str) -> tuple[int, dict[str, object]]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = bc.main(list(args))
        return code, json.loads(buf.getvalue())

    def test_restore_pass_and_failures(self) -> None:
        a = self.write('a', line('store', 'k1', 0, 0, 'x'), line('store', 'k1', 0, 1, 'y'), line('store', 'k2', 1, 0, 'z'),
                       line('load', 'k1', 0, 0, 'ignored'))
        b = self.write('b', line('load', 'k1', 0, 0, 'x'), line('load', 'k1', 0, 1, 'y'))
        code, out = self.run_main('restore', '--in-memory', a, '--restored', b)
        self.assertEqual((code, out['compared'], out['equal']), (0, 2, 2))
        for bad, field in ((line('load', 'k1', 0, 1, 'WRONG'), 'differ'), (line('load', 'k9', 0, 0, 'q'), 'missing'),
                           (line('load', None, 0, 0, 'x'), 'unlabeled')):
            code, out = self.run_main('restore', '--in-memory', a, '--restored', self.write('b2', bad))
            self.assertEqual(code, 1)
            self.assertEqual(out[field], 1)
        code, out = self.run_main('restore', '--in-memory', a, '--restored', self.write('empty', ''))
        self.assertEqual((code, out['compared']), (1, 0))

    def test_unstable_dump_refused(self) -> None:
        a = self.write('a', line('store', 'k1', 0, 0, 'x'), line('store', 'k1', 0, 0, 'other'))
        with self.assertRaises(ValueError):
            bc.load(Path(a), 'store')

    def test_ranks_report_differing_segments(self) -> None:
        seg = '0' * 16 * 4
        r0 = self.write('r0', line('store', 'k1', 0, 0, 'x', seg), line('store', 'k2', 0, 0, 'y', seg))
        r1 = self.write('r1', line('store', 'k1', 0, 0, 'x', seg),
                        line('store', 'k2', 0, 0, 'Y', '0' * 16 + '1' * 16 + '0' * 32))
        code, out = self.run_main('ranks', '--rank0', r0, '--rank1', r1)
        self.assertEqual(code, 1)
        self.assertEqual(out['differing_segments'], {'1': 1})
        r1b = self.write('r1b', line('store', 'k1', 0, 0, 'x', seg))
        code, out = self.run_main('ranks', '--rank0', r0, '--rank1', r1b)
        self.assertEqual((code, out['only_in_reference']), (1, 1))  # ranks must cover the same blocks


if __name__ == '__main__':
    unittest.main()
