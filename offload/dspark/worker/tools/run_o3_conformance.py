"""Lane O3's worker-side tests against this worker's state machine (DraftCore + FakeBackend).

O3's tests build `remote_dspark.mock_worker.WorkerCore(hello)`; this runner replaces that name with an adapter around
worker.server.DraftCore (O3's fake_draft behind FakeBackend) before pytest imports the test modules, then runs O3's
test files unchanged. Usage (needs O3's spark dir: $O3_SPARK_DIR, default the O3 worktree; OFL_PY_DIR / libofl for the
ofl test):  python -m worker.tools.run_o3_conformance [extra pytest args]
"""

import os
import sys

O3 = os.environ.get("O3_SPARK_DIR", "/opt/dsv41/repos/.worktrees/dsv41-o3/offload/dspark/spark")
TESTS = ("test_mock_worker.py", "test_round.py", "test_o1_transport.py")


def main() -> int:
    sys.path.insert(0, O3)
    sys.path.insert(0, os.path.join(O3, "tests"))
    import pytest
    import remote_dspark.mock_worker as mw
    from remote_dspark import wire

    from .. import protocol as P
    from ..server import DraftCore, FakeBackend

    class O2WorkerCore(DraftCore):
        def __init__(self, expected, *, worker_name="o2-conformance", draft_fn=mw.fake_draft,
                     capabilities=wire.CAP_TAPS_BF16 | wire.CAP_HAS_MARGIN):
            exp = P.Hello.decode(expected.pack())
            super().__init__(exp, FakeBackend(exp.window, exp.row_bytes, exp.max_slots, draft_fn, exp.vocab_size),
                             worker_name=worker_name, capabilities=capabilities)
            O2WorkerCore.instances += 1

    O2WorkerCore.instances = 0

    mw.WorkerCore = O2WorkerCore
    args = ["-q", "-p", "no:cacheprovider", "--rootdir", O3] + [os.path.join(O3, "tests", t) for t in TESTS]
    rc = pytest.main(args + sys.argv[1:])
    print(f"O2 DraftCore instances created by O3's tests: {O2WorkerCore.instances}")
    if rc == 0 and O2WorkerCore.instances == 0:
        print("no test built a worker core: the patch did not take")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
