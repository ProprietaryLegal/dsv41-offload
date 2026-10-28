from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_probe_refuses_filesystem_writes(tmp_path: Path) -> None:
    target = tmp_path / 'must-not-exist'
    code = '''from pathlib import Path
import sys
from offload.dspark.native.tools.n6.batch_probe import readonly_runtime
readonly_runtime(Path(sys.argv[1]))
Path(sys.argv[2]).write_text('forbidden')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(target)],
                            text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'filesystem write refused' in result.stderr
    assert not target.exists()
