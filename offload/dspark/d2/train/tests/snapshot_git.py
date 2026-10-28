"""Read historical source through a disposable local clone, never inherited git state."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    scratch = os.environ.get('TMPDIR') or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix='source-git-', dir=scratch) as directory:
        clone = Path(directory)/'repo'
        subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(repo), str(clone)],
                       env=env, check=True, capture_output=True)
        return subprocess.run(['git', '-C', str(clone), *args], env=env, check=True, capture_output=True)
