"""decode_apply runs on sparkhead's host python3: standard library only (no numpy/torch there)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]


def test_decode_apply_imports_and_renders_without_numpy_or_torch(tmp_path):
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None\n"
            "import hashlib\n"
            "from d2 import decode_apply as A\n"
            "base = b'PROD_K1=1\\n'\n"
            "A.BASE_ENV = hashlib.sha256(base).hexdigest()\n"
            "out = A.render(base, 'a' * 40, 'run1')\n"
            "assert out.startswith(base) and b'VLLM_DSPARK_DECODE_CAPTURE_DIR=/d2-decode' in out\n"
            "assert 'numpy' not in {m for m, v in sys.modules.items() if v is not None}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONPATH'] = str(REPO / 'offload/dspark')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, ' + repr(env['PYTHONPATH'])
                    + '); exec(' + repr(code) + ')'], check=True, env=env, cwd=tmp_path)
