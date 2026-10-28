"""The coordinator wrapper stays local and inert unless explicitly authorized."""
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / 'assemble.sh'


def test_dry_never_invokes_interpreter(tmp_path):
    hit = tmp_path / 'hit'
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/bash\ntouch "' + str(hit) + '"\nexit 91\n')
    fake.chmod(0o700)
    env = dict(os.environ, D2_ASSEMBLY_PYTHON=str(fake))
    result = subprocess.run(['bash', str(SCRIPT), '--root', '/nonexistent/tree with spaces'],
                            env=env, text=True, capture_output=True, check=True)
    assert 'DRY CPU:' in result.stdout and 'No commands executed' in result.stdout
    assert not hit.exists()


def test_execute_refuses_without_coordinator_before_interpreter(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'D2_COORDINATOR'}
    env['D2_ASSEMBLY_PYTHON'] = '/does/not/exist'
    result = subprocess.run(['bash', str(SCRIPT), '--execute', '--root', str(tmp_path)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'D2_COORDINATOR=Deepseek/coordinator' in result.stderr
