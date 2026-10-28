"""Refuse to run any offload test suite with git's repository-locating variables set.

2026-10-27 ~20:02Z: an R-SP suite run exported GIT_DIR=<dsv41-deploy>/.git so that lineage tests could find the repo. Every
fixture that builds a throwaway repository with `git init/commit/checkout` then wrote into the real repository instead:
offload-x86host moved onto a fixture commit and the main checkout was detached onto three commits that delete every tracked
file (the coordinator restored 535f344). Fixture git calls rely on cwd discovery, so these variables must never reach them.
"""
from __future__ import annotations

import os

_REPO_VARS = ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY',
              'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE')
_set = {name: os.environ[name] for name in _REPO_VARS if name in os.environ}
if _set:
    raise RuntimeError('offload tests refuse to run with ' + ', '.join(f'{k}={v}' for k, v in sorted(_set.items()))
                       + ': fixture git commands would write into that repository. Unset them; tests that need the real '
                         'repository find it from the working directory.')
