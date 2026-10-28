"""Coordinator post-window commands on x86host only. Default is offline dry-run.

Build -> verify/plumbing -> S25 aggregates -> bounded solo G1 -> fresh source and
Amendment-6ab pilot prepare/sample. Does not train, deploy or contact a Spark.
Phases allow resuming G1 without rebuilding or redrawing a prepared sample.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess

from .replay_window import CAMPAIGN, ROOT, PYTHON, DryIO, IO, LiveIO, equivalence, require, source_pin_command

SOLO = 'GPU-bc62df2a-59cf-9828-7d43-558d1355bab7'


class X86hostIO(LiveIO):
    """Reuse owned-command lease supervision with NO Spark memory probes."""
    def memcheck(self) -> None:
        require(not (self.out/'STOP').exists(), 'coordinator STOP file present')
        self.call('x86host-memory', 'x86host', "awk '/^MemAvailable:/ {a=$2} /^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} "
                  "END {if (a<16777216 || t!=f) exit 1}' /proc/meminfo", timeout=8)


def run(io: IO, source: str, context: str, phase: str) -> None:
    for path in (context,):
        require(Path(path).is_relative_to(ROOT) and '..' not in Path(path).parts, 'context must be inside private D2 root')
    code = f'{ROOT}/source/{source}/repo'
    caches = (f'XDG_CACHE_HOME={ROOT}/cache HF_HOME={ROOT}/cache/hf CUDA_CACHE_PATH={ROOT}/cache/cuda '
              f'TORCHINDUCTOR_CACHE_DIR={ROOT}/cache/inductor EXLLAMAV3_TUNE_CACHE={ROOT}/cache/exl3-tune.bin')
    base = (f'set -euo pipefail; umask 077; export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {caches} '
            f'PYTHONPATH={code}:{code}/offload/dspark; cd {code}; ')
    def cmd(command: str) -> str:
        return base+f'{PYTHON} -u -m offload.dspark.d2.'+command
    def step(name: str, command: str, timeout: int = 1800) -> str:
        # DryIO's replay-memory simulation is unnecessary here: post-window has no Spark steps.
        if isinstance(io, DryIO):
            return io.call(name, 'x86host', command, timeout, supervised=False)
        return io.call(name, 'x86host', command, timeout, supervised=True)
    eq = ROOT+'/equiv-w9-'+source[:12]
    shards, work = ROOT+'/shards', ROOT+'/g1'
    manifest = shards+'/MANIFEST.json'
    step('post-source', f'set -euo pipefail; test "$(cat {ROOT}/source/{source}/SOURCE_COMMIT)" = {source}; '
         f'cd {ROOT}; sha256sum -c INPUTS.SHA256 >/dev/null')
    step('post-source-bytes', source_pin_command(Path(__file__).resolve().parents[3], source))
    equivalence(io.call('equiv-gate', 'x86host', f'cat {eq}/gate.json'))
    step('context', cmd(f'post_replay context --root {shlex.quote(context)} --manifest {ROOT}/prompts/manifest.json '
                        f'--plan {ROOT}/replay/plan.jsonl'))
    if phase in ('all', 'build'):
        step('dataset-build', cmd(f'dataset build --replay {ROOT}/replay --out {shards} '
                                 f'--prompt-manifest {ROOT}/prompts/manifest.json --equiv-gate {eq}/gate.json '
                                 f'--sequence-context {shlex.quote(context)}'), timeout=7200)
    step('dataset-verify', cmd(f'dataset verify --out {shards}'), timeout=3600)
    step('dataset-binding', cmd(f'post_replay binding --root {shards} --replay {ROOT}/replay --gate {eq}/gate.json '
                               f'--context {shlex.quote(context)} --manifest {ROOT}/prompts/manifest.json'))
    step('plumbing', cmd(f'g1 plumbing --shards {shards}'), timeout=3600)
    if phase in ('all', 'build'):
        step('s25-aggregate', cmd(f'post_replay aggregate --root {shards} --out {ROOT}/s25-{source[:12]}.json'))
    if phase == 'build':
        return
    if phase in ('all', 'g1'):
        # Only the named solo. Fixed nonblocking flock and <=14-minute holds. No unit/pair commands.
        for split in ('acap', 'heldout_legal', 'heldout_general', 'dev'):
            gpu = ('set -euo pipefail; '+base+f'export CUDA_VISIBLE_DEVICES={SOLO} OMP_NUM_THREADS=8 '
                   f'TORCH_EXTENSIONS_DIR={ROOT}/g1-build TRITON_CACHE_DIR={ROOT}/g1-triton '
                   f'EXLLAMAV3_TUNE_CACHE={ROOT}/g1-tune.bin; '
                   'tmux has-session -t =dsv41-powerlog; '
                   f'test "$(nvidia-smi --id={SOLO} --query-gpu=power.limit --format=csv,noheader,nounits)" = 225.00; '
                   f'exec 9> "$HOME/dsv41-offload/locks/gpu-bc62df2a.lock"; flock -n 9; '
                   f'timeout 840 {PYTHON} -u -m offload.dspark.d2.g1 k2 --shards {shards} --split {split} '
                   f'--work {work} --deadline-s 780')
            step('g1-k2-'+split, gpu, timeout=850)
            # G1 can return 0 at its deadline with unfinished sequences. Never score that as completion.
            step('g1-complete-'+split, cmd(f'post_replay k2-complete --root {shards} --work {work} --split {split}'))
            step('g1-score-'+split, cmd(f'g1 score --shards {shards} --split {split} --work {work} --out {work}/{split}.json'))
            if split == 'acap':
                step('g1-binding', cmd(f'post_replay g1 --root {work}/acap.json'))
    step('g1-binding-final', cmd(f'post_replay g1 --root {work}/acap.json'))
    if phase == 'g1':
        return
    for split in ('acap', 'heldout_legal'):
        step('pilot-k2-complete-'+split, cmd(f'post_replay k2-complete --root {shards} --work {work} --split {split}'))
    # Create a new immutable pilot source + NEW plan. Never reuse the older rehearsal snapshots.
    fresh = ROOT+'/pilot/w9-'+source
    script = ('from offload.dspark.d2.train.pilot_verdict import RULE_VERSION; '
              'from offload.dspark.d2.train.pilot_eval import EVALUATOR; '
              'assert RULE_VERSION=="amendment-6ab-source-prompt-bootstrap-v2"; '
              'assert EVALUATOR=="G2-reference-top20-tail-fixed-K2-amend6ab-v2"; '
              'print(RULE_VERSION,EVALUATOR)')
    step('pilot-rule', base+PYTHON+' -c '+shlex.quote(script))
    step('fresh-pilot-source', 'set -euo pipefail; umask 077; '
         f'test ! -e {fresh}; mkdir -m 700 {fresh}; cp -a {code} {fresh}/repo; '
         f'printf "%s\\n" {source} > {fresh}/SOURCE_COMMIT; chmod -R go-rwx {fresh}; '
         f'find {fresh}/repo -type f -exec chmod 400 {{}} +; find {fresh}/repo -type d -exec chmod 700 {{}} +')
    parity = ROOT+'/train/parity50-solo225/parity.json'
    plan = fresh+'/real/plan.json'
    prep = (f'export D2_TRAIN_ROOT={fresh} D2_GIT_SHA={source} D2_PYTHON={PYTHON} D2_PILOT_GPU={SOLO} '
            f'PYTHONDONTWRITEBYTECODE=1 {caches}; '
            f'bash {fresh}/repo/offload/dspark/d2/train/run_pilot_locked.sh prepare '
            f'-m offload.dspark.d2.train.pilot prepare --manifest {manifest} '
            f'--ckpt /opt/dsv41/x86host-data/native-1017 --parity-receipt {parity} '
            f'--g1-receipt {work}/acap.json --plan {plan}')
    step('pilot-prepare', prep, timeout=1800)
    step('pilot-sample', f'export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; cd {fresh}/repo; '+PYTHON+
         f' -m offload.dspark.d2.train.pilot sample --plan {plan} --manifest {manifest} '
         f'--k2-acap {work}/acap.k2_drafts.jsonl --k2-legal {work}/heldout_legal.k2_drafts.jsonl')
    step('pilot-plan-check', f'export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; cd {fresh}/repo; '+PYTHON+
         ' -c '+shlex.quote('from pathlib import Path; from offload.dspark.d2.train.pilot import read_spec; '
                           f's=read_spec(Path({plan!r}),sampled=True); assert s["synthetic"] is False; '
                           'print("real sampled plan validated; no training started")'))
    print('Post-replay preparation complete. No training, Spark access, production service change or deployment.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    ap.add_argument('--source', default='DRY_RUN_SOURCE')
    ap.add_argument('--context', default=ROOT+'/provenance/sequence-context.json')
    ap.add_argument('--phase', choices=('all', 'build', 'g1', 'pilot'), default='all')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    io: IO
    if a.execute:
        require(os.environ.get('D2_COORDINATOR') == 'Deepseek/coordinator', 'coordinator execution only')
        require(re.fullmatch('[0-9a-f]{40}', a.source) is not None, 'full source SHA required')
        repo = Path(__file__).resolve().parents[3]
        require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == a.source,
                'local source differs from immutable pin')
        require(a.out is not None and a.out.resolve().is_relative_to(CAMPAIGN), 'new campaign output directory required')
        os.umask(0o077)
        a.out.mkdir(mode=0o700, parents=True, exist_ok=False)
        io = X86hostIO(a.out, a.source)
    else:
        io = DryIO()
    run(io, a.source, a.context, a.phase)


if __name__ == '__main__':
    main()
