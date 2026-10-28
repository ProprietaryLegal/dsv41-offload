"""CPU-only checks: source configs without any launcher, connection or service call."""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from offload.dspark.depth.screen import obj

ROOT = Path(__file__).resolve().parents[2]/'launch/configs'
CONTROL = ROOT/'D-kv6-async-depth-control.env'


def load(path: Path) -> dict[str,str]:
    script = 'set -eu; source ~/.config/ndm/paths.sh; . "$1"; printf "%s\\0" "$EXP_DOCKER_ENV" "$SPEC_OVERRIDE" "$KV_BYTES_OVERRIDE" "$MAX_LEN_OVERRIDE" "$LABEL" "$OVERLAYS" "$EXTRA_DOCKER_ARGS"'
    data = subprocess.check_output(['bash','-c',script,'config-test',str(path)],text=True).split('\0')
    return dict(zip(['env','spec','kv','maxlen','label','overlays','mounts'],data))


def env(config: dict[str,str]) -> dict[str,str]:
    pairs = [s.split('=',1) for s in shlex.split(config['env'])]
    assert len({p[0] for p in pairs}) == len(pairs), 'duplicate environment key'
    return dict(pairs)


def test_control_matches_pinned_production_environment() -> None:
    text = subprocess.check_output(['git','show','3c1dc6b:dsv41.env'],text=True)
    expected: dict[str,str] = {}
    for key in ('OFFLOAD_DSPARK_ENV','OFFLOAD_DOCKER_ENV'):
        line = next(s for s in text.splitlines() if s.startswith(key+'='))
        expected.update(dict(s.split('=',1) for s in shlex.split(shlex.split(line.split('=',1)[1])[0])))
    c = load(CONTROL)
    assert env(c) == expected
    assert c['kv'] == '6442450944' and c['maxlen'] == '327680'


def test_candidates_change_only_declared_cost_and_cap() -> None:
    control = load(CONTROL)
    ce = env(control)
    for suffix,cost in [('residual0','0'),('residual08','0.8'),('residual15','1.5'),('cap3','0.8'),('cap5','0.8')]:
        c = load(ROOT/f'D-kv6-async-depth-{suffix}.env')
        assert env(c) == dict(ce,VLLM_DSPARK_REMOTE_DRAFT_MS=cost)
        for key in ('kv','maxlen','overlays','mounts'):
            assert c[key] == control[key]
        spec = obj(json.loads(c['spec']))
        assert spec['num_speculative_tokens'] == 5
        expected = [[1,4,int(suffix[-1])]] if suffix.startswith('cap') else [[1,2,5],[3,4,4]]
        assert spec['num_speculative_tokens_per_batch_size'] == expected
        assert c['label'] == f'D-kv6-async-depth-{suffix}'


def test_fixed_calibration_disables_scheduler_adaptation_only() -> None:
    ce = env(load(CONTROL))
    for d in range(6):
        c = load(ROOT/f'D-kv6-async-depth-fixed{d}.env')
        assert env(c) == dict(ce,VLLM_DSPARK_DEPTH_POLICY=f'fixed:{d}',VLLM_DSV41_ADAPTIVE_K='0')
        assert obj(json.loads(c['spec']))['num_speculative_tokens_per_batch_size'] == [[1,4,5]]
