"""The engine controls vary scheduling while keeping arm identity and capacity."""
from __future__ import annotations

from pathlib import Path
import shlex


def test_each_native_arm_has_matched_sync_async_configs():
    root=Path(__file__).resolve().parents[3]/'launch/configs'
    identities=set()
    labels=set()
    for prefix in ('E2-native-packed','E3-native-expanded','E4-native-kx','E4-native-kx2'):
        envs=[]
        for mode in ('async','sync'):
            text=(root/f'{prefix}-{mode}.env').read_text()
            values=dict(line.split('=',1) for line in text.splitlines() if line and not line.startswith('#'))
            env=dict(token.split('=',1) for token in shlex.split(values['EXP_DOCKER_ENV'].strip('"')))
            assert env['VLLM_DSPARK_REMOTE_ASYNC']==('1' if mode=='async' else '0')
            assert values['KV_BYTES_OVERRIDE']=='6442450944'
            assert values['LABEL']==f'{prefix}-{mode}'
            labels.add(values['LABEL'])
            envs.append(env)
        assert {k:v for k,v in envs[0].items() if k!='VLLM_DSPARK_REMOTE_ASYNC'}=={
                k:v for k,v in envs[1].items() if k!='VLLM_DSPARK_REMOTE_ASYNC'}
        identities.add(envs[0]['VLLM_DSPARK_REMOTE_IDENTITY'])
    assert len(identities)==4 and len(labels)==8
