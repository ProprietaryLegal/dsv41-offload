from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_configs
from verify_overlay import rows, verify_tree


class ConfigTests(unittest.TestCase):
    def source(self, name, **overrides):
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PREFIX_', 'K1_', 'EXP_'))}
        env.update({'PREFIX_MODEL_SHA256':'a'*64, **overrides})
        command = ('set -e; source "$1"; python3 -c '\
                   + shlex.quote('import json,os; print(json.dumps(dict(os.environ)))'))
        return subprocess.run(['bash', '-ac', command, 'source-test', str(ROOT/'offload/launch/configs'/name)],
                              cwd=ROOT, env=env, text=True, capture_output=True)

    def test_fresh_generation_and_shell_contract(self):
        for name, params in make_configs.CONFIGS.items():
            self.assertEqual((ROOT/'offload/launch/configs'/name).read_text(), make_configs.render(name,*params))
            r = self.source(name)
            self.assertEqual(r.returncode,0,r.stderr)
            env = json.loads(r.stdout)
            args = shlex.split(env['EXP_EXTRA_ARGS'])
            self.assertEqual(args[0], '--kv-transfer-config')
            cfg = json.loads(args[1])['kv_connector_extra_config']
            self.assertEqual(cfg['cpu_bytes_to_use'], params[0])
            self.assertEqual(cfg['prefix_store_identity']['checkpoint_sha256'], 'a'*64)
            self.assertEqual(cfg['prefix_store_identity']['layout_sha256'], make_configs.layout_sha())
            pairs = [p.split('=',1)[1] for p in env['OVERLAYS'].split()]
            self.assertEqual(len(set(pairs)), len(pairs))
            self.assertIn('VLLM_DSPARK_REMOTE_ASYNC_PROXY=process',env['EXP_DOCKER_ENV'])
            self.assertIn('VLLM_ENGRAM_REMOTE_REQUIRED=1',env['EXP_DOCKER_ENV'])
            self.assertIn('VLLM_DSV41_KV_PAIR=3',env['EXP_DOCKER_ENV'])
            self.assertEqual(env.get('MAX_LEN_OVERRIDE'), '1048576' if params[1] else None)

    def test_control_does_not_need_identity(self):
        r = self.source(next(iter(make_configs.CONFIGS)), PREFIX_ENABLE='0', PREFIX_MODEL_SHA256='')
        self.assertEqual(r.returncode,0,r.stderr)
        env = json.loads(r.stdout)
        self.assertNotIn('EXP_EXTRA_ARGS',env)
        self.assertNotIn('VLLM_DSV41_OFFLOAD_MULTINODE', env['EXP_DOCKER_ENV'])

    def test_invalid_overrides_refused(self):
        for overrides in ({'PREFIX_MODEL_SHA256':''}, {'PREFIX_MODEL_SHA256':'z'*64},
            {'PREFIX_CPU_BYTES':'-1'}, {'PREFIX_CPU_BYTES':str(4<<30)}, {'PREFIX_LOAD_FRACTION':'0.45'},
            {'PREFIX_IO_CPUS':'$(false)'}, {'PREFIX_ENABLE':'2'}, {'PREFIX_MN_VERIFY':'bad'}):
            r = self.source(next(iter(make_configs.CONFIGS)), **overrides)
            self.assertNotEqual(r.returncode,0,overrides)

    def test_launcher_extra_args_reaches_real_launcher(self):
        # Execute serve_exp's actual prefix up to JIT, capturing its exported EXTRA.
        # No SSH, docker, start_exp, JIT, GPU or Spark contact is executed.
        script = (ROOT/'offload/launch/serve_exp.sh').read_text().split('JIT="')[0]
        env = {**os.environ, 'HEAD':'unused','WORKER':'unused'}
        for value in ('', '--kv-transfer-config \'{"kv_connector":"OffloadingConnector"}\''):
            result = subprocess.run(['bash','-c',script+'\nprintf "%s" "$EXTRA"'],
                env={**env,'EXP_EXTRA_ARGS':value},text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual('--kv-transfer-config' in result.stdout,bool(value))
            self.assertIn('--enable-prompt-tokens-details',result.stdout)

    def test_combined_base_and_runtime_tamper(self):
        self.assertEqual(len(rows()),20)
        verify_tree(Path('/opt/dsv41/data/src-image'),base=True)
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            tree = Path(tmp)
            for entry in rows():
                path = tree/entry['path']; path.parent.mkdir(parents=True,exist_ok=True)
                lane = 'kvpack' if (ROOT/'offload/kvpack/overlay'/entry['path']).exists() else 'prefixstore'
                shutil.copyfile(ROOT/'offload'/lane/'overlay'/entry['path'],path)
            verify_tree(tree)
            path.write_text('tampered')
            with self.assertRaises(ValueError): verify_tree(tree)


if __name__ == '__main__': unittest.main()
