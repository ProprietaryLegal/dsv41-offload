"""Admit an existing immutable pilot without rewriting its source-bound evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .pilot import Recipe
    from .loop import Run

from .data import sha256

# Reviewed immutable pilot-2 snapshot, integration d1dd41a2744fd78cc9b74f83a98af1765e9cb812.
# All entries are pinned, including the full_* modules already present in that pilot.
PILOT_COMMIT = 'd1dd41a2744fd78cc9b74f83a98af1765e9cb812'
PILOT_SOURCES: dict[str, str] = {'d2/g1.py': 'ea9b45b69a338a228dd341b866231523463e699e53f765864c27eff4b2f55063',
 'd2/live_reference.py': 'b97d34e3d7903f2ab32397f3a067d1369a612192ace256534619661a1ace1bc5',
 'd2/train/__init__.py': 'b3186ec73bb9333a41bbf69bbad10e57937db3af9cd59a10c6599484966344f2',
 'd2/train/agreement.py': 'f36138eaeeca1f77754fd11149fd4e43c67af3ee20752ca8919e77ebf48d1898',
 'd2/train/audits.py': '033d3da43418c97ca272279d43b81dbc4352a21d8a096248279b50b96e83490f',
 'd2/train/benchmark.py': 'c270fef348fc3400dc3cb84d91bb89a73aaa0c298bf19518110e5f6b05b7b4d8',
 'd2/train/cache_guard.py': 'dba4d5a0a611bc8db5e89099385c23ddc6713584fc1857bf300eb1bb2b21758c',
 'd2/train/data.py': '1d9ed09777de905e2397311ec6c5aabaa7ef325ea950449c2d46d080f32dc46b',
 'd2/train/diagnose.py': 'bfc2dc0b5f62bcfee7be2ed8b00319cc9e525d8d9a97c13f4e142c2086053ba3',
 'd2/train/evaluate.py': 'f7612f273eb003684d86c4cf4e5ccd95311940e21fce64278af64c61212b6622',
 'd2/train/fast_gpu.py': '5417d405ffd6f0caae3d543d8452285254d0863850b0ebefd26c596b17970817',
 'd2/train/fast_loss.py': '838f3924048717debb28f823be23c13137d421ead1d23ab858edab7b11babe9f',
 'd2/train/fast_policy.py': '10581a954243532fd365f6dbe83b0c1cef55aea68524312a320395fa79b52afa',
 'd2/train/full_admission.py': 'de4af31f8822c73af34f1921a955b28f1eb9552de1a4feea92c348b6942c4bcc',
 'd2/train/full_eval.py': '4ad5958f3f304e5e01f4c76fb9862d9bbb2ab88166bdc9499eea5b5f18f5bae6',
 'd2/train/full_stage.py': 'cd8cc4b052a8bb8c7a03b4611c7cf336d489cc4b583bca48c29571b965a48bec',
 'd2/train/full_train.py': '6e92b24c81485ec1b583d7c1225cd9b266eb072c3fc005cfb4b94a8a42ad1d8a',
 'd2/train/lease.py': '8cba44de6993506d4899812b049c3adb13ce5429a886e925e9f2f2f06ca75859',
 'd2/train/loop.py': '6c5cad94195581efc25452df28d8884f0511a04df7e7a65b5390cb1a61ce9858',
 'd2/train/loss.py': '600b0c2c97eb7002cab10dc4363885aebce0c3e68cfb7d49341f1b3410696281',
 'd2/train/model.py': 'ddbc9e9286253ea26d2c3ac03a5888a084139f1943cfcc85fb7051333333a52f',
 'd2/train/n1_gate.py': 'a9003ff53b2aac1e6dbb6439045e1ae80ff617ac7d71b099f3f1931a6a3db18e',
 'd2/train/ops.py': '8397d3e21fcd8bdb620d70d948f6083a81be57a0f73735039bf6e4e8c66dc53e',
 'd2/train/package.py': '80894df5828b27be174b0a0d08967deadb6f61376469f8d9b3ee1acd2e4023e3',
 'd2/train/packed_forward.py': 'c1a92e3d7c8933fdda40c70d4b2d58c1d3de1c33205970530d2651dfff184f98',
 'd2/train/perf.py': 'f36d8ba04608c42ccae7bc13bdf692afce51d77d08d7f31762145d8b858be51e',
 'd2/train/pilot.py': 'd0b58a78e553780f50869b3c66dc8f3b7333e3498f1b25a192dca43b47b48983',
 'd2/train/pilot_eval.py': '6e58c760d0162bc1b721a2a19760ae995e7f219c5ddf7beef9a0fac16a11ade5',
 'd2/train/pilot_sample.py': 'f4c20097634c8cc427821c512eda5cfd925ceaa16628b56f905e0628d9b668cf',
 'd2/train/pilot_stage.py': 'be1c60456592731a415ef445793ba6338713b35eee398d9da2ebfa332fc639a9',
 'd2/train/pilot_statistics.py': '88296213d0d9ad6caa97724ca7c43e19ed727d5f1a7a53f8bbdb3c3b70995fca',
 'd2/train/pilot_synthetic.py': '1539f010ee0cd78a80c477a7eede1e7404fb6d92d91d6429858e9bdfbaccabef',
 'd2/train/pilot_units.py': 'bdca478a64366b242fb01c920206ca6e8b3de90e3671b17bba07088bf1ebc4b4',
 'd2/train/pilot_verdict.py': 'bb8efa7078a126d78864d8692ec7739f13900d7b029653cec76dee3c709a5c2a',
 'd2/train/probe.py': '67c1997aa531e713f3aca5442a3bb186cb2a0c73f7a7c2416079dbf3e6b5cae3',
 'd2/train/reference.py': 'aa28077e13373c58524583363b6c37e9bc437ba1c16817f72c9d4c51e9ff6f81',
 'd2/train/replay.py': '7e7cb6291d57ecb0fe9431c45bb1eda356600b36be9726fe4f47624018f5970b',
 'd2/train/run_pilot_locked.sh': '894fa95b8cf5884d6a4ffc65975a9479b62f72cfe526cfee98473afb7f0f300c',
 'd2/train/safety.py': '4cbe87751f93e2dc85a2ffb756261f4b255d2950ae636e2da539dff9d5eea595',
 'd2/train/synthetic.py': 'e86b496105f94a045ac916e9982d172cbf21ec2f273a8612cec1b29ad381cb3a',
 'd2/train/vocab_audit.py': '24aa9ad825701862f4c63741fc57c57b0c14a8153f2272f0d075b9c98e279fe5',
 'd2/train/worker_adapter.py': 'bd7c1b31977be86684f905fc4f340b389adf3a5f30e1605c1be353fac91d41cf',
 'd2/train/worker_eval.py': '48fa147bfedf25a9cc07a6d00bf87c15a293df82c738ae51ebc91b31155521bc',
 'd2/train/worker_eval_build.py': 'f5acde63dc4d434b9e7926c71149cdd0e5ac8773df95ae9319e28ec088dd2f23',
 'd2/train/worker_eval_lease.py': 'd1435419d2b2e921f8c802106b9e2869332481ddf706570dbbf2833560e9d7d0',
 'native/formats.py': '7af5205b0feec1f96b22b68a42aa219cf8f836350fe01630d2998de3580b7f8d',
 'native/official.py': '21f46db81eadbcc5afacc282678269f99ed57297df08ceb6b17268407f331853'}
PACKAGE_SOURCES = {'bench.py': 'cd3ca6059faa538efa44554aba25f396d17536cb95f4bc6725931fa7f0aa964b',
 'functest.py': 'e988dc502a8cd871e7829134f8b02e4a2342cb5ae36cf56615a3a5e80362973b',
 'quality.py': '3cd295f4d16eb55415bb040c8d10fcc97f60fac73de535494d27f16a663e0325',
 'report_tables.py': '7f47437869e017f82b940777701c05aea92e9ba3455f629a8168d68b5049086a',
 'verify_hashes.py': '9826d2afbdbce87cb118d6fe91677166c16b6c2ba1907b6772f5f6fef3a76fd0',
 'offload/dspark/build_inventory.py': 'bf71308c2556fdf1a2a81390aa277550d11d7e00497af7e86d7bcd784ddaea40',
 'offload/dspark/copy_to_x86host.py': '648195654cd560a1ccab1092df11dcbfbfd225e7878d81fb0374541086b5361f',
 'offload/dspark/d2/__init__.py': 'f0543e9e09db6e8a2cd4589a3b5054b528f3587042728119f80378fd7fbc0bb8',
 'offload/dspark/d2/after_replay.py': 'cc37d80c69a9009df96ee4178902bb89d0ecd0e0a5865ee356d8e7dd01501fa1',
 'offload/dspark/d2/assemble.py': 'adb1f924a9fd49b0595537fa056cf4c2da8b835765343de7271d170d70a90585',
 'offload/dspark/d2/assembly_paths.py': '883d69daec4020b3e5e22df43439cc81f32159809824a96cbff14951f2d0893e',
 'offload/dspark/d2/copy_binding.py': '21a66ce410f2f62e97a7f925f0147b706db1ee2fdb9a19970d20c4c10583d0db',
 'offload/dspark/d2/dataset.py': '5cc089ab9a9a0c05c80f4defa7df5c7b5b7e562c07f27ee46aa59e6130e73462',
 'offload/dspark/d2/decode_apply.py': '6cf189790fc09f41fe7869df921ee7ed66eef302907b0e0613abdf5848e8e66e',
 'offload/dspark/d2/decode_control.py': '1bd9e4fb1bd3ea4476cdd06dcc63d5588f88d5df93b9463f7e10c601b049adfa',
 'offload/dspark/d2/decode_copy.py': 'ce44b3f339a9eed12badaab74d817fb5c771de1131bbe6ff353f8f0fcc217b36',
 'offload/dspark/d2/decode_dataset.py': '2516c443bf0d945860fbba011ffdd45dfe9bd5bfdd62da0617dbca6ab587c331',
 'offload/dspark/d2/decode_smoke.py': '1a0cd273b434b50e92f122e5aab077c792310bebbc55fb2254a065b93ca94e76',
 'offload/dspark/d2/decode_smoke_dry.py': 'f61a84eef19c5265ceeecdc49f0a15fc43e91f76ceadf996d70e395c3b383735',
 'offload/dspark/d2/equiv.py': 'c5ff34ac131b2917446ce945dc0d576246605bb90300b03abba00f507062aebd',
 'offload/dspark/d2/g1.py': 'ea9b45b69a338a228dd341b866231523463e699e53f765864c27eff4b2f55063',
 'offload/dspark/d2/generate.py': '56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a',
 'offload/dspark/d2/generation_export.py': '680b2c89badbac4aeb1c531da3a746528ba59056307f2dc70d4aca3784244e00',
 'offload/dspark/d2/generation_top20.py': '984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d',
 'offload/dspark/d2/greedy_reference.py': '53950f25073e0135b19558619ca253a573a13108246d09afc4c340f752b20d08',
 'offload/dspark/d2/live_reference.py': 'b97d34e3d7903f2ab32397f3a067d1369a612192ace256534619661a1ace1bc5',
 'offload/dspark/d2/owned_command.py': '5f9e537318cfb30b745bcafd401d87ea7dccc62896ad156bcbb0f8cb542e8291',
 'offload/dspark/d2/pass2_attest.py': '28bd603c8d960b11c9628d42bb4c092086f76a4a309e792c2696bf9bd9b17d6d',
 'offload/dspark/d2/pass2_context.py': 'fb79c7790855d99a21f8f838a9e335cbd20d29d1b2663ea0cb88abb009208c90',
 'offload/dspark/d2/post_replay.py': 'bad93505e6a77854d0874a016d48d9ae71b7a02b1e527ad0e5978dc2846f28a5',
 'offload/dspark/d2/prompts.py': '117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966',
 'offload/dspark/d2/replay.py': 'c18d2aaef417bdeb9a3a1197d0d18cc4553bbe2949d4f8a290d82dd282518d8e',
 'offload/dspark/d2/replay_evidence.py': '5c215e23e41c36d96d42aa3a98c54ce9162c156e8c6589517b10eb5b9891a307',
 'offload/dspark/d2/replay_window.py': '4d0111ae3c9bde2e17407cc767588f87ebc817e7b68d0562f765fa8b12062811',
 'offload/dspark/d2/reserved_test.py': 'cd30c7a707faf98718cdfdbd5e1356c8d8fb20b7d7a2cc611b2bbc7cfb54dfdd',
 'offload/dspark/d2/scheduler_cpu.py': 'b7fcfd4252cb534cbece59b80c58b24791a29707680aa2a7262fc4f9b9dfa9de',
 'offload/dspark/d2/sequence_audit.py': '0d28bb5f7770e0c79e1e4e8253cf12b2916011fa607b845c024827ef8a07ba0f',
 'offload/dspark/d2/shards.py': '4096002f84b203414068962cdbe248962dc05d1e78a2ee5bd22aced094955483',
 'offload/dspark/d2/smoke_active.py': 'cd7f1a2e9c53645bbe89988a19eb858ff08a8a8bc56975aac001704270af1967',
 'offload/dspark/d2/smoke_latency.py': '7ea3ac79d9ac3898bf941b9fa3735dfe7058aaa9dffffd58548bbccbb35e9160',
 'offload/dspark/d2/smoke_prepare.py': '359fd99c1307a3742c59b66335992aa9e94946fb166c37624861b6045ffe82dc',
 'offload/dspark/d2/smoke_window.py': '0d0c60d6aa888a17f4d5aa4cfa53805215b97033ece28ed8a43546ed72ce0e78',
 'offload/dspark/d2/split_audit.py': '350fe94aae30f76c77ca05ee7568cc9136502d3baeb1c71385e0f1ed0dece92b',
 'offload/dspark/d2/window_checks.py': 'ff2e5d972fbd120648e74d0f2d61fb486e397bc4c384104b2d64a87d7a47e040',
 'offload/dspark/inventory_source.py': '6ac7d62ece153f8e4a0e5594109cc66c7f4691d9c705dca14f29f20a0845c3d9',
 'offload/dspark/native/__init__.py': 'b18ff0f695960a6296552134890d507bfd562ea1522239c4c094d4dd7de77d19',
 'offload/dspark/resume_engram.py': 'd7eedcdf24d9fcf37c89134bb75df1c953d738b9e994c14ad4242733bea20286',
 'offload/dspark/test_transfer.py': '664cc4202edb74c7f1af921ad5bb2f94a118b60b1468696ea0f7df50f3484b8a'}
PILOT_LOOP = PILOT_SOURCES['d2/train/loop.py']
FULL_LOOP = '6157ca76201688eadff65ba934758aff68fb50684afa60f2a15a9dfd89174cf0'
# Only explicitly reviewed full-run bridge modules can differ. The admission
# module is this policy itself; all other differences have literal byte pins.
FULL_PATCHES: dict[str, str] = {'d2/train/full_eval.py': '7ee82cc675252163f8f1011509037818aa5cc822dae3f3418674a6e39b626c2e',
 'd2/train/full_stage.py': 'ed1156548b819baa92b8755fbf9404a0c425a4e22c8b0837bfc90f79784e8905',
 'd2/train/full_train.py': '6e92b24c81485ec1b583d7c1225cd9b266eb072c3fc005cfb4b94a8a42ad1d8a'}


# Read-only modules present in the full-run tree but never in the pilot-2 snapshot and never imported on the full-run
# path; each is byte-pinned (S41 FIX 2: offload-x86host's D2-DIAG tool, merged f76a787).
FULL_EXTRAS: dict[str, str] = {'d2/train/pilot_diag.py': 'a20987d409738ce45b4dce7670a6fefcd4a655c8ab37ebc446173499827f2dbb'}


def verify_pilot_source(root: Path, sources: dict[str, str]) -> None:
    """Require the entire reviewed pilot-2 map and its unchanged original bytes."""
    from .pilot import sources as current_sources
    if sources != PILOT_SOURCES:
        raise ValueError('unsupported pilot source lineage')
    current = current_sources()
    if current.get('d2/train/loop.py') != FULL_LOOP:
        raise ValueError('full loop changed beyond reviewed admission patch')
    extras = set(current) - set(sources)
    if set(sources) - set(current) or extras - set(FULL_EXTRAS) or any(current[k] != FULL_EXTRAS[k] for k in extras):
        raise ValueError('full training source file set differs from qualified pilot source')
    dspark = root/'offload/dspark'
    actual_files = {str(p.relative_to(dspark)) for p in (dspark/'d2/train').glob('*.py')}
    expected_files = {p for p in sources if p.startswith('d2/train/') and p.endswith('.py')}
    if actual_files != expected_files:
        raise ValueError('immutable pilot source file set changed')
    for name, expected in sources.items():
        path = dspark/name
        if not path.resolve().is_relative_to(dspark.resolve()) or sha256(path) != expected:
            raise ValueError('immutable pilot source changed: '+name)
        reviewed = FULL_PATCHES.get(name, FULL_LOOP if name == 'd2/train/loop.py' else expected)
        if name != 'd2/train/full_admission.py' and current.get(name) != reviewed:
            raise ValueError('full training differs from qualified pilot source: '+name)
    # Unbound startup hooks and package initializers must never execute in the child.
    for directory in (root, root/'offload', dspark, dspark/'d2'):
        if {str(p.relative_to(root)) for p in directory.glob('*.py')} != {n for n in PACKAGE_SOURCES if root/Path(n).parent == directory}:
            raise ValueError('unbound pilot import source')
    # A package directory beside a pinned module would shadow it on import (S41 observation): refuse it.
    pinned = [dspark/n for n in sources] + [root/n for n in PACKAGE_SOURCES]
    if any(m.with_suffix('').is_dir() for m in pinned if m.name != '__init__.py'):
        raise ValueError('unbound pilot import shadow directory')
    # A source hash cannot authenticate a cached/extension implementation or
    # a package directory shadowing stdlib/third-party imports in sys.path.
    if any(p.suffix in ('.pyc', '.pyo', '.so', '.pyd') for p in root.rglob('*') if p.is_file()):
        raise ValueError('unbound compiled pilot import source')
    shadowed = set(sys.stdlib_module_names) | {'sitecustomize', 'usercustomize', 'torch', 'numpy', 'safetensors'}
    for directory in (root, dspark):
        names = shadowed | ({'native', 'd2'} if directory == root else {'offload'})   # S41: dspark/offload would hijack the package
        if any(p.stem in names for p in directory.iterdir()):
            raise ValueError('unbound pilot import shadow')
    for name, expected in PACKAGE_SOURCES.items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError('immutable pilot package initializer changed')



def require_full_pilot(path: Path, frozen: str, manifest: Path, parity: Path, g1: Path,
                       batch: int, chunk: int, rank: int, pilot_source: Path | None, k2: dict[str, Path] | None = None) -> str:
    """Run the UNCHANGED policy in its verified original source namespace, CPU only.

    It recomputes PASS/intervals, sample, units and all arm/checkpoint hashes.
    No old source is modified and no pilot data is copied or relabelled.
    """
    if pilot_source is None:
        raise ValueError('pilot source identity mismatch: immutable --pilot-source required')
    before = sha256(path)
    receipt = cast(dict[str, object], json.loads(path.read_text()))
    plan = cast(dict[str, object], receipt['plan'])
    verify_pilot_source(pilot_source, cast(dict[str, str], plan['source_sha256']))
    if sha256(path) != before:
        raise ValueError('pilot receipt changed during source verification')
    if k2 is None:
        k2 = {s: g1.parent/(s+'.k2_drafts.jsonl') for s in ('dev', 'heldout_legal')}
    if set(k2) != {'dev', 'heldout_legal'}:
        raise ValueError('full admission requires both pilot K2 sources')
    code = ('import sys; sys.path[:0] = '+repr([str(pilot_source), str(pilot_source/'offload/dspark')])+'; from pathlib import Path; '
            'from offload.dspark.d2.train.fast_policy import require_pilot; '
            'result = require_pilot(Path(sys.argv[1]),sys.argv[2],'
            'Path(sys.argv[3]),Path(sys.argv[4]),Path(sys.argv[5]),'
            'int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8])); '
            'import json; from offload.dspark.d2.train.pilot_eval import verify_sample_draw; '
            'verify_sample_draw(json.loads(Path(sys.argv[1]).read_text())["plan"], Path(sys.argv[3]), '
            '{"dev": Path(sys.argv[9]), "heldout_legal": Path(sys.argv[10])}); print(result)')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(PYTHONPATH=str(pilot_source)+':'+str(pilot_source/'offload/dspark'),
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    try:
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(path), frozen, str(manifest), str(parity),
                             str(g1), str(batch), str(chunk), str(rank), str(k2['dev']), str(k2['heldout_legal'])], cwd=pilot_source,
                            env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError('original pilot verifier refused: '+exc.stderr) from exc
    if result.stdout.strip() != before or sha256(path) != before:
        raise ValueError('pilot receipt changed during original-source verification')
    verify_pilot_source(pilot_source, cast(dict[str, str], plan['source_sha256']))
    pilot_recipe(path, before)
    return before


def pilot_recipe(path: Path, expected_hash: str | None = None) -> "Recipe":
    """Read an explicit recipe, bound to the plan's exact matched Run contract.

    This is an identity check, never a substitute for require_full_pilot.
    Consumers pass the admitted byte hash to prevent a swapped receipt.
    """
    from dataclasses import asdict
    from .pilot import RECIPES, Recipe, pilot_run
    before = sha256(path)
    plan = cast(dict[str, object], json.loads(path.read_text())['plan'])
    recipe = plan.get('recipe')
    if recipe not in RECIPES:
        raise ValueError('full run requires explicit qualified pilot recipe')
    selected: Recipe = recipe
    if plan.get('run') != asdict(pilot_run(selected)):
        raise ValueError('pilot recipe and matched run differ')
    if sha256(path) != before or (expected_hash is not None and before != expected_hash):
        raise ValueError('pilot receipt changed while reading recipe')
    return selected


def full_run(recipe: "Recipe") -> "Run":
    """Same qualified recipe, three full epochs; scheduler uses full update count."""
    from dataclasses import replace
    from .pilot import pilot_run
    return replace(pilot_run(recipe), epochs=3)
