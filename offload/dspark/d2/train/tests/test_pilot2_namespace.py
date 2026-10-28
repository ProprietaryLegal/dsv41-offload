from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from offload.dspark.d2.train import pilot, pilot_stage as S
from offload.dspark.d2.train.pilot_sample import sample_plan
from offload.dspark.d2.train.safety import GPU, SOLO_B
from offload.dspark.d2.train.tests.test_pilot_sample import sample_fixture
from offload.dspark.d2.train.tests.test_pilot_stage import settings


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_dry_recipe_tag_namespace_and_commands(recipe: pilot.Recipe, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, 'argv', ['pilot_stage', '--recipe', recipe, '--run-tag', 'run-42',
        '--commit', 'a'*40, '--g1', '/absent/g1', '--k2-dev', '/absent/dev', '--k2-legal', '/absent/legal'])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('dry attempted subprocess'))
    S.main()
    rendered = json.loads(capsys.readouterr().out)
    suffix = '/run-42-p2-'+recipe
    assert rendered['snapshot'].endswith(suffix)
    assert rendered['result_receipt'].endswith(suffix+'/receipt.json')
    assert rendered['recipe'] == recipe
    sessions = [n for group in rendered['sessions'].values() for n in group]
    assert len(sessions) == len(set(sessions)) == 6
    assert all(n.startswith('d2pilot2-'+recipe+'-run-42-') for n in sessions)
    prepare = rendered['commands']['prepare'][0][1]
    assert '--recipe '+recipe in prepare and '--previous-plan ' in prepare
    assert prepare.index('pilot prepare') < prepare.index('pilot sample')


@pytest.mark.parametrize('tag', ['', '../old', 'A', 'x\ny', 'a'*41, 'literal;cmd'])
def test_invalid_tag_refuses_even_dry(tag: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='run tag'):
        S.commands(replace(settings(tmp_path), run_tag=tag))


@pytest.mark.parametrize('field', ['output', 'snapshot'])
@pytest.mark.parametrize('root', [S.FIRST_OUTPUT, Path.home()/'dsv41-offload/d2/pilot'/S.FIRST_NAME])
@pytest.mark.parametrize('relative', ['', 'arms/REF', '..'])
def test_first_pilot_and_ancestors_cannot_be_targets(field: str, root: Path, relative: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='first pilot namespace'):
        s = settings(tmp_path)
        S.check_settings(replace(s, output=root/relative) if field == 'output' else replace(s, snapshot=root/relative))


@pytest.mark.parametrize('recipe', pilot.RECIPES)
def test_sample_redraw_exactly_identical_across_recipe_plans(recipe: pilot.Recipe, tmp_path: Path) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    sample_plan(plan, manifest, paths)
    previous = tmp_path/'first-plan.json'
    previous.write_bytes(plan.read_bytes())
    original = json.loads(plan.read_text())
    original['recipe'] = recipe
    from dataclasses import asdict
    original['run'] = dict(asdict(pilot.pilot_run(recipe)))
    # The draw is independent of settings/source; prepare fresh then sample once.
    del original['sample']
    second = tmp_path/'second-plan.json'
    second.write_text(json.dumps(original))
    sample_plan(second, manifest, paths)
    before = previous.read_bytes(), second.read_bytes()
    pilot.assert_same_sample(second, previous)
    assert before == (previous.read_bytes(), second.read_bytes())
    forged = json.loads(previous.read_text())
    forged['sample']['k2_sha256']['dev'] = 'different'
    previous.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='differs from first pilot'):
        pilot.assert_same_sample(second, previous)


def test_rendered_recipes_keep_existing_leases_and_valid_shell(tmp_path: Path) -> None:
    for recipe in pilot.RECIPES:
        s = replace(settings(tmp_path), recipe=recipe)
        assert S.namespace(recipe, s.run_tag).endswith('-p2-'+recipe)
        for jobs in S.commands(s).values():
            for _, body in jobs:
                subprocess.run(['bash', '-n'], input=body, text=True, check=True)
        assert GPU in S.commands(s)['train'][0][1]
        assert SOLO_B in S.commands(s)['train'][1][1]


def test_preregistered_commands_have_valid_bash_syntax() -> None:
    import re
    path = Path(__file__).parents[1]/'PILOT2.md'
    blocks = re.findall(r'```bash\n(.*?)```', path.read_text(), re.S)
    assert len(blocks) == 4
    for block in blocks:
        subprocess.run(['bash', '-n'], input=block, text=True, check=True)
