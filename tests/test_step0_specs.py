"""experiments/step0_specs.py: every Step 0 spec builds, and the commands are well formed."""
import importlib.util
import os
import shlex

import pytest
from conftest import ROOT

_spec = importlib.util.spec_from_file_location('step0_specs', os.path.join(ROOT, 'experiments', 'step0_specs.py'))
step0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step0)


@pytest.mark.parametrize('name, group, spec, _why', step0.CONFIGS, ids=[c[0] for c in step0.CONFIGS])
def test_every_spec_builds(name, group, spec, _why):
    params, rows, frames = step0.describe(spec)
    assert params > 0 and frames > 0
    if group == 'cnn':
        assert rows >= 8, f'{name}: only {rows} rows reach the collapse'
        assert frames == 170                       # CNN swaps keep B0's T = W/3


def test_b0_is_the_thesis_model():
    assert dict((c[0], c[2]) for c in step0.CONFIGS)['B0'] == \
        '[1,64,0,1 Ct5,5,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]'
    assert step0.describe(step0.CONFIGS[0][2])[0] == 929_264        # EXPERIMENT_PLAN.md §1b item 1


def test_names_are_unique_and_groups_are_covered():
    names = [c[0] for c in step0.CONFIGS]
    assert len(names) == len(set(names))
    assert {c[1] for c in step0.CONFIGS} == {'baseline', 'squash', 'rnn', 'cnn', 'xstride'}


def test_lcn4_backbone_is_about_1m_params():
    lcn4 = dict((c[0], c[2]) for c in step0.CONFIGS)['C-Lcn4']
    assert 0.9e6 < step0.describe(lcn4)[0] < 1.1e6


def test_commands_for_three_seeds(capsys):
    step0.main(['--commands', '--only', 'B0', 'rnn', '--extra', '--padding legacy'])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 * 3
    args = shlex.split(lines[0])
    assert args[:3] == ['python', 'torch_train.py', 'training']
    assert args[args.index('--net_spec') + 1].startswith('[1,64,0,1 ')
    assert {shlex.split(l)[shlex.split(l).index('--seed') + 1] for l in lines} == {'1', '2', '3'}
    assert lines[0].endswith('--padding legacy') and '--lr_scheduler plateau' in lines[0]


def test_table_lists_every_config(capsys):
    step0.main([])
    out = capsys.readouterr().out
    assert all(c[0] in out for c in step0.CONFIGS) and '929,264' in out
