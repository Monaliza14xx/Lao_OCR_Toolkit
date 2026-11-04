import os
import subprocess
import sys

import pytest
from conftest import ROOT

SCRIPTS = ['torch_train.py', 'gen_data_from_text.py', 'gen_documents.py', 'segment_document.py',
           'compare_line_stats.py', 'tesseract_tools.py']


@pytest.mark.parametrize('script', SCRIPTS)
def test_help_runs(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), '--help'],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
