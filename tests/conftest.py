"""Make the repository root importable (tesstrain_core, gen_data_from_text, ...)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FONTS = os.path.join(ROOT, 'fonts')
