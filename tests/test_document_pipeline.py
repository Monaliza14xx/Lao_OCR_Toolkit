"""End to end: synthetic letter page → layout analysis → labelled line crops."""
import os
import random

import numpy as np
import pytest
from conftest import FONTS
from PIL import features

from gen_documents import label_line
from tesstrain_core.doc_layout import deskew, segment_lines
from tesstrain_core.doc_synth import Content, FontSet, make_letter

LAO_FONT = os.path.join(FONTS, 'Phetsarath-Regular.ttf')
CORPUS = ['ກະຊວງສຶກສາທິການ ແລະ ກິລາ', 'ສາທາລະນະລັດ ປະຊາທິປະໄຕ ປະຊາຊົນລາວ',
          'ນ້ຳ ຂຶ້ນ ເຂົ້າ ຜູ້ ຫຼ້າ', 'ວັນທີ 16 ຕຸລາ 2026', 'ສັນຕິພາບ ເອກະລາດ ປະຊາທິປະໄຕ']


@pytest.fixture(scope='module')
def page():
    if not os.path.exists(LAO_FONT):
        pytest.skip('Lao font not available')
    if not features.check('raqm'):
        pytest.skip('Pillow without libraqm: letters need HarfBuzz shaping')
    rng = random.Random(3)
    np.random.seed(3)
    fonts = FontSet([LAO_FONT], [])
    img, words = make_letter(Content(CORPUS, rng), fonts, rng, dpi=150)
    return img, words


def _text_line_count(words):
    """Group drawn words into lines: a new line starts when the vertical centre jumps."""
    centres = sorted((w['box'][1] + w['box'][3]) / 2 for w in words)
    height = float(np.median([w['box'][3] - w['box'][1] for w in words]))
    return 1 + sum(b - a > 0.5 * height for a, b in zip(centres, centres[1:]))


def test_segmentation_finds_the_text_lines(page):
    img, words = page
    straight, angle = deskew(img)
    assert abs(angle) < 0.5                       # an unscanned page is already straight
    lines, _ = segment_lines(straight)
    expected = _text_line_count(words)
    # Blocks side by side on one row (header columns, signature blocks) become
    # separate crops, so there can be more crops than rows of text, never far fewer.
    assert 0.9 * expected <= len(lines) <= 1.5 * expected, (len(lines), expected)


def test_most_lines_get_a_label(page):
    img, words = page
    lines, info = segment_lines(img)
    margin = max(2, int(round(0.25 * info['core_height'])))
    labelled = 0
    for ln in lines:
        L, T, R, B = ln['box']
        crop = (L - margin, max(ln['limits'][0], T - margin), R + margin, min(ln['limits'][1], B + margin))
        text, _ = label_line(crop, ln['box'], words)
        labelled += text is not None
    assert labelled >= 0.7 * len(lines)


def test_label_line_rules():
    words = [{'text': 'ກາ', 'box': (10, 10, 50, 30)}, {'text': 'ນາ', 'box': (60, 10, 100, 30)}]
    assert label_line((0, 0, 120, 40), (10, 10, 100, 30), words) == ('ກາ ນາ', None)
    assert label_line((0, 0, 80, 40), (10, 10, 100, 30), words)[1] == 'cuts a word'
    assert label_line((200, 0, 300, 40), (200, 10, 300, 30), words)[1] == 'no words (graphic)'
