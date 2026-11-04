import unicodedata

import pytest

from tesstrain_core.lao_classes import LAO_CLASSES, LAO_COMPONENT_MAP, fold_sara_am, get_component_type


def test_every_character_is_in_exactly_one_class():
    chars = [c for cls in LAO_COMPONENT_MAP.values() for c in cls]
    assert len(chars) == len(set(chars))


def test_modern_lao_block_is_fully_classified():
    # Letters and marks of modern Lao; Pali / Sanskrit / Khmu additions are left to 'other'
    pali_khmu = set('ຆຉຌຎຏຐຑຒຓຘຠຨຩຬໞໟ຺໎')
    for cp in range(0x0E80, 0x0F00):
        ch = chr(cp)
        try:
            unicodedata.name(ch)
        except ValueError:
            continue   # unassigned code point
        expected_other = ch in pali_khmu
        assert (get_component_type(ch) == 'other') == expected_other, ch


@pytest.mark.parametrize('ch, cls', [
    ('່', 'tone'), ('ິ', 'upper_vowel'), ('ກ', 'consonant'), ('ຽ', 'consonant'),
    ('ເ', 'before_vowel'), ('ຳ', 'rear_vowel'), ('ຼ', 'special'), ('ຸ', 'lower_vowel'),
    ('໕', 'lao_number'), ('5', 'digit'), ('Q', 'latin'), (' ', 'space'),
    ('ໆ', 'symbol'), ('ຯ', 'symbol'), ('.', 'symbol'), ('ຆ', 'other'), ('ก', 'other'),
])
def test_class_of(ch, cls):
    assert get_component_type(ch) == cls


def test_lao_classes_are_the_five_tiers():
    assert set(LAO_CLASSES) == {'tone', 'upper_vowel', 'consonant', 'before_vowel', 'rear_vowel',
                                'special', 'lower_vowel', 'lao_number'}


@pytest.mark.parametrize('text, folded', [
    ('ນໍາ', 'ນຳ'),
    ('ນໍ້າ', 'ນ້ຳ'),          # tone between ໍ and າ moves in front of ຳ
    ('ບໍ່', 'ບໍ່'),           # ໍ without າ is untouched
    ('ຄຳ', 'ຄຳ'),
    ('', ''),
])
def test_fold_sara_am(text, folded):
    assert fold_sara_am(text) == folded
