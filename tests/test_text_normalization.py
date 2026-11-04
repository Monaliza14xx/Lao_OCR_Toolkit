import pytest

from gen_data_from_text import normalize_lao_text


@pytest.mark.parametrize('raw, clean', [
    ('ກ່ີ', 'ກີ່'),               # tone mark stored after the vowel mark
    ('ນໍ້າ', 'ນ້ຳ'),               # ໍ + tone + າ  →  tone + ຳ
    ('ກ້ຼ', 'ກຼ້'),               # subscript ລ right after the consonant
    ('ຂ​ອງ', 'ຂອງ'),          # zero-width space removed
    ('  ສະ   ບາຍ  ', 'ສະ ບາຍ'),    # whitespace collapsed
    ('ຫຼ້າ', 'ຫຼ້າ'),               # already standard
])
def test_normalize(raw, clean):
    assert normalize_lao_text(raw) == clean
