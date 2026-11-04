import pytest

from tesstrain_core.metrics import (align_strings, compute_component_metrics, edit_distance,
                                    normalize_lao_text)


def rates(gt, pred):
    st = compute_component_metrics(align_strings(pred, gt))
    return {k: (v['errors'], v['total']) for k, v in st.items() if v['total'] or v['errors']}


def test_edit_distance_basic():
    assert edit_distance('ກາ', 'ກາ') == 0
    assert edit_distance('ກາ', 'ຂາ') == 1
    assert edit_distance(['a', 'b'], ['a']) == 1


def test_same_looking_sara_am_is_not_an_error():
    assert edit_distance('ນ້ໍາ', 'ນ້ຳ') == 0
    assert rates('ນ້ຳ', 'ນ້ໍາ')['rear_vowel'] == (0, 1)


def test_normalize_strips_zero_width_and_trailing_punctuation():
    assert normalize_lao_text('ຂ​ອງ.') == 'ຂອງ'


def test_component_errors_by_class():
    r = rates('ຂຶ້ນ', 'ຂືນ')           # wrong upper vowel, missing tone
    assert r['upper_vowel'] == (1, 1)
    assert r['tone'] == (1, 1)
    assert r['consonant'] == (0, 2)


def test_insertions_count_as_errors_of_the_inserted_class():
    r = rates('ກາ', 'ກາາ')
    assert r['rear_vowel'] == (1, 1)


@pytest.mark.parametrize('gt, pred, cls', [
    ('ສຽງ ດັງ', 'ສຽງດັງ', 'space'),
    ('Lao PDR', 'Lao PDK', 'latin'),
])
def test_new_classes_are_scored(gt, pred, cls):
    assert rates(gt, pred)[cls][0] == 1
