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


# ── P0.2: strict CER and cluster error rate ──────────────────────────────────────────────
from tesstrain_core.lao_classes import split_clusters  # noqa: E402
from tesstrain_core.metrics import cluster_error_counts, levenshtein, strict_cer_counts  # noqa: E402


@pytest.mark.parametrize('text, clusters', [
    ('ກິ່ນເຂົ້າ', ['ກິ່', 'ນ', 'ເ', 'ຂົ້', 'າ']),
    ('ປຼູກ', ['ປຼູ', 'ກ']),                       # subscript ລ + lower vowel stay with the consonant
    ('ນ້ຳ 5', ['ນ້', 'ຳ', ' ', '5']),              # ຳ is a spacing vowel: its own unit
    ('່ກ', ['່', 'ກ']),                           # a mark with no base is its own unit
    ('', []),
])
def test_split_clusters(text, clusters):
    assert split_clusters(text) == clusters
    assert ''.join(clusters) == text


def test_strict_cer_is_plain_levenshtein():
    # lenient CER counts ດ/ຕ as 0.5 and strips the trailing dot; strict does neither
    assert edit_distance('ດາ.', 'ຕາ') == 0.5
    assert strict_cer_counts('ດາ.', 'ຕາ') == (2, 2)
    assert levenshtein('kitten', 'sitting') == 3
    assert levenshtein(['ກິ', 'ນ'], ['ກີ', 'ນ']) == 1


def test_cluer_counts_a_missing_tone_as_a_whole_cluster():
    gt, pred = 'ເຂົ້າໄປ', 'ເຂົາໄປ'                    # tone ້ missing
    errors, n = cluster_error_counts(pred, gt)
    assert (errors, n) == (1, 5)                    # ເ ຂົ້ າ ໄ ປ
    assert strict_cer_counts(pred, gt) == (1, 7)    # same error, diluted over 7 characters
    assert errors / n > 1 / 7


def test_cluer_two_errors_in_one_cluster_cost_one():
    assert cluster_error_counts('ກ', 'ກິ່') == (1, 1)
    assert strict_cer_counts('ກ', 'ກິ່') == (2, 3)
