from tesstrain_core.report import format_coer, mark_diff, render_progress


def test_mark_diff_substitution_deletion_insertion():
    assert mark_diff('ປະຊາທິປະໄຕ', 'ປະຊາທີປະໄຕ') == 'ປະຊາ⟦ທິ→ທີ⟧ປະໄຕ'
    assert mark_diff('ກະຊວງ ແລະ ກິລາ', 'ກະຊວງ ກິລາ') == 'ກະຊວງ ⟦−ແລະ ⟧ກິລາ'
    assert mark_diff('ວັນທີ 16', 'ວັນທີ 16 ຕ') == 'ວັນທີ 16⟦+ ຕ⟧'


def test_mark_diff_never_splits_a_cluster():
    # the upper vowel differs, but the whole cluster is marked, not a lone mark
    out = mark_diff('ກິ', 'ກີ')
    assert out == '⟦ກິ→ກີ⟧'


def test_worst_class_and_ties():
    _, worst = format_coer({'tone': 0.12, 'consonant': 0.05, 'space': 0.9})
    assert worst == 'tone marks'      # space is not a Lao-script class
    _, worst = format_coer({'tone': 1.0, 'lower_vowel': 1.0, 'consonant': 0.5})
    assert worst == 'tone marks, lower vowels'
    _, worst = format_coer({'tone': 1.0, 'consonant': 1.0})
    assert worst is None              # nothing stands out


def test_render_progress_contains_the_essentials():
    text = render_progress(
        iteration=500, max_iter=10000, epoch=2, sps=312, elapsed=160, iters_done=500,
        loss=2.1, avg_loss=2.3, train={'cer': 0.45, 'wer': 0.78, 'acc': 0.12},
        evaluation={'loss': 2.4, 'cer': 0.48, 'wer': 0.80, 'acc': 0.11},
        coer={'tone': 0.12, 'consonant': 0.05}, is_best=True, mean_rms=0.04, lr=1e-4,
        samples=[('ກາ', 'ກາ'), ('ນ້ຳ', '')])
    for piece in ('iter 500/10,000', 'ETA', '★ best so far', 'CoER', 'tone 12.0%',
                  '✓ exact', '(empty)'):
        assert piece in text, piece


def test_render_progress_shows_strict_cer_and_cluer_when_given():
    text = render_progress(
        iteration=500, max_iter=1000, epoch=1, sps=100, elapsed=10, iters_done=500,
        loss=1.0, avg_loss=1.0, train={'cer': 0.1, 'wer': 0.2, 'acc': 0.3},
        evaluation={'loss': 1.0, 'cer': 0.1, 'wer': 0.2, 'acc': 0.3, 'cer_strict': 0.125, 'cluer': 0.2})
    assert 'strict CER 12.50%' in text and 'CluER 20.00%' in text
