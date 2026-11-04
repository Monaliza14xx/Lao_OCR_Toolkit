"""Lao characters for CoER (component error rate) — the single source for every script.

Lao stacks in 5 tiers, and the classes follow them, because the marks above and
below the baseline are where OCR models fail most:

    tier 1  tone          tone marks                              ່ ້ ໊ ໋
    tier 2  upper_vowel   marks above the consonant               ັ ິ ີ ຶ ື ົ ໍ ໌
    tier 3  consonant     the 27 consonants (ກ–ຮ) plus ໜ ໝ and ຽ
            before_vowel  vowels written before the consonant     ເ ແ ໂ ໃ ໄ
            rear_vowel    vowels written after it                 ະ າ ຳ
    tier 4  special       subscript ລ                             ຼ
    tier 5  lower_vowel   vowels below the consonant              ຸ ູ

Non-Lao-script classes (all on the baseline):
    lao_number  ໐–໙    digit  0–9    latin  A–Z a–z
    space       the space character (Lao uses it between phrases, so spacing errors are common)
    symbol      punctuation and signs, incl. ໆ (repetition) and ຯ (abbreviation)
    other       anything else. This includes the Pali / Sanskrit / Khmu letters and marks
                (ຆ ຉ ... ໞ ໟ, ຺ ໎): modern official Lao doesn't use them, so seeing
                them in a prediction is itself a sign of an error.
"""

import string

LAO_COMPONENT_MAP = {
    'tone': ['່', '້', '໊', '໋'],
    'consonant': ['ກ', 'ຂ', 'ຄ', 'ງ', 'ຈ', 'ຊ', 'ຍ', 'ດ', 'ຕ', 'ຖ', 'ທ', 'ນ', 'ບ', 'ປ',
                  'ຜ', 'ຝ', 'ພ', 'ຟ', 'ມ', 'ຢ', 'ຣ', 'ລ', 'ວ', 'ສ', 'ຫ', 'ອ', 'ຮ', 'ໜ', 'ໝ', 'ຽ'],
    'lao_number': ['໐', '໑', '໒', '໓', '໔', '໕', '໖', '໗', '໘', '໙'],
    'digit': list('0123456789'),
    'latin': list(string.ascii_letters),
    'space': [' '],
    'symbol': ['.', ',', '?', '/', '!', ':', ';', '-', '+', '*', '=', '(', ')', '[', ']', '{', '}',
               '"', "'", '@', '#', '$', '%', '^', '&', '_', '|', '\\', '<', '>', '«', '»', '—',
               '…', '“', '”', '‘', '’', '•', '·', '°', '±', '×', '÷', '₭', '¥', '€', '£',
               'ໆ', 'ຯ'],
    'before_vowel': ['ແ', 'ໂ', 'ເ', 'ໄ', 'ໃ'],
    'rear_vowel': ['ະ', 'າ', 'ຳ'],
    'upper_vowel': ['ັ', 'ິ', 'ີ', 'ົ', 'ຶ', 'ື', '໌', 'ໍ'],
    'lower_vowel': ['ຸ', 'ູ'],
    'special': ['ຼ'],
}

_CHAR_TO_CAT = {ch: cat for cat, chars in LAO_COMPONENT_MAP.items() for ch in chars}
assert len(_CHAR_TO_CAT) == sum(map(len, LAO_COMPONENT_MAP.values())), 'a character is in two classes'

# Classes that describe Lao script itself (used e.g. to pick the "worst" class)
LAO_CLASSES = ['tone', 'upper_vowel', 'lower_vowel', 'before_vowel', 'rear_vowel',
               'consonant', 'special', 'lao_number']


def get_component_type(ch):
    return _CHAR_TO_CAT.get(ch, 'other')


def fold_sara_am(text):
    """ໍ (+ tone) + າ → (tone) + ຳ. Both spellings look the same; ຳ is the standard one."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == 'ໍ':
            j = i + 1
            tone = text[j] if j < n and '່' <= text[j] <= '໋' else ''
            j += len(tone)
            if j < n and text[j] == 'າ':
                out.append(tone + 'ຳ')
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return ''.join(out)
