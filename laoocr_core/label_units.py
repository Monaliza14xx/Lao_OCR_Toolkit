"""CTC label units: one label per code point (default) or per Lao grapheme cluster.

Experiment E3. Lao stacks tone marks, upper / lower vowels and subscript ລ on
a consonant. With code-point labels CTC has to emit 2–3 labels for one glyph column, in a fixed
order. With cluster labels the whole stack is one class, the idea of AksonOCR-Lite's Thai
recogniser (github.com/Remtir/AksonOCR-Lite).

Units (``--label_unit``):

    codepoint   one label per character                      ນ ້ ຳ          (B0, unchanged)
    cluster     consonant + its marks = one label            ນ້ ຳ
    cluster_am  as cluster, and a following ຳ joins it       ນ້ຳ
    open        cluster, but only clusters seen >= min_count times in the training text get
                their own label; rarer ones are spelled as code points

Clusters are split with lao_classes.split_clusters (the same units as CluER). Every unit also
keeps every single character of the training text in its charset, so any string that B0 could
write is still writable: a cluster that is not in the charset is spelled as its code points.
Under ``cluster`` / ``cluster_am`` such single marks get almost no training examples, so the
vocabulary is closed in practice; under ``open`` the rare clusters train them.

Decoding needs nothing new: the charset maps label strings to ids, and every decoder joins the
strings of the predicted ids. Text is not normalised here (same as B0), so the arms differ
only in the label unit.
"""
from collections import Counter

from .lao_classes import split_clusters

LABEL_UNITS = ('codepoint', 'cluster', 'cluster_am', 'open')
SARA_AM = 'ຳ'
DEFAULT_MIN_CLUSTER_COUNT = 20


def segment(text, unit='codepoint'):
    """Split text into label strings for `unit` (before the charset is consulted)."""
    if unit == 'codepoint':
        return list(text)
    if unit not in LABEL_UNITS:
        raise ValueError(f'unknown label unit {unit!r}; choose from {", ".join(LABEL_UNITS)}')
    parts = split_clusters(text)
    if unit != 'cluster_am':
        return parts
    out = []
    for p in parts:
        if p == SARA_AM and out and _is_base_cluster(out[-1]):
            out[-1] += p
        else:
            out.append(p)
    return out


def _is_base_cluster(s):
    """True for a consonant (+ marks), i.e. something ຳ can be written after."""
    from .lao_classes import get_component_type
    return get_component_type(s[0]) == 'consonant'


def build_charset(texts, unit='codepoint', min_count=DEFAULT_MIN_CLUSTER_COUNT):
    """{label string: id} for `texts` (ids from 1; 0 is the CTC blank), sorted for stable ids.

    Always contains every single character of the texts. 'cluster' / 'cluster_am' add every
    multi-character unit; 'open' only those seen >= min_count times.
    """
    chars, units = set(), Counter()
    for t in texts:
        chars.update(t)
        if unit != 'codepoint':
            units.update(u for u in segment(t, unit) if len(u) > 1)
    labels = set(chars)
    if unit in ('cluster', 'cluster_am'):
        labels.update(units)
    elif unit == 'open':
        labels.update(u for u, n in units.items() if n >= min_count)
    return {s: i + 1 for i, s in enumerate(sorted(labels))}


def infer_unit(charset):
    """Label unit of a saved charset (old checkpoints carry no 'label_unit' key).

    'cluster' and 'open' encode text the same way, so either answer gives the same labels.
    """
    multi = [k for k in charset if len(k) > 1]
    if not multi:
        return 'codepoint'
    return 'cluster_am' if any(k.endswith(SARA_AM) for k in multi) else 'cluster'


def encode(text, charset, unit='codepoint'):
    """Label ids for `text`, plus the number of characters that could not be written.

    A unit missing from the charset is spelled as its characters; characters missing from the
    charset are dropped (as B0 always did) and counted.
    """
    ids, dropped = [], 0
    for u in segment(text, unit):
        i = charset.get(u)
        if i is not None:
            ids.append(i)
            continue
        for ch in u:
            j = charset.get(ch)
            if j is None:
                dropped += 1
            else:
                ids.append(j)
    return ids, dropped
