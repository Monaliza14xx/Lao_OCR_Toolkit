#!/usr/bin/env python3
"""
build_augmented_corpus.py — Phase 3: Corpus Enrichment from Real-World Ground Truth

Merges real-world GT text into the existing training corpus, applies
character-frequency-weighted oversampling for under-represented Lao characters,
and generates line-length-stratified corpus splits matching the real-world
document distribution (68% body_long, 22% header_medium, 10% label_short).

Usage:
    python3 experiments/build_augmented_corpus.py \
        --real_gt /path/to/lao-real-documents \
        --base_corpus data/lao-corpus.txt \
        --output data/lao-corpus-augmented-v2.txt \
        --n_oversample 5
"""

import argparse
import random
from collections import Counter
from pathlib import Path


# ============================================================================
#  Lao Unicode ranges for character classification
# ============================================================================

_LAO_CONSONANTS = set(range(0x0E81, 0x0EAD))
_LAO_VOWELS     = {0x0EB1, 0x0EB3, 0x0EB4, 0x0EB5, 0x0EB6, 0x0EB7,
                   0x0EBB, 0x0EB8, 0x0EB9, 0x0EBC}
_LAO_TONES      = {0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB, 0x0ECC, 0x0ECD}
_ALL_LAO        = _LAO_CONSONANTS | _LAO_VOWELS | _LAO_TONES


def is_lao(ch):
    return ord(ch) in _ALL_LAO


def load_gt_texts(data_dir):
    """Load all .gt.txt files from a directory."""
    data_dir = Path(data_dir)
    texts = []
    for gt in sorted(data_dir.glob("*.gt.txt")):
        t = gt.read_text(encoding="utf-8").strip()
        if t:
            texts.append(t)
    return texts


def load_corpus(corpus_path):
    """Load text lines from a corpus file."""
    if not corpus_path or not Path(corpus_path).exists():
        return []
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                lines.append(t)
    return lines


def classify_line(text):
    """Classify line type based on character count."""
    n = len(text.strip())
    if n <= 15:
        return "label_short"
    if n <= 40:
        return "header_medium"
    return "body_long"


def build_char_frequency(lines):
    """Build character frequency from a list of lines."""
    freq = Counter()
    for line in lines:
        freq.update(line)
    return freq


def find_underrepresented(real_freq, corpus_freq, threshold=0.3):
    """
    Identify Lao characters that appear in real-world data but are
    under-represented in the training corpus (ratio < threshold).
    Returns a set of under-represented characters.
    """
    under = set()
    real_total = max(sum(real_freq.values()), 1)
    corpus_total = max(sum(corpus_freq.values()), 1)

    for ch, real_count in real_freq.items():
        if not is_lao(ch):
            continue
        real_rate = real_count / real_total
        corpus_rate = corpus_freq.get(ch, 0) / corpus_total
        if corpus_rate < real_rate * threshold:
            under.add(ch)
    return under


def oversample_lines(lines, underrepresented_chars, n_oversample=5):
    """
    Return additional copies of lines that contain under-represented characters.
    """
    extra = []
    for line in lines:
        if any(ch in underrepresented_chars for ch in line):
            extra.extend([line] * n_oversample)
    return extra


def stratified_sample(lines, target_dist, total_count):
    """
    Sample lines from the corpus to match a target line-type distribution.

    target_dist: dict like {'body_long': 0.68, 'header_medium': 0.22, 'label_short': 0.10}
    total_count: desired total output count
    """
    # Group by type
    groups = {"body_long": [], "header_medium": [], "label_short": []}
    for line in lines:
        t = classify_line(line)
        groups[t].append(line)

    result = []
    for line_type, fraction in target_dist.items():
        target_n = int(total_count * fraction)
        pool = groups.get(line_type, [])
        if not pool:
            # Fall back to all lines
            pool = lines
        # Sample with replacement if needed
        if len(pool) >= target_n:
            sampled = random.sample(pool, target_n)
        else:
            sampled = pool * (target_n // max(len(pool), 1) + 1)
            sampled = sampled[:target_n]
        result.extend(sampled)

    random.shuffle(result)
    return result


def deduplicate(lines):
    """Remove duplicate lines while preserving order."""
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(description="Enrich training corpus from real-world GT text")
    parser.add_argument("--real_gt", required=True,
                        help="Directory containing .gt.txt files from real-world documents")
    parser.add_argument("--base_corpus", default=None,
                        help="Existing training corpus text file (optional)")
    parser.add_argument("--output", required=True,
                        help="Output enriched corpus text file")
    parser.add_argument("--n_oversample", type=int, default=5,
                        help="How many times to repeat lines with under-represented chars (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*60}")
    print("  Phase 3: Corpus Enrichment")
    print(f"{'='*60}\n")

    # Load data
    real_texts = load_gt_texts(args.real_gt)
    base_lines = load_corpus(args.base_corpus)
    print(f"  Real-world GT lines: {len(real_texts)}")
    print(f"  Base corpus lines:   {len(base_lines)}")

    # Merge
    all_lines = deduplicate(base_lines + real_texts)
    print(f"  After merge+dedup:   {len(all_lines)} unique lines")

    # Character analysis
    real_freq = build_char_frequency(real_texts)
    corpus_freq = build_char_frequency(base_lines) if base_lines else Counter()
    under = find_underrepresented(real_freq, corpus_freq, threshold=0.3)

    print(f"\n  Real-world unique chars: {len(real_freq)}")
    print(f"  Under-represented Lao chars: {len(under)}")
    if under:
        print(f"    → Chars: {''.join(sorted(under))}")

    # Oversample lines containing under-represented chars
    extra = oversample_lines(all_lines, under, n_oversample=args.n_oversample)
    enriched = all_lines + extra
    print(f"\n  After oversampling ({args.n_oversample}×): {len(enriched)} lines")

    # Line-type distribution (real-world: 68% body_long, 22% header_medium, 10% label_short)
    real_type_counts = Counter(classify_line(t) for t in real_texts)
    real_total = len(real_texts)
    real_dist = {k: v / real_total for k, v in real_type_counts.items()}
    print("\n  Real-world line-type distribution:")
    for t, f in sorted(real_dist.items()):
        print(f"    {t}: {f:.0%}")

    # Save output (keep all lines — training will sample randomly)
    random.shuffle(enriched)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(enriched) + "\n", encoding="utf-8")

    # Also save metadata
    meta_path = output_path.with_suffix(".meta.txt")
    meta_lines = [
        "# Enriched corpus metadata",
        f"# Source real-world GT: {args.real_gt}",
        f"# Base corpus: {args.base_corpus}",
        f"# Total lines: {len(enriched)}",
        f"# Real-world lines: {len(real_texts)}",
        f"# Base corpus lines: {len(base_lines)}",
        f"# Under-represented chars ({len(under)}): {''.join(sorted(under)) if under else 'none'}",
        f"# Oversample factor: {args.n_oversample}×",
        "#",
        "# Real-world line-type distribution (use with --preset real-match):",
    ]
    for t, f in sorted(real_dist.items()):
        meta_lines.append(f"#   {t}: {f:.1%}")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"\n  ✓ Enriched corpus: {output_path}")
    print(f"  ✓ Metadata:        {meta_path}")
    print("\n  Usage:")
    print("    python3 gen_data_from_text.py \\")
    print(f"        --corpus {output_path} \\")
    print("        --output_dir data/lao-augmented-v2 \\")
    print("        --font tesstrain_core/font/Phetsarath-Regular.ttf \\")
    print("               tesstrain_core/font/Phetsarath-Bold.ttf \\")
    print("               tesstrain_core/font/phetsarath_ot.ttf \\")
    print("        --font_size_min 28 --font_size_max 72 \\")
    print("        --count 50000 \\")
    print("        --preset real-match \\")
    print("        --device cuda --batch_size 256 --ext jpg --to_lstmf")
    print()


if __name__ == "__main__":
    main()
