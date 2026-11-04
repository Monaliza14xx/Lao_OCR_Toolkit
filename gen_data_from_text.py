#!/usr/bin/env python3
"""
gen_data_from_text.py — Synthetic OCR Training Data Generator

Renders text lines from a corpus file into .tif/.png/.jpg images with matching
.gt.txt ground truth files. Supports noise, blur, augmentation, .lstmf conversion,
and GPU-accelerated augmentations via PyTorch.

Usage:
    # Basic: render text to images (CPU)
    python3 gen_data_from_text.py \
        --corpus corpus.txt \
        --output_dir data/lao-ground-truth \
        --font /path/to/LaoFont.ttf \
        --count 10000

    # GPU-accelerated augmentation (recommended for large datasets)
    python3 gen_data_from_text.py \
        --corpus corpus.txt \
        --output_dir data/lao-ground-truth \
        --font /path/to/LaoFont.ttf \
        --count 50000 \
        --noise 0.3 --blur 0.3 --augment 0.5 \
        --device cuda --batch_size 256 \
        --ext png --to_lstmf

    # CPU multiprocessing (alternative)
    python3 gen_data_from_text.py \
        --corpus corpus.txt \
        --output_dir data/lao-ground-truth \
        --font font1.ttf font2.ttf font3.ttf \
        --count 100000 \
        --noise 0.4 --blur 0.3 --augment 0.6 \
        --device cpu --to_lstmf --workers 8
"""

import argparse
import io
import math
import multiprocessing
import os
import random
import re
import struct
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from PIL import features as _pil_features


# ============================================================================
#  Text Rendering
# ============================================================================

def load_corpus(corpus_path, max_line_chars=200):
    """Load text lines from a UTF-8 corpus and normalize Lao orthography.

    Lines that are empty after normalization or longer than max_line_chars
    are skipped.
    """
    # Strict UTF-8 only: utf-16 / latin-1 fallbacks turn Lao bytes into mojibake.
    for encoding, errors in (('utf-8-sig', 'strict'), ('utf-8', 'replace')):
        try:
            lines = []
            with open(corpus_path, 'r', encoding=encoding, errors=errors) as f:
                for line in f:
                    line = normalize_lao_text(line)
                    if line and len(line) <= max_line_chars:
                        lines.append(line)
            if errors == 'replace':
                print(f"  Warning: malformed UTF-8 in {corpus_path}; bad bytes were replaced.")
            print(f"  Loaded corpus {corpus_path} ({len(lines)} lines)")
            return lines
        except UnicodeError:
            continue
    return []


# Lao codepoints that attach to the preceding base letter (no cursor advance of
# their own). ຳ (AM) is a spacing letter in Unicode, but its ring sits over the
# base, so it is drawn together with the base in the fallback renderer.
_LAO_COMBINING = frozenset([
    0x0EB1,  # MAI KAN        (above)
    0x0EB3,  # AM             (spacing, ring over base)
    0x0EB4, 0x0EB5, 0x0EB6, 0x0EB7,  # I, II, Y, YY (above)
    0x0EBB,  # MAI KON        (above)
    0x0EB8, 0x0EB9,  # U, UU  (below)
    0x0EBC,  # SEMIVOWEL LO   (below)
    0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB,  # tone marks (above)
    0x0ECC,  # CANCELLATION MARK (above)
    0x0ECD,  # NIGGAHITA      (above)
])

_LAO_SUBSCRIPT_LO = '\u0EBC'
# Above/below vowel signs, including NIGGAHITA (ໍ), which is a vowel, not a tone.
_LAO_VOWEL_MARKS = frozenset([0x0EB1, 0x0EB4, 0x0EB5, 0x0EB6, 0x0EB7, 0x0EBB,
                              0x0EB8, 0x0EB9, 0x0EBC, 0x0ECD])
_LAO_TONE_MARKS = frozenset([0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB, 0x0ECC])
_LAO_AM = '\u0EB3'
_ZERO_WIDTH = ('\u200b', '\u200c', '\u200d', '\u2060', '\ufeff')


def _split_grapheme_clusters(text):
    """Split text into clusters: one base character plus its Lao combining marks.

    Whitespace and non-Lao characters each form their own cluster.
    """
    clusters = []
    current = ''
    for ch in text:
        if ord(ch) in _LAO_COMBINING and current and not current.isspace():
            current += ch
        else:
            if current:
                clusters.append(current)
            current = ch
    if current:
        clusters.append(current)
    return clusters


def normalize_lao_text(text):
    """Return text in the standard Lao storage order used by real typed documents.

    - NFC, zero-width characters removed, whitespace collapsed and trimmed.
    - ໍ + າ typed as two characters (optionally with a tone mark between them)
      becomes the single letter ຳ, which renders identically.
    - Inside each cluster: base, subscript ຼ, vowel marks (incl. ໍ), tone marks,
      then ຳ. Tone marks are stored BEFORE ຳ, e.g. ນ້ຳ = ນ + ້ + ຳ, which is how
      Lao keyboards and real ground truth encode it.
    """
    if not text:
        return ''
    text = unicodedata.normalize('NFC', text)
    for zw in _ZERO_WIDTH:
        text = text.replace(zw, '')
    text = re.sub(r'\u0ECD([\u0EC8-\u0ECB]?)\u0EB2', '\\1\u0EB3', text)
    text = re.sub(r'\s+', ' ', text).strip()

    out = []
    for cluster in _split_grapheme_clusters(text):
        if len(cluster) > 1:
            base, marks = cluster[0], cluster[1:]
            vowels = [c for c in marks if ord(c) in _LAO_VOWEL_MARKS]
            vowels.sort(key=lambda c: c != _LAO_SUBSCRIPT_LO)  # ຼ first, rest keep order
            tones = [c for c in marks if ord(c) in _LAO_TONE_MARKS]
            am = [c for c in marks if c == _LAO_AM]
            cluster = base + ''.join(vowels) + ''.join(tones) + ''.join(am)
        out.append(cluster)
    return ''.join(out)


# ----------------------------------------------------------------------------
#  Fonts
# ----------------------------------------------------------------------------

_HAS_RAQM = _pil_features.check('raqm')
# Words with stacked marks: a font whose HarfBuzz output for these is much wider
# than the same word without marks has broken mark positioning.
_SHAPING_PROBES = ('ຂຶ້ນ', 'ເຂົ້າ', 'ນ້ຳ', 'ກີ່', 'ຜູ້', 'ຫຼ້າ')


@lru_cache(maxsize=None)
def _load_font(path, size, shaped):
    engine = ImageFont.Layout.RAQM if shaped else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=engine)


@lru_cache(maxsize=None)
def _font_cmap(path):
    """Codepoints the font maps, or None when fontTools isn't installed."""
    try:
        from fontTools.ttLib import TTFont
        return frozenset(TTFont(path, lazy=True, fontNumber=0).getBestCmap())
    except Exception:
        return None


def font_covers(path, text):
    """True if the font has a glyph for every non-space character in text."""
    cmap = _font_cmap(path)
    return cmap is None or all(ord(c) in cmap for c in text if not c.isspace())


@lru_cache(maxsize=None)
def font_shapes_correctly(path):
    """True if HarfBuzz places stacked Lao marks sanely with this font.

    Some fonts (e.g. phetsarath_ot.ttf) have mark tables that push a tone mark
    over a vowel ~1300 px to the right under HarfBuzz; those use the fallback
    cluster renderer instead.
    """
    if not _HAS_RAQM:
        return False
    try:
        f = _load_font(path, 40, True)
        for word in _SHAPING_PROBES:
            base = ''.join(c for c in word if unicodedata.category(c) != 'Mn')
            if f.getlength(word) > f.getlength(base) + 8:
                return False
        return True
    except Exception:
        return False


def _is_lao(ch):
    return 0x0E80 <= ord(ch) <= 0x0EFF


def _script_runs(text):
    """Split text into [is_lao, run] pieces; spaces join the run they follow."""
    runs = []
    for ch in text:
        lao = runs[-1][0] if (ch.isspace() and runs) else (_is_lao(ch) or ch.isspace())
        if runs and runs[-1][0] == lao:
            runs[-1][1] += ch
        else:
            runs.append([lao, ch])
    return runs


# ----------------------------------------------------------------------------
#  Line rendering
# ----------------------------------------------------------------------------

def render_line(text, font_path, size, sec_font_path=None, bg_color=255, fg_color=0):
    """Render one text line to a grayscale image.

    Lao runs use font_path; digits, Latin and punctuation use sec_font_path when
    it has the glyphs (the primary font otherwise). All runs share one baseline.

    Returns None when this font cannot render the text faithfully (a missing
    glyph, or a cluster that draws no ink), so the caller can try another font
    instead of saving an image that doesn't match its label.
    """
    if not text or text.isspace():
        return None
    sec_font_path = sec_font_path or font_path
    plan = []
    for lao, run in _script_runs(text):
        path = font_path if lao else sec_font_path
        if not font_covers(path, run):
            if font_covers(font_path, run):
                path = font_path
            else:
                return None
        plan.append((lao, run, path))

    shaped = font_shapes_correctly(font_path)
    ascent, descent = _load_font(font_path, size, shaped).getmetrics()
    margin = 2 * size
    canvas_h = ascent + descent + 2 * margin
    baseline = margin + ascent

    if shaped:
        fonts = [_load_font(p, size, _HAS_RAQM) for _, _, p in plan]
        widths = [f.getlength(run) for f, (_, run, _) in zip(fonts, plan)]
        canvas = Image.new('L', (int(sum(widths)) + 2 * margin, canvas_h), bg_color)
        draw = ImageDraw.Draw(canvas)
        x = margin
        for f, (lao, run, _), w in zip(fonts, plan, widths):
            draw.text((x, baseline), run, font=f, fill=fg_color, anchor='ls',
                      language='lo' if lao else None)
            x += w
    else:
        canvas = _render_clusters(plan, size, canvas_h, baseline, margin, bg_color, fg_color)
        if canvas is None:
            return None

    return _crop_line(canvas, baseline, ascent, descent, size, bg_color)


def _crop_line(canvas, baseline, ascent, descent, size, bg_color):
    """Crop to the ink horizontally and to the font's line box vertically.

    Cropping vertically to the font's ascent/descent (not to the ink) keeps the
    letter size the same whether or not a line has upper/lower marks; the
    trainer rescales every line to a fixed height. Margins are randomized.
    """
    arr = np.array(canvas)
    ink = arr < (bg_color - 55)
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    if cols.size == 0:
        return None
    top = min(baseline - ascent, int(rows[0]))
    bottom = max(baseline + descent, int(rows[-1]) + 1)
    pad_v = lambda: int(round(random.uniform(0.03, 0.15) * size))
    pad_h = lambda: int(round(random.uniform(0.1, 0.6) * size))
    w, h = canvas.size
    return canvas.crop((max(0, int(cols[0]) - pad_h()), max(0, top - pad_v()),
                        min(w, int(cols[-1]) + 1 + pad_h()), min(h, bottom + pad_v())))


def _draw_cluster_and_realign(cluster, font, canvas_h, scratch_w, y_baseline, bg_color, fg_color):
    """Draw a grapheme cluster on a scratch canvas and pull runaway marks back.

    Without working shaping, some fonts (e.g. phetsarath_ot.ttf) place stacked
    marks (tone above vowel, as in 'ຂຶ້') at wild horizontal offsets. Ink islands
    that land far right of the base letter are moved back above its centre.
    """
    scratch = Image.new('L', (scratch_w, canvas_h), bg_color)
    ImageDraw.Draw(scratch).text((0, y_baseline), cluster, font=font, fill=fg_color, anchor='ls')
    s_arr = np.array(scratch)

    ink_cols = np.where(np.any(s_arr < (bg_color - 15), axis=0))[0]
    if ink_cols.size == 0:
        return scratch, s_arr

    # Contiguous horizontal ink islands (gaps > 5 px split them)
    islands = []
    start = ink_cols[0]
    for idx in range(1, len(ink_cols)):
        if ink_cols[idx] > ink_cols[idx - 1] + 5:
            islands.append((start, ink_cols[idx - 1]))
            start = ink_cols[idx]
    islands.append((start, ink_cols[-1]))

    if len(islands) > 1:
        isl0_l, isl0_r = islands[0]
        c0 = isl0_l + (isl0_r - isl0_l + 1) // 2
        recomp = np.full((canvas_h, scratch_w), bg_color, dtype=np.uint8)
        recomp[:, isl0_l:isl0_r + 1] = s_arr[:, isl0_l:isl0_r + 1]
        for isl_l, isl_r in islands[1:]:
            if isl_l - isl0_r > 15:  # runaway mark: centre it over the base
                w_i = isl_r - isl_l + 1
                t_l = max(0, c0 - w_i // 2)
                t_r = min(scratch_w - 1, t_l + w_i - 1)
                aw = t_r - t_l + 1
                recomp[:, t_l:t_r + 1] = np.minimum(recomp[:, t_l:t_r + 1], s_arr[:, isl_l:isl_l + aw])
            else:
                recomp[:, isl_l:isl_r + 1] = np.minimum(recomp[:, isl_l:isl_r + 1], s_arr[:, isl_l:isl_r + 1])
        s_arr = recomp
        scratch = Image.fromarray(s_arr)
    return scratch, s_arr


def _render_clusters(plan, size, canvas_h, baseline, margin, bg_color, fg_color):
    """Fallback renderer for fonts whose HarfBuzz output is broken.

    Draws one grapheme cluster at a time (base + marks together) so marks can't
    drift, then advances by the font's advance width plus light tracking.
    Returns None if any cluster draws no ink (its label would be wrong).
    """
    tracking = max(1, int(round(size * 0.02)))
    scratch_w = 4000  # wide enough for runaway marks before they are realigned
    pieces = []
    x = margin
    for _, run, path in plan:
        font = _load_font(path, size, False)
        space_w = font.getlength(' ') or size * 0.3
        for cluster in _split_grapheme_clusters(run):
            if cluster.isspace():
                x += space_w * len(cluster)
                continue
            scratch, s_arr = _draw_cluster_and_realign(cluster, font, canvas_h, scratch_w,
                                                       baseline, bg_color, fg_color)
            ink_cols = np.where(np.any(s_arr < (bg_color - 15), axis=0))[0]
            if ink_cols.size == 0:
                return None
            ink_l, ink_r = int(ink_cols[0]), int(ink_cols[-1]) + 1
            pieces.append((int(round(x)) - ink_l, s_arr[:, :ink_r]))
            ink_w = ink_r - ink_l
            adv = font.getlength(cluster)
            if adv < 0 or adv > max(ink_w * 2 + 15, size * 1.5):  # broken advance
                adv = ink_w + 2
            x += max(adv, ink_w + 1) + tracking

    width = int(x) + margin
    canvas = np.full((canvas_h, width), bg_color, dtype=np.uint8)
    for left, piece in pieces:
        l0 = max(0, left)
        r0 = min(width, left + piece.shape[1])
        if r0 > l0:
            canvas[:, l0:r0] = np.minimum(canvas[:, l0:r0], piece[:, l0 - left:r0 - left])
    return Image.fromarray(canvas)


# ============================================================================
#  Augmentation Pipeline
# ============================================================================

def add_gaussian_noise(img, intensity=0.05):
    """Add Gaussian noise to image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_salt_pepper_noise(img, amount=0.02):
    """Add salt-and-pepper noise."""
    arr = np.array(img, dtype=np.uint8)
    # Salt
    n_salt = int(amount * arr.size * 0.5)
    coords = tuple(np.random.randint(0, d, n_salt) for d in arr.shape)
    arr[coords] = 255
    # Pepper
    n_pepper = int(amount * arr.size * 0.5)
    coords = tuple(np.random.randint(0, d, n_pepper) for d in arr.shape)
    arr[coords] = 0
    return Image.fromarray(arr)


def add_blur(img, radius=1.0):
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _background_level(img):
    """Median of the border pixels: the paper colour to fill new areas with."""
    arr = np.asarray(img)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return int(np.median(border))


def random_rotation(img, max_angle=2.0):
    """Slight random rotation (skew), filled with the image's own background.

    The angle is capped so the line's vertical drift is at most its own height:
    a long line rotated by a few degrees would otherwise become mostly empty
    space and shrink the text when the trainer rescales it to a fixed height.
    """
    w, h = img.size
    max_angle = min(max_angle, math.degrees(math.atan(h / max(w, 1))))
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=_background_level(img))


def random_scale(img, scale_range=(0.9, 1.1)):
    """Random scale."""
    scale = random.uniform(*scale_range)
    w, h = img.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def random_erosion_dilation(img, prob=0.3):
    """Simulate ink thickness variation (bleed) via morphological erosion."""
    if random.random() > prob:
        return img
    # Only apply MinFilter (erosion of bright background / thickening of dark ink)
    # to prevent erasing delicate Lao combining marks (tone marks, top vowels).
    return img.filter(ImageFilter.MinFilter(3))


def random_brightness_contrast(img, brightness_range=(-20, 20), contrast_range=(0.8, 1.2)):
    """Random brightness and contrast adjustment."""
    arr = np.array(img, dtype=np.float32)
    # Contrast
    contrast = random.uniform(*contrast_range)
    arr = (arr - 127.5) * contrast + 127.5
    # Brightness
    brightness = random.uniform(*brightness_range)
    arr += brightness
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def random_perspective(img, distortion=0.05):
    """Slight perspective distortion."""
    w, h = img.size
    d = distortion * min(w, h)
    pts1 = [(0, 0), (w, 0), (w, h), (0, h)]
    pts2 = [
        (random.uniform(0, d), random.uniform(0, d)),
        (w - random.uniform(0, d), random.uniform(0, d)),
        (w - random.uniform(0, d), h - random.uniform(0, d)),
        (random.uniform(0, d), h - random.uniform(0, d)),
    ]
    # Find perspective coefficients
    coeffs = _find_perspective_coeffs(pts2, pts1)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                         fillcolor=_background_level(img))


def _find_perspective_coeffs(src, dst):
    """Compute perspective transform matrix."""
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for p in src for c in p], dtype=np.float64)
    try:
        res = np.linalg.solve(A, B)
        return tuple(res.flatten())
    except np.linalg.LinAlgError:
        return (1, 0, 0, 0, 1, 0, 0, 0)


def add_background_texture(img, prob=0.3):
    """Add subtle paper-like background texture."""
    if random.random() > prob:
        return img
    arr = np.array(img, dtype=np.float32)
    # Generate low-frequency noise as paper texture
    texture = np.random.normal(0, 5, arr.shape).astype(np.float32)
    arr = np.clip(arr + texture, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_image(img, noise_prob=0.3, blur_prob=0.3, augment_prob=0.5,
                  noise_intensity=0.04, blur_radius=0.9,
                  brightness_max=20.0, contrast_max=0.2, morph_prob=0.3,
                  rotation_max_angle=1.5):
    """Apply random augmentations to an image."""
    # Always-optional augmentations
    if rotation_max_angle > 0 and random.random() < augment_prob:
        img = random_rotation(img, max_angle=rotation_max_angle)
    if random.random() < augment_prob:
        img = random_brightness_contrast(img, 
              brightness_range=(-brightness_max, brightness_max),
              contrast_range=(1.0 - contrast_max, 1.0 + contrast_max))
    if random.random() < augment_prob * 0.5:
        img = random_perspective(img, distortion=0.03)
    if random.random() < augment_prob * 0.5:
        img = random_scale(img, scale_range=(0.93, 1.07))
    if random.random() < augment_prob * morph_prob:
        img = random_erosion_dilation(img, prob=1.0)
    if random.random() < augment_prob * 0.5:
        img = add_background_texture(img, prob=1.0)

    # Noise
    if random.random() < noise_prob:
        if random.random() > 0.5:
            # Vary slightly around target intensity
            intensity = random.uniform(noise_intensity * 0.5, noise_intensity * 1.5)
            img = add_gaussian_noise(img, intensity=intensity)
        else:
            amount = random.uniform(noise_intensity * 0.2, noise_intensity * 0.8)
            img = add_salt_pepper_noise(img, amount=amount)

    # Blur
    if random.random() < blur_prob:
        radius = random.uniform(blur_radius * 0.5, blur_radius * 1.5)
        img = add_blur(img, radius=radius)

    return img


# ============================================================================
#  Real-World-Calibrated Augmentation Functions (Phase 2)
# ============================================================================

def add_jpeg_artifacts(img, quality_range=(65, 90)):
    """
    Simulate JPEG scan compression artifacts by re-encoding at a random quality.
    This introduces blocking, ringing, and chroma subsampling noise that matches
    real scanned document JPEG characteristics.
    """
    import io as _io
    quality = random.randint(*quality_range)
    buf = _io.BytesIO()
    # Convert to RGB for JPEG (JPEG requires 3-channel for authentic artifact simulation)
    rgb = img.convert('RGB')
    rgb.save(buf, format='JPEG', quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert('L')


def add_uneven_illumination(img, max_gradient=0.12):
    """
    Simulate scanner light falloff by applying a smooth brightness gradient.
    Real scanners have slightly uneven illumination (edges darker or lighter
    than the centre). Uses a bilinear gradient in a random direction.
    max_gradient: maximum relative brightness change across the image (0.12 = 12%).
    """
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # Random gradient direction and magnitude
    direction = random.choice(['left-right', 'right-left', 'top-bottom', 'bottom-top',
                               'radial-center', 'radial-edge'])
    magnitude = random.uniform(max_gradient * 0.3, max_gradient)

    if direction in ('left-right', 'right-left'):
        grad = np.linspace(0, 1, w, dtype=np.float32)
        if direction == 'right-left':
            grad = grad[::-1]
        gradient = np.tile(grad, (h, 1))
    elif direction in ('top-bottom', 'bottom-top'):
        grad = np.linspace(0, 1, h, dtype=np.float32)
        if direction == 'bottom-top':
            grad = grad[::-1]
        gradient = np.tile(grad[:, None], (1, w))
    else:
        # Radial gradient
        cx, cy = w / 2, h / 2
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        gradient = dist / (dist.max() + 1e-9)
        if direction == 'radial-center':
            gradient = 1 - gradient

    # Apply: gradient shifts toward white (255) or away, scaled by pixel brightness
    # so reflective white paper experiences falloff while dark ink remains dark.
    shift = (gradient - 0.5) * magnitude * 255 * (arr / 255.0)
    arr = np.clip(arr + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_colored_ink(img, color='red', fg_threshold=100):
    """
    Convert grayscale ink to a specified color (red, blue, dark-green, dark-gray).
    Background remains white/light. This simulates colored stamp ink, official
    document header colors, and colored signature lines common in Lao admin docs.
    """
    COLOR_MAP = {
        'red':        (180, 20,  20),
        'dark-red':   (140, 10,  10),
        'blue':       (20,  40,  180),
        'dark-blue':  (10,  20,  120),
        'dark-green': (10,  100, 30),
        'dark-gray':  (50,  50,  50),
    }
    target_rgb = COLOR_MAP.get(color, (0, 0, 0))
    arr_gray = np.array(img, dtype=np.float32)

    # Create RGB output starting from white
    out = np.ones((arr_gray.shape[0], arr_gray.shape[1], 3), dtype=np.float32) * 255

    # Ink mask: dark pixels in grayscale become the target color
    ink_mask = arr_gray < fg_threshold
    # Blend: darkest ink → target color; lighter ink → lighter tint of target
    alpha = np.clip((fg_threshold - arr_gray) / fg_threshold, 0, 1)  # 0=bg, 1=darkest
    for c, val in enumerate(target_rgb):
        out[:, :, c] = np.where(
            ink_mask,
            val * alpha + 255 * (1 - alpha),  # blend from white to target color
            arr_gray  # background: keep original gray
        )
    # Grayscale the final output (OCR models train on L-mode)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).convert('L')


def add_real_paper_texture(img, bg_mean=252.8, bg_std=6.4, texture_scale=1.0):
    """
    Add realistic paper grain texture calibrated from real-world measurements.
    Uses a Perlin-like approximation (sum of scaled Gaussian noise) to simulate
    the low-frequency variation of real paper rather than pure white noise.

    bg_mean: mean background brightness of real paper (from Phase 1: 252.8)
    bg_std:  std of real paper background (from Phase 1: 6.4)
    """
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # Low-frequency base texture (paper grain)
    # Multiple scales to create realistic grain
    texture = np.zeros((h, w), dtype=np.float32)
    for scale in [4, 8, 16]:
        sh, sw = max(1, h // scale), max(1, w // scale)
        low_freq = np.random.normal(0, bg_std * 0.5, (sh, sw))
        # Upscale via PIL for smooth interpolation
        low_pil = Image.fromarray(low_freq.astype(np.float32)).resize((w, h), Image.BILINEAR)
        texture += np.array(low_pil)

    texture = texture / 3.0  # average across scales

    # Only apply texture to background regions (bright pixels)
    bg_mask = arr > 200
    bg_current_mean = arr[bg_mask].mean() if bg_mask.any() else 255.0
    # Shift background toward real paper mean
    mean_shift = bg_mean - bg_current_mean
    arr = np.where(bg_mask, arr + mean_shift + texture * texture_scale, arr)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_scan_noise_profile(img, noise_sigma=11.75, noise_prob=0.7):
    """
    Apply noise calibrated from real scan measurements (Phase 1: σ=11.748).
    Uses a mix of Gaussian and correlated noise to match scan sensor characteristics,
    which differ from pure i.i.d. Gaussian noise in synthetic generation.
    """
    if random.random() > noise_prob:
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # 70% Gaussian, 30% correlated (row/column sensor noise)
    if random.random() < 0.7:
        noise = np.random.normal(0, noise_sigma, (h, w)).astype(np.float32)
    else:
        # Correlated noise: row-wise and column-wise sensor drift
        row_noise = np.random.normal(0, noise_sigma * 0.5, (h, 1))
        col_noise = np.random.normal(0, noise_sigma * 0.5, (1, w))
        pixel_noise = np.random.normal(0, noise_sigma * 0.5, (h, w))
        noise = (row_noise + col_noise + pixel_noise).astype(np.float32)

    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================================
#  Augmentation Presets (Phase 2)
# ============================================================================

# Each preset is a dict of keyword arguments for augment_image() plus
# extra flags consumed by apply_preset_augmentation().
_AUGMENTATION_PRESETS = {
    'clean': {
        '_description': 'Clean synthetic output — no augmentation (baseline)',
        'noise_prob': 0.0, 'blur_prob': 0.0, 'augment_prob': 0.0,
        'noise_intensity': 0.0, 'blur_radius': 0.0,
        'brightness_max': 0.0, 'contrast_max': 0.0, 'morph_prob': 0.0,
        # Extra real-world augmentation flags
        'jpeg_quality_range': None,
        'uneven_illumination_prob': 0.0,
        'colored_ink_prob': 0.0,
        'paper_texture_prob': 0.0,
        'bg_mean': 255.0, 'bg_std': 0.0,
        'rotation_max_angle': 0.0,
    },
    'scan-mild': {
        '_description': 'Light scan degradation — good quality scanned documents',
        'noise_prob': 0.5, 'blur_prob': 0.2, 'augment_prob': 0.4,
        'noise_intensity': 0.025, 'blur_radius': 0.5,
        'brightness_max': 10.0, 'contrast_max': 0.1, 'morph_prob': 0.15,
        'jpeg_quality_range': [80, 92],
        'uneven_illumination_prob': 0.2,
        'colored_ink_prob': 0.01,
        'paper_texture_prob': 0.4,
        'bg_mean': 250.0, 'bg_std': 4.0,
        'rotation_max_angle': 2.0,
    },
    'scan-heavy': {
        '_description': 'Heavy scan degradation — aged/low-quality scanned documents',
        'noise_prob': 0.85, 'blur_prob': 0.6, 'augment_prob': 0.7,
        'noise_intensity': 0.07, 'blur_radius': 1.5,
        'brightness_max': 30.0, 'contrast_max': 0.35, 'morph_prob': 0.4,
        'jpeg_quality_range': [50, 75],
        'uneven_illumination_prob': 0.5,
        'colored_ink_prob': 0.03,
        'paper_texture_prob': 0.7,
        'bg_mean': 240.0, 'bg_std': 12.0,
        'rotation_max_angle': 5.0,
    },
    'real-match': {
        '_description': 'Parameters calibrated from Phase 1 analysis of 50 real Lao document lines',
        # Calibrated from analyze_real_data.py output
        'noise_prob': 0.70,
        'blur_prob': 0.20,           # Real docs are sharp (all 50 classified as sharp)
        'augment_prob': 0.60,
        'noise_intensity': 0.0461,   # σ=11.75 / 255 = 0.0461
        'blur_radius': 0.0,          # Real images are uniformly sharp
        'brightness_max': 3.5,       # Controlled brightness range to keep text contrast sharp
        'contrast_max': 0.138,       # Calibrated from contrast gap
        'morph_prob': 0.15,          # Reduced probability so thin diacritics stay crisp
        'jpeg_quality_range': [65, 90],
        'uneven_illumination_prob': 0.30,
        'colored_ink_prob': 0.02,    # 1/50 = 2% red ink
        'paper_texture_prob': 0.50,
        'bg_mean': 252.8,            # Real paper mean background (Phase 1)
        'bg_std': 6.4,               # Real paper std (Phase 1)
        'rotation_max_angle': 3.96,  # Calibrated from skew range ±1.97°
    },
}


_BASE_AUG_KEYS = ('noise_prob', 'blur_prob', 'augment_prob', 'noise_intensity',
                  'blur_radius', 'brightness_max', 'contrast_max', 'morph_prob')


def preset_base_params(preset_name):
    """The augment_image() keyword arguments a preset sets."""
    p = _AUGMENTATION_PRESETS[preset_name]
    return {k: p[k] for k in _BASE_AUG_KEYS}


def apply_preset_augmentation(img, preset_name):
    """
    Apply a named augmentation preset to a PIL image.
    Runs base augment_image() first (without rotation), then the real-world
    augmentations, then one final rotation with the preset's angle.
    """
    if preset_name not in _AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. "
                         f"Choose from: {list(_AUGMENTATION_PRESETS.keys())}")
    img = augment_image(img, rotation_max_angle=0, **preset_base_params(preset_name))
    return apply_realworld_augmentations(img, preset_name)


def apply_realworld_augmentations(img, preset_name):
    """Paper texture, uneven lighting, coloured ink, JPEG, scan noise, then skew."""
    p = _AUGMENTATION_PRESETS[preset_name]

    # 2. Real-world augmentations
    if p.get('paper_texture_prob', 0) > 0 and random.random() < p['paper_texture_prob']:
        img = add_real_paper_texture(
            img,
            bg_mean=p.get('bg_mean', 252.8),
            bg_std=p.get('bg_std', 6.4),
        )

    if p.get('uneven_illumination_prob', 0) > 0 and random.random() < p['uneven_illumination_prob']:
        img = add_uneven_illumination(img)

    if p.get('colored_ink_prob', 0) > 0 and random.random() < p['colored_ink_prob']:
        color = random.choice(['red', 'dark-red', 'blue'])
        img = add_colored_ink(img, color=color)

    if p.get('jpeg_quality_range') is not None and random.random() < 0.5:
        img = add_jpeg_artifacts(img, quality_range=p['jpeg_quality_range'])

    # 3. Calibrated scan noise (applied after JPEG to simulate real sensor+compression stack)
    if p.get('noise_intensity', 0) > 0:
        img = add_scan_noise_profile(img, noise_sigma=p['noise_intensity'] * 255 * 0.5,
                                     noise_prob=p['noise_prob'] * 0.5)

    # 4. Rotation (done last to keep background fill correct)
    max_rot = p.get('rotation_max_angle', 1.5)
    if max_rot > 0 and p.get('augment_prob', 0) > 0 and random.random() < p['augment_prob']:
        img = random_rotation(img, max_angle=max_rot)

    return img


# ============================================================================
#  GPU-Accelerated Augmentation Pipeline (PyTorch)
# ============================================================================

def _gpu_available():
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_augment_batch(pil_images, device, noise_prob=0.3, blur_prob=0.3, augment_prob=0.5,
                      noise_intensity=0.04, blur_radius=0.9,
                      brightness_max=20.0, contrast_max=0.2, morph_prob=0.15):
    """
    Apply augmentations to a batch of PIL images using GPU tensors.
    Text rendering stays on CPU; only the pixel-level transforms run on GPU.
    Returns a list of augmented PIL images.
    """
    import torch
    import torch.nn.functional as F

    B = len(pil_images)
    if B == 0:
        return []

    # Find max dimensions for padding into a uniform batch
    max_h = max(img.size[1] for img in pil_images)
    max_w = max(img.size[0] for img in pil_images)
    orig_sizes = [(img.size[0], img.size[1]) for img in pil_images]  # (w, h)

    # Convert PIL → tensor batch [B, 1, H, W] on GPU (initialize with 255.0 WHITE so padding matches background)
    batch = torch.full((B, 1, max_h, max_w), 255.0, device=device, dtype=torch.float32)
    for i, img in enumerate(pil_images):
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        batch[i, 0, :h, :w] = torch.from_numpy(arr)

    # --- GPU Augmentations ---

    # 1. Brightness & Contrast
    if augment_prob > 0:
        mask = torch.rand(B, device=device) < augment_prob
        if mask.any():
            contrast = (1.0 - contrast_max) + (contrast_max * 2.0) * torch.rand(B, 1, 1, 1, device=device)
            brightness = -brightness_max + (brightness_max * 2.0) * torch.rand(B, 1, 1, 1, device=device)
            adjusted = (batch - 127.5) * contrast + 127.5 + brightness
            batch = torch.where(mask.view(B, 1, 1, 1), adjusted, batch)

    # 2. Gaussian noise
    if noise_prob > 0:
        mask = torch.rand(B, device=device) < noise_prob
        if mask.any():
            # Vary intensity around target
            intensity = noise_intensity * (0.5 + torch.rand(1, device=device).item())
            noise = torch.randn_like(batch) * intensity * 255
            noisy = batch + noise
            batch = torch.where(mask.view(B, 1, 1, 1), noisy, batch)

    # 3. Salt-and-pepper noise (reduced rate and 0.1% speckle so diacritics stay intact)
    if noise_prob > 0:
        mask = torch.rand(B, device=device) < (noise_prob * 0.15)
        if mask.any():
            sp = torch.rand_like(batch)
            salt = (sp > 0.999).float() * 255
            pepper = (sp < 0.001).float() * (-255)
            sp_noise = batch + salt + pepper
            batch = torch.where(mask.view(B, 1, 1, 1), sp_noise, batch)

    # 4. Background texture (paper grain)
    if augment_prob > 0:
        mask = torch.rand(B, device=device) < (augment_prob * 0.5)
        if mask.any():
            texture = torch.randn_like(batch) * 5
            batch = torch.where(mask.view(B, 1, 1, 1), batch + texture, batch)

    # 5. Gaussian blur (3x3 kernel, applied per-image via depthwise conv)
    if blur_prob > 0:
        mask = torch.rand(B, device=device) < blur_prob
        if mask.any():
            # Simple 3x3 Gaussian kernel
            kernel = torch.tensor([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=torch.float32, device=device)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, 3, 3)
            padded = F.pad(batch, (1, 1, 1, 1), mode='reflect')
            blurred = F.conv2d(padded, kernel, groups=1)
            batch = torch.where(mask.view(B, 1, 1, 1), blurred, batch)

    # 6. Erosion (simulate ink thickness / bleed)
    if augment_prob > 0 and morph_prob > 0:
        mask_erode = torch.rand(B, device=device) < (augment_prob * morph_prob)
        if mask_erode.any():
            padded = F.pad(batch, (1, 1, 1, 1), mode='reflect')
            eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1, padding=0)
            batch = torch.where(mask_erode.view(B, 1, 1, 1), eroded, batch)

    # Clamp to valid range
    batch = batch.clamp(0, 255)

    # Convert back to PIL
    batch_np = batch.cpu().numpy().astype(np.uint8)
    results = []
    for i in range(B):
        ow, oh = orig_sizes[i]
        arr = batch_np[i, 0, :oh, :ow]
        results.append(Image.fromarray(arr, mode='L'))

    return results


# ============================================================================
#  .lstmf Writer
# ============================================================================

def write_uint32(f, v):
    f.write(struct.pack('<I', v))

def write_int32(f, v):
    f.write(struct.pack('<i', v))

def write_int16(f, v):
    f.write(struct.pack('<h', v))

def write_uint8(f, v):
    f.write(struct.pack('<B', v))

def write_int8(f, v):
    f.write(struct.pack('<b', v))

def write_string(f, s):
    data = s.encode('utf-8')
    write_uint32(f, len(data))
    f.write(data)

def write_char_vector(f, data):
    write_uint32(f, len(data))
    f.write(data)


def image_to_lstmf(img, text, filename, render_text=None, page_num=0):
    """Convert a PIL image + text to .lstmf binary format.

    Args:
        img:         PIL image (rendered WITHOUT spaces for Lao).
        text:        Original GT text (WITH spaces) — used as transcription.
        render_text: Space-stripped text used for rendering (WITH spaces removed).
                     If None, falls back to text. Box count must match rendered chars.
    """
    buf = io.BytesIO()

    # Save image as PNG bytes
    png_buf = io.BytesIO()
    img.save(png_buf, format='PNG')
    png_bytes = png_buf.getvalue()

    # Use render_text (no spaces) for pixel-box alignment since the image
    # was rendered without spaces. Use original text as transcription label.
    box_source = render_text if render_text is not None else text

    # Build boxes (one box per character spanning the whole image)
    w, h = img.size
    char_w = max(1, w // max(len(box_source), 1))
    boxes = []
    box_texts = []
    for i, ch in enumerate(box_source):
        x1 = i * char_w
        x2 = min((i + 1) * char_w, w)
        boxes.append((x1, 0, x2, h))
        box_texts.append(ch)

    # Serialize as TFile/ImageData
    # num_pages = 1
    write_uint32(buf, 1)
    # non_null flag
    write_uint8(buf, 1)
    # imagefilename
    write_string(buf, filename)
    # page_number
    write_int32(buf, page_num)
    # image_data (PNG bytes as char vector)
    write_char_vector(buf, png_bytes)
    # language
    write_string(buf, '')
    # transcription
    write_string(buf, text)
    # boxes (vector of TBOX: left, bottom, right, top as int16)
    write_uint32(buf, len(boxes))
    for (x1, y1, x2, y2) in boxes:
        write_int16(buf, x1)
        write_int16(buf, y1)
        write_int16(buf, x2)
        write_int16(buf, y2)
    # box_texts
    write_uint32(buf, len(box_texts))
    for bt in box_texts:
        write_string(buf, bt)
    # vertical
    write_int8(buf, 0)

    return buf.getvalue()


# ============================================================================
#  Sample generation (shared by the CPU and GPU paths)
# ============================================================================

_CFG = {}


def _init_worker(cfg):
    """Pool initializer: share the generation config (corpus, fonts, params)."""
    global _CFG
    _CFG = cfg


def _seed_sample(idx):
    """Seed each sample from (seed, idx): reproducible, and different in every
    worker process (forked workers would otherwise repeat the same choices)."""
    s = _CFG['seed'] * 1_000_003 + idx
    random.seed(s)
    np.random.seed(s % (2 ** 32))


def _render_sample(text):
    """Render text with a random font and size.

    If a font can't render the text faithfully, other fonts are tried, then other
    corpus lines. Returns (image, text the image shows) or (None, text).
    """
    cfg = _CFG
    for attempt in range(30):
        if attempt and attempt % 10 == 0:
            text = random.choice(cfg['lines'])
        sec = random.choice(cfg['sec_font_paths']) if cfg['sec_font_paths'] else None
        img = render_line(text, random.choice(cfg['font_paths']),
                          random.choice(cfg['font_sizes']), sec_font_path=sec)
        if img is not None:
            return img, text
    return None, text


def _save_sample(img, text, name):
    cfg = _CFG
    out, ext = cfg['output_dir'], cfg['ext']
    img_path = os.path.join(out, f"{name}.{ext}")
    if ext == 'tif':
        img.save(img_path, format='TIFF', compression='tiff_lzw')
    elif ext == 'jpg':
        img.convert('RGB').save(img_path, format='JPEG', quality=95)
    else:
        img.save(img_path, format='PNG')
    with open(os.path.join(out, f"{name}.gt.txt"), 'w', encoding='utf-8') as f:
        f.write(text)
    if cfg['to_lstmf']:
        with open(os.path.join(out, f"{name}.lstmf"), 'wb') as f:
            f.write(image_to_lstmf(img, text, name))


def generate_sample(task):
    """Generate and save one sample (CPU path). Returns its name, or None if skipped."""
    idx, text = task
    _seed_sample(idx)
    img, text = _render_sample(text)
    if img is None:
        return None
    if _CFG['preset']:
        img = apply_preset_augmentation(img, _CFG['preset'])
    else:
        img = augment_image(img, **_CFG['aug'])
    name = f"{_CFG['prefix']}_{idx:07d}"
    _save_sample(img, text, name)
    return name


def generate_batch_gpu(tasks, device):
    """Render on CPU, augment pixels on GPU, then geometric and real-world ops on CPU.

    Returns the number of samples saved.
    """
    cfg = _CFG
    imgs, texts, names = [], [], []
    for idx, text in tasks:
        _seed_sample(idx)
        img, text = _render_sample(text)
        if img is not None:
            imgs.append(img)
            texts.append(text)
            names.append(f"{cfg['prefix']}_{idx:07d}")

    aug = cfg['aug']
    if aug['noise_prob'] > 0 or aug['blur_prob'] > 0 or aug['augment_prob'] > 0:
        imgs = gpu_augment_batch(imgs, device, **aug)

    p_aug = aug['augment_prob']
    for i in range(len(imgs)):
        # Presets rotate once at the end of apply_realworld_augmentations()
        if not cfg['preset'] and p_aug > 0 and random.random() < p_aug:
            imgs[i] = random_rotation(imgs[i], max_angle=1.5)
        if p_aug > 0 and random.random() < p_aug * 0.5:
            imgs[i] = random_perspective(imgs[i], distortion=0.03)
        if p_aug > 0 and random.random() < p_aug * 0.5:
            imgs[i] = random_scale(imgs[i], scale_range=(0.93, 1.07))
        if cfg['preset']:
            imgs[i] = apply_realworld_augmentations(imgs[i], cfg['preset'])

    for img, text, name in zip(imgs, texts, names):
        _save_sample(img, text, name)
    return len(imgs)


# ============================================================================
#  Conversion: existing .tif + .gt.txt → .lstmf
# ============================================================================

def convert_to_lstmf(output_dir, ext='tif'):
    """Convert all existing .{ext} + .gt.txt pairs to .lstmf files."""
    import glob
    img_files = sorted(glob.glob(os.path.join(output_dir, f'*.{ext}')))
    converted = 0
    for img_path in img_files:
        base = os.path.splitext(img_path)[0]
        gt_path = base + '.gt.txt'
        lstmf_path = base + '.lstmf'

        if not os.path.exists(gt_path):
            continue
        if os.path.exists(lstmf_path):
            continue

        try:
            img = Image.open(img_path).convert('L')
            with open(gt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                continue

            lstmf_data = image_to_lstmf(img, text, os.path.basename(base))
            with open(lstmf_path, 'wb') as f:
                f.write(lstmf_data)
            converted += 1
        except Exception as e:
            print(f"  Warning: {img_path}: {e}")

    print(f"  ✓ Converted {converted} files to .lstmf")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Synthetic OCR Training Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic generation
    python3 gen_data_from_text.py --corpus text.txt --font font.ttf --count 10000

    # With heavy augmentation
    python3 gen_data_from_text.py --corpus text.txt --font font.ttf --count 50000 \\
        --noise 0.4 --blur 0.3 --augment 0.6 --to_lstmf

    # Convert existing image+.gt.txt to .lstmf
    python3 gen_data_from_text.py --convert_only --ext png --output_dir data/my-ground-truth
""")

    parser.add_argument('--corpus', help='Text file with one line per sample')
    parser.add_argument('--output_dir', required=True, help='Output directory for generated files')
    parser.add_argument('--font', nargs='+', default=[], help='Primary Lao font file(s) (.ttf/.otf)')
    parser.add_argument('--sec_font', nargs='+', default=[],
                        help='Secondary font file(s) for numbers/symbols (.ttf/.otf). Defaults to TimesNewRoman.ttf if available.')
    parser.add_argument('--count', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--prefix', default='synth', help='Filename prefix (default: synth)')

    # Augmentation preset (Phase 2: real-world calibrated)
    parser.add_argument('--preset', default=None,
                        choices=list(_AUGMENTATION_PRESETS.keys()),
                        help=('Augmentation preset. Overrides --noise/--blur/--augment and '
                              'enables real-world augmentations (JPEG artifacts, paper texture, '
                              'uneven illumination, colored ink). '
                              'Choices: clean | scan-mild | scan-heavy | real-match. '
                              'real-match is calibrated from Phase 1 analysis of 50 real '
                              'Lao document lines (noise_σ=11.75, contrast=0.773).'))

    # Manual augmentation (used when --preset is not specified)
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise probability 0.0-1.0 (default: 0.0)')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='Blur probability 0.0-1.0 (default: 0.0)')
    parser.add_argument('--augment', type=float, default=0.0,
                        help='General augmentation probability 0.0-1.0 (rotation, scale, '
                             'perspective, erosion, contrast, texture)')
    parser.add_argument('--noise_intensity', type=float, default=0.04,
                        help='Amount/intensity of noise if applied e.g. 0.05 for 5%% (default: 0.04)')
    parser.add_argument('--blur_radius', type=float, default=0.9,
                        help='Radius/amount of blur if applied (default: 0.9)')
    parser.add_argument('--brightness_max', type=float, default=20.0,
                        help='Max random brightness shift if applied, e.g. 50 (default: 20.0)')
    parser.add_argument('--contrast_max', type=float, default=0.2,
                        help='Max random contrast shift if applied, e.g. 0.5 (default: 0.2)')
    parser.add_argument('--morph_prob', type=float, default=0.15,
                        help='Probability multiplier for morphology erosion/dilation (default: 0.15). Set to 0 to disable.')

    # Font rendering
    parser.add_argument('--font_size_min', type=int, default=28, help='Min font size (default: 28)')
    parser.add_argument('--font_size_max', type=int, default=48, help='Max font size (default: 48)')
    parser.add_argument('--max_line_chars', type=int, default=200, help='Max chars per line')

    # Output format
    parser.add_argument('--to_lstmf', action='store_true', help='Also generate .lstmf files')
    parser.add_argument('--ext', default='tif', choices=['tif', 'png', 'jpg'], help='Image output format (default: tif)')
    parser.add_argument('--convert_only', action='store_true',
                        help='Only convert existing images+.gt.txt to .lstmf (no generation)')

    # Performance
    parser.add_argument('--workers', type=int, default=4, help='Parallel CPU workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device for augmentation: cpu (multiprocessing) or cuda (GPU batched)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for GPU mode (default: 128)')

    args = parser.parse_args()

    # Convert-only mode
    if args.convert_only:
        print(f"  Converting .{args.ext}+.gt.txt → .lstmf in {args.output_dir}")
        convert_to_lstmf(args.output_dir, args.ext)
        return

    # Validate
    if not args.corpus:
        print("Error: --corpus required for generation mode")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load corpus
    lines = load_corpus(args.corpus, max_line_chars=args.max_line_chars)
    if not lines:
        print(f"Error: No valid lines in {args.corpus}")
        sys.exit(1)
    print(f"  Corpus: {len(lines)} lines from {args.corpus}")

    # Font sizes
    font_sizes = list(range(args.font_size_min, args.font_size_max + 1, 2))
    if not font_sizes:
        font_sizes = [36]

    # Font paths (validate they exist)
    font_paths = [f for f in args.font if os.path.exists(f)]
    for f in args.font:
        if not os.path.exists(f):
            print(f"  Warning: font not found: {f}")
    if not font_paths:
        # Fall back to system fonts that actually contain Lao glyphs
        import glob
        sys_fonts = sorted(glob.glob('/usr/share/fonts/**/*.ttf', recursive=True) +
                           glob.glob('/Library/Fonts/**/*.ttf', recursive=True))
        font_paths = [f for f in sys_fonts if _font_cmap(f) and font_covers(f, 'ກາ')][:3]
        if not font_paths:
            print("Error: no Lao font found. Pass one with --font (e.g. fonts/saysettha_ot.ttf).")
            sys.exit(1)
        print(f"  No --font specified, using system Lao fonts: {font_paths}")

    # If font_paths contains both Lao/primary fonts and Times/Roman/Latin-only fonts (e.g. from --font fonts/*),
    # separate them so primary font_paths only has Lao fonts and sec_font_paths gets the Latin/Roman fonts.
    if len(font_paths) > 1:
        lao_candidates = [f for f in font_paths if not any(k in os.path.basename(f).lower() for k in ['times', 'roman', 'serif', 'arial', 'liberation', 'courier'])]
        if lao_candidates and len(lao_candidates) < len(font_paths):
            if not args.sec_font:
                args.sec_font = [f for f in font_paths if f not in lao_candidates]
            font_paths = lao_candidates

    print(f"  Primary Fonts: {len(font_paths)} font(s), sizes {args.font_size_min}-{args.font_size_max}")

    # Secondary font paths for numbers/symbols (Times New Roman fallback)
    sec_font_paths = [f for f in args.sec_font if os.path.exists(f)]
    if not sec_font_paths:
        import glob
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_tnr = [
            os.path.join(script_dir, 'tesstrain_core/font/TimesNewRoman.ttf'),
            'tesstrain_core/font/TimesNewRoman.ttf',
            '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'
        ]
        sec_font_paths = [f for f in default_tnr if os.path.exists(f)]
        if not sec_font_paths:
            sec_font_paths = font_paths
    print(f"  Secondary Fonts: {len(sec_font_paths)} font(s) for numbers/symbols")

    # How each primary font will be rendered
    for fp in font_paths:
        mode = ('HarfBuzz shaping' if font_shapes_correctly(fp)
                else 'cluster fallback renderer (HarfBuzz unavailable or broken for this font)')
        print(f"    {os.path.basename(fp)}: {mode}")
    if _font_cmap(font_paths[0]) is None:
        print("  Note: install fonttools (pip install fonttools) to skip glyphs a font lacks")

    # Drop corpus lines containing characters no font can draw: their images
    # would show boxes or gaps while the label still names the character.
    all_fonts = font_paths + [f for f in sec_font_paths if f not in font_paths]
    drawable = lambda ch: ch.isspace() or any(font_covers(f, ch) for f in all_fonts)
    bad_chars = {ch for line in set(lines) for ch in line if not drawable(ch)}
    if bad_chars:
        before = len(lines)
        lines = [l for l in lines if not any(ch in bad_chars for ch in l)]
        print(f"  Dropped {before - len(lines)} corpus lines with characters no font can draw: "
              f"{' '.join(sorted(bad_chars))[:120]}")
        if not lines:
            print("Error: no renderable lines left in the corpus.")
            sys.exit(1)

    print(f"  Count: {args.count}")
    if args.preset:
        pdesc = _AUGMENTATION_PRESETS[args.preset]['_description']
        print(f"  Preset: {args.preset} — {pdesc}")
    else:
        print(f"  Noise: {args.noise:.0%} (intensity: {args.noise_intensity})  Blur: {args.blur:.0%} (radius: {args.blur_radius})")
        print(f"  Augment: {args.augment:.0%} (bright_max: {args.brightness_max}, cont_max: {args.contrast_max}, morph_prob: {args.morph_prob:.2f})")
    print(f"  Output: {args.output_dir}")
    print(f"  Format: .{args.ext} + .gt.txt" + (" + .lstmf" if args.to_lstmf else ""))

    use_gpu = (args.device == 'cuda')
    if use_gpu:
        if not _gpu_available():
            print("  ⚠ CUDA not available, falling back to CPU")
            use_gpu = False
        else:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  Device: GPU ({gpu_name})  batch_size={args.batch_size}")
    else:
        print(f"  Device: CPU  workers={args.workers}")
    print()

    # Resolve preset vs. manual augmentation params
    if args.preset:
        aug = preset_base_params(args.preset)
    else:
        aug = dict(noise_prob=args.noise, blur_prob=args.blur, augment_prob=args.augment,
                   noise_intensity=args.noise_intensity, blur_radius=args.blur_radius,
                   brightness_max=args.brightness_max, contrast_max=args.contrast_max,
                   morph_prob=args.morph_prob)

    cfg = dict(lines=lines, font_paths=font_paths, sec_font_paths=sec_font_paths,
               font_sizes=font_sizes, output_dir=args.output_dir, ext=args.ext,
               to_lstmf=args.to_lstmf, prefix=args.prefix, seed=args.seed,
               preset=args.preset, aug=aug)
    tasks = [(i, random.choice(lines)) for i in range(args.count)]

    t0 = time.time()
    done = 0

    def report(n):
        if n and n % 1000 == 0:
            sps = n / (time.time() - t0)
            print(f"  [{n:7d}/{args.count}] {sps:.0f} samples/sec  ETA: {(args.count - n) / max(sps, 1):.0f}s")

    if use_gpu:
        import torch
        device = torch.device('cuda')
        _init_worker(cfg)
        bs = args.batch_size
        for start in range(0, len(tasks), bs):
            done += generate_batch_gpu(tasks[start:start + bs], device)
            if done % 1000 < bs:
                report(done - done % 1000)
    else:
        workers = max(1, min(args.workers, args.count))
        if workers > 1:
            with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for name in pool.imap_unordered(generate_sample, tasks, chunksize=50):
                    if name:
                        done += 1
                        report(done)
        else:
            _init_worker(cfg)
            for task in tasks:
                if generate_sample(task):
                    done += 1
                    report(done)

    elapsed = time.time() - t0
    print(f"\n  ✓ Generated {done} samples in {elapsed:.1f}s ({done/max(elapsed,0.01):.0f}/sec)")
    if done < args.count:
        print(f"  ⚠ Skipped {args.count - done} samples that no font could render faithfully")
    print(f"  Output: {args.output_dir}")

    # Count files
    n_img = len([f for f in os.listdir(args.output_dir) if f.endswith(f'.{args.ext}')])
    n_gt = len([f for f in os.listdir(args.output_dir) if f.endswith('.gt.txt')])
    n_lstmf = len([f for f in os.listdir(args.output_dir) if f.endswith('.lstmf')])
    print(f"  Files: {n_img} .{args.ext}, {n_gt} .gt.txt" +
          (f", {n_lstmf} .lstmf" if args.to_lstmf else ""))


if __name__ == '__main__':
    main()
