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
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


# ============================================================================
#  Text Rendering
# ============================================================================

def load_corpus(corpus_path, max_line_chars=200):
    """Load text lines from corpus file and normalize Lao orthography."""
    # We strictly use utf-8 and utf-8-sig to ensure proper Lao UTF-8 character decoding.
    # We do NOT use utf-16 or latin-1 as fallback because they misinterpret Lao UTF-8 bytes into mojibake that fails to render.
    encodings = ['utf-8', 'utf-8-sig']
    for encoding in encodings:
        try:
            lines = []
            with open(corpus_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) <= max_line_chars:
                        line = normalize_lao_text(line)
                        lines.append(line)
            print(f"  Successfully loaded corpus {corpus_path} using {encoding} encoding ({len(lines)} lines)")
            return lines
        except (UnicodeError, LookupError):
            continue
            
    # Ultimate fallback: load as UTF-8 with replacement for any isolated malformed bytes
    print(f"  Warning: Strict UTF-8 decoding encountered malformed bytes in {corpus_path}. Loading with utf-8 (errors='replace').")
    lines = []
    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line and len(line) <= max_line_chars:
                line = normalize_lao_text(line)
                lines.append(line)
    return lines


def load_fonts(font_paths, font_size=36):
    """Load PIL fonts from paths. Falls back to default if not found."""
    fonts = []
    for fp in font_paths:
        try:
            fonts.append(ImageFont.truetype(fp, font_size))
        except Exception as e:
            print(f"  Warning: Cannot load font {fp}: {e}")
    if not fonts:
        print("  Using default font (limited character support)")
        fonts.append(ImageFont.load_default())
    return fonts


# Lao combining character codepoints that should NOT advance the cursor.
_LAO_COMBINING = frozenset([
    # Upper vowel signs (above/right)
    0x0EB1,  # LAO VOWEL SIGN MAI KAN
    0x0EB3,  # LAO VOWEL SIGN AM (Technically spaced but clusters structurally)
    0x0EB4,  # LAO VOWEL SIGN I
    0x0EB5,  # LAO VOWEL SIGN II
    0x0EB6,  # LAO VOWEL SIGN Y
    0x0EB7,  # LAO VOWEL SIGN YY
    0x0EBB,  # LAO VOWEL SIGN MAI KON
    # Lower vowel signs (below)
    0x0EB8,  # LAO VOWEL SIGN U
    0x0EB9,  # LAO VOWEL SIGN UU
    0x0EBC,  # LAO SEMIVOWEL SIGN LO
    # Tone marks (above)
    0x0EC8,  # LAO TONE MAI EK
    0x0EC9,  # LAO TONE MAI THO
    0x0ECA,  # LAO TONE MAI TI
    0x0ECB,  # LAO TONE MAI CATAWA
    0x0ECC,  # LAO CANCELLATION MARK
    0x0ECD,  # LAO NIGGAHITA
])


def _split_grapheme_clusters(text):
    """
    Split text into grapheme clusters for Lao rendering.

    A cluster is one base character followed by any number of Lao combining
    marks. Whitespace is kept as its own cluster so the caller can skip it.
    All other scripts fall through as single-codepoint clusters.
    """
    clusters = []
    current = ''
    for ch in text:
        cp = ord(ch)
        if cp in _LAO_COMBINING and current:
            # Attach to the current cluster (no cursor advance)
            current += ch
        else:
            if current:
                clusters.append(current)
            current = ch
    if current:
        clusters.append(current)
    return clusters


def normalize_lao_text(text):
    """
    Ensure correct Lao orthographic order: Base Consonant + Vowels + Tone Marks.
    Many corpora have typos where tone marks are typed before vowels (e.g. ຜູ້ instead of ຜູ້).
    This function reorders them to canonical sequence to ensure correct rendering
    and high-quality ground truth labels for the OCR model.
    """
    _LAO_VOWELS = {0x0EB1, 0x0EB3, 0x0EB4, 0x0EB5, 0x0EB6, 0x0EB7, 0x0EBB, 0x0EB8, 0x0EB9, 0x0EBC}
    _LAO_TONES = {0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB, 0x0ECC, 0x0ECD}
    
    clusters = _split_grapheme_clusters(text)
    normalized = []
    
    for cluster in clusters:
        if len(cluster) > 1:
            base = cluster[0]
            vowels = [c for c in cluster[1:] if ord(c) in _LAO_VOWELS]
            tones = [c for c in cluster[1:] if ord(c) in _LAO_TONES]
            others = [c for c in cluster[1:] if ord(c) not in _LAO_VOWELS and ord(c) not in _LAO_TONES]
            cluster = base + "".join(vowels) + "".join(tones) + "".join(others)
        normalized.append(cluster)
        
    return "".join(normalized)



def _draw_cluster_and_realign(cluster, font, canvas_h, scratch_w, padding, bg_color, fg_color):
    """Draw a grapheme cluster on a scratch canvas and horizontally realign any runaway combining marks.

    Without HarfBuzz (libraqm), PIL/FreeType on systems/fonts like phetsarath_ot.ttf pushes
    stacked combining diacritics (e.g. tone mark above vowel, as in 'ຂຶ້') to wild horizontal
    offsets (like x = +1563).
    This function detects separated horizontal ink islands and shifts them left so that
    they align perfectly above the center of the base consonant at their designed vertical levels.
    """
    scratch = Image.new('L', (scratch_w, canvas_h), bg_color)
    ImageDraw.Draw(scratch).text((0, padding), cluster, font=font, fill=fg_color)
    s_arr = np.array(scratch)

    # Find ink columns
    ink_cols = np.where(np.any(s_arr < (bg_color - 15), axis=0))[0]
    if ink_cols.size == 0:
        return scratch, s_arr

    # Segment into contiguous horizontal islands (gaps > 5 pixels)
    islands = []
    start = ink_cols[0]
    for idx in range(1, len(ink_cols)):
        if ink_cols[idx] > ink_cols[idx-1] + 5:
            islands.append((start, ink_cols[idx-1]))
            start = ink_cols[idx]
    islands.append((start, ink_cols[-1]))

    # If multiple islands exist and the others are pushed far to the right,
    # shift them horizontally to center-align above Island 0 (the base consonant)
    if len(islands) > 1:
        isl0_l, isl0_r = islands[0]
        w0 = isl0_r - isl0_l + 1
        c0 = isl0_l + w0 // 2

        # Create a recomposited canvas array
        recomp_arr = np.ones((canvas_h, scratch_w), dtype=np.uint8) * bg_color
        recomp_arr[:, isl0_l:isl0_r+1] = s_arr[:, isl0_l:isl0_r+1]

        for isl_l, isl_r in islands[1:]:
            # If the island is pushed far right (runaway diacritic)
            if isl_l - isl0_r > 15:
                w_i = isl_r - isl_l + 1
                target_l = max(0, c0 - w_i // 2)
                target_r = min(scratch_w - 1, target_l + w_i - 1)
                actual_w = target_r - target_l + 1

                slice_target = recomp_arr[:, target_l:target_r+1]
                slice_source = s_arr[:, isl_l:isl_l+actual_w]
                recomp_arr[:, target_l:target_r+1] = np.minimum(slice_target, slice_source)
            else:
                # Normal close/overlapping components, copy directly
                recomp_arr[:, isl_l:isl_r+1] = np.minimum(recomp_arr[:, isl_l:isl_r+1], s_arr[:, isl_l:isl_r+1])
        s_arr = recomp_arr
        scratch = Image.fromarray(s_arr)

    return scratch, s_arr


def render_text_line(text, font, sec_font=None, padding=10, bg_color=255, fg_color=0):
    """Render a single text line to a grayscale PIL Image.

    Uses grapheme-cluster-aware rendering to fix PIL's gap bug with Lao
    combining characters. PIL/FreeType without HarfBuzz advances the cursor
    after EVERY codepoint including combining vowels/tone marks, producing
    spurious gaps (e.g. "ລຶ    ບລ້າງ").

    Strategy:
      1. Split text into grapheme clusters (base + all combining marks).
      2. Render each cluster on a large scratch canvas at the SAME x position.
      3. Measure the actual ink width of that cluster to get the true advance.
      4. Composite onto the main canvas and advance x by the ink width.
      5. Auto-crop the final canvas to the ink region + padding.
    """
    if sec_font is None:
        try:
            tnr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tesstrain_core/font/TimesNewRoman.ttf")
            sec_font = ImageFont.truetype(tnr_path if os.path.exists(tnr_path) else "tesstrain_core/font/TimesNewRoman.ttf", getattr(font, 'size', 36))
        except Exception:
            sec_font = font

    # Estimate a safe canvas height from font metrics
    try:
        ascent, descent = font.getmetrics()
        canvas_h = max(ascent + descent + padding * 4, 60)
    except Exception:
        canvas_h = 120

    canvas_w = 6000
    canvas = Image.new('L', (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    clusters = _split_grapheme_clusters(text)

    # Estimate a reasonable word-space width from the font metrics.
    # Use ~40% of the em-square (ascent) so spaces look natural.
    try:
        space_width = max(6, int(font.getmetrics()[0] * 0.4))
    except Exception:
        space_width = 12

    x = padding * 2
    scratch_w = 4000  # wide enough for any single cluster and runaway marks

    for cluster in clusters:
        # Handle whitespace clusters as a fixed-width visual gap.
        # ASCII space, NBSP, and common Unicode spaces all advance by space_width.
        if all(ord(c) <= 0x20 or 0x2000 <= ord(c) <= 0x200F for c in cluster):
            x += space_width * len(cluster)
            continue

        # Use primary Lao font for any cluster containing Lao codepoints (\u0E80-\u0EDF),
        # otherwise use secondary font (Times New Roman) for numbers (0-9), Latin, symbols, and punctuation.
        if any(0x0E80 <= ord(c) <= 0x0EDF for c in cluster):
            c_font = font
        else:
            c_font = sec_font

        # Render the cluster on a scratch canvas to capture its pixel width.
        # This is critical for systems without libraqm where marks float right.
        scratch, s_arr = _draw_cluster_and_realign(cluster, c_font, canvas_h, scratch_w, padding, bg_color, fg_color)
        ink_cols = np.where(np.any(s_arr < (bg_color - 15), axis=0))[0]

        # If chosen c_font drew 0 ink pixels (e.g. font chosen was Latin-only or load_default()),
        # try sec_font, and if still 0 ink and cluster contains Lao, try fallback Lao font.
        if ink_cols.size == 0:
            fallback_lao = None
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                for p_lao in [
                    os.path.join(script_dir, "tesstrain_core/font/Phetsarath-Regular.ttf"),
                    os.path.join(script_dir, "fonts/Phetsarath-Regular.ttf"),
                    "tesstrain_core/font/Phetsarath-Regular.ttf",
                    "/usr/share/fonts/truetype/lao/Phetsarath_OT.ttf"
                ]:
                    if os.path.exists(p_lao):
                        fallback_lao = ImageFont.truetype(p_lao, getattr(font, 'size', 36))
                        break
            except Exception:
                pass

            for alt_font in [sec_font, fallback_lao]:
                if alt_font is not None and alt_font is not c_font:
                    scratch, s_arr = _draw_cluster_and_realign(cluster, alt_font, canvas_h, scratch_w, padding, bg_color, fg_color)
                    ink_cols = np.where(np.any(s_arr < (bg_color - 15), axis=0))[0]
                    if ink_cols.size > 0:
                        c_font = alt_font
                        break

        if ink_cols.size == 0:
            continue

        cluster_ink_w = int(ink_cols[-1]) + 1
        cluster_ink_l = int(ink_cols[0])

        box_left = int(x - cluster_ink_l)
        box_right = min(canvas_w, box_left + cluster_ink_w)
        if box_left < 0 or box_left >= canvas_w or box_right <= box_left:
            continue
            
        crop = scratch.crop((0, 0, box_right - box_left, canvas_h))
        # Composite using minimum so white background in the bounding box crop never clips/overwrites neighboring dark ink
        canvas_slice = np.array(canvas.crop((box_left, 0, box_right, canvas_h)))
        blended = np.minimum(canvas_slice, np.array(crop))
        canvas.paste(Image.fromarray(blended), (box_left, 0))

        # Advance x using chosen font's designed advance width plus proportional letter spacing (tracking)
        # to ensure that adjacent characters (such as ຽ and ງ) do not collide or merge when thickened/blurred.
        try:
            letter_spacing = max(3, int(round(getattr(c_font, 'size', 36) * 0.10)))
            ink_w = (cluster_ink_w - cluster_ink_l)
            raw_len = int(round(c_font.getlength(cluster)))
            # Protect against PIL/FreeType unshaped OpenType mark table overflow (e.g. returning 1565px for stacked diacritics in phetsarath_ot.ttf)
            max_sane_adv = max(ink_w * 2 + 15, int(round(getattr(c_font, 'size', 36) * 1.5)))
            if raw_len > max_sane_adv or raw_len < 0:
                raw_len = ink_w + 4
            adv = max(raw_len, ink_w + 2) + letter_spacing
        except Exception:
            adv = (cluster_ink_w - cluster_ink_l) + 5
        x += adv

    # Find actual ink bounding box
    arr = np.array(canvas)
    threshold = bg_color - 15
    ink_mask = arr < threshold

    rows_with_ink = np.any(ink_mask, axis=1)
    cols_with_ink = np.any(ink_mask, axis=0)

    if rows_with_ink.any() and cols_with_ink.any():
        r_min = int(np.where(rows_with_ink)[0][0])
        r_max = int(np.where(rows_with_ink)[0][-1])
        c_min = int(np.where(cols_with_ink)[0][0])
        c_max = int(np.where(cols_with_ink)[0][-1])

        left   = max(0, c_min - padding)
        top    = max(0, r_min - padding)
        right  = min(canvas_w, c_max + padding + 1)
        bottom = min(canvas_h, r_max + padding + 1)
        img = canvas.crop((left, top, right, bottom))
    else:
        img = Image.new('L', (max(len(text) * 10, 10), canvas_h), bg_color)

    return img



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


def random_rotation(img, max_angle=2.0):
    """Slight random rotation."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


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
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=255)


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
                  brightness_max=20.0, contrast_max=0.2, morph_prob=0.3):
    """Apply random augmentations to an image."""
    # Always-optional augmentations
    if random.random() < augment_prob:
        img = random_rotation(img, max_angle=1.5)
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


def apply_preset_augmentation(img, preset_name):
    """
    Apply a named augmentation preset to a PIL image.
    Runs base augment_image() first, then applies real-world augmentations
    (JPEG, illumination, ink color, paper texture, scan noise).
    """
    if preset_name not in _AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. "
                         f"Choose from: {list(_AUGMENTATION_PRESETS.keys())}")

    p = _AUGMENTATION_PRESETS[preset_name]

    # 1. Base augmentations
    img = augment_image(
        img,
        noise_prob=p['noise_prob'],
        blur_prob=p['blur_prob'],
        augment_prob=p['augment_prob'],
        noise_intensity=p['noise_intensity'],
        blur_radius=p['blur_radius'],
        brightness_max=p['brightness_max'],
        contrast_max=p['contrast_max'],
        morph_prob=p['morph_prob'],
    )

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


def generate_batch_gpu(batch_items, device, noise_prob, blur_prob, augment_prob,
                       noise_intensity, blur_radius, brightness_max, contrast_max, 
                       morph_prob, to_lstmf, ext, output_dir):
    """
    Generate a batch of samples using GPU-accelerated augmentations.
    1. Render all text lines on CPU (PIL)
    2. Augment the batch on GPU (PyTorch)
    3. Save results to disk
    Returns number of samples generated.
    """
    pil_images = []
    texts = []
    names = []

    for item in batch_items:
        if len(item) == 6:
            idx, text, font_paths, sec_font_paths, font_sizes, prefix = item
        else:
            idx, text, font_paths, font_sizes, prefix = item
            sec_font_paths = []

        for attempt in range(max(len(font_paths) + 3, 10)):
            font_size = random.choice(font_sizes)
            font_path = random.choice(font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()

            sec_font = None
            if sec_font_paths:
                sec_path = random.choice(sec_font_paths)
                try:
                    sec_font = ImageFont.truetype(sec_path, font_size)
                except Exception:
                    sec_font = font

            img = render_text_line(text, font, sec_font=sec_font)
            arr_test = np.array(img)
            if np.any(arr_test < 240):
                break
        pil_images.append(img)
        texts.append(text)
        names.append(f"{prefix}_{idx:07d}")

    # GPU augmentation
    if noise_prob > 0 or blur_prob > 0 or augment_prob > 0:
        pil_images = gpu_augment_batch(pil_images, device, noise_prob, blur_prob, augment_prob,
                                       noise_intensity, blur_radius, brightness_max, contrast_max, 
                                       morph_prob)

    # CPU rotation & perspective (these change dimensions, harder to batch)
    for i in range(len(pil_images)):
        if augment_prob > 0 and random.random() < augment_prob:
            pil_images[i] = random_rotation(pil_images[i], max_angle=1.5)
        if augment_prob > 0 and random.random() < augment_prob * 0.5:
            pil_images[i] = random_perspective(pil_images[i], distortion=0.03)
        if augment_prob > 0 and random.random() < augment_prob * 0.5:
            pil_images[i] = random_scale(pil_images[i], scale_range=(0.93, 1.07))

    # Save
    for i, (img, text, name) in enumerate(zip(pil_images, texts, names)):
        img_path = os.path.join(output_dir, f"{name}.{ext}")
        gt_path = os.path.join(output_dir, f"{name}.gt.txt")

        if ext == 'tif':
            img.save(img_path, format='TIFF', compression='tiff_lzw')
        elif ext == 'png':
            img.save(img_path, format='PNG')
        elif ext == 'jpg' or ext == 'jpeg':
            img.convert('RGB').save(img_path, format='JPEG', quality=95)
        else:
            img.save(img_path)

        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if to_lstmf:
            lstmf_path = os.path.join(output_dir, f"{name}.lstmf")
            lstmf_data = image_to_lstmf(img, text, name)
            with open(lstmf_path, 'wb') as f:
                f.write(lstmf_data)

    return len(pil_images)


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
#  Worker for parallel generation
# ============================================================================

def generate_sample(args_tuple):
    """Generate a single sample (called by multiprocessing Pool)."""
    if len(args_tuple) == 17:
        (idx, text, font_paths, sec_font_paths, font_sizes, output_dir, noise_prob,
         blur_prob, augment_prob, noise_intensity, blur_radius, 
         brightness_max, contrast_max, morph_prob, to_lstmf, prefix, ext) = args_tuple
    else:
        (idx, text, font_paths, font_sizes, output_dir, noise_prob,
         blur_prob, augment_prob, noise_intensity, blur_radius, 
         brightness_max, contrast_max, morph_prob, to_lstmf, prefix, ext) = args_tuple
        sec_font_paths = []

    # Random font and size with retry if rendering produces blank image
    for attempt in range(max(len(font_paths) + 3, 10)):
        font_size = random.choice(font_sizes)
        font_path = random.choice(font_paths)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        sec_font = None
        if sec_font_paths:
            sec_path = random.choice(sec_font_paths)
            try:
                sec_font = ImageFont.truetype(sec_path, font_size)
            except Exception:
                sec_font = font

        img = render_text_line(text, font, sec_font=sec_font)
        arr_test = np.array(img)
        if np.any(arr_test < 240):
            break

    # Augment
    img = augment_image(img, noise_prob=noise_prob, blur_prob=blur_prob,
                        augment_prob=augment_prob, noise_intensity=noise_intensity,
                        blur_radius=blur_radius, brightness_max=brightness_max,
                        contrast_max=contrast_max, morph_prob=morph_prob)

    # Filenames
    name = f"{prefix}_{idx:07d}"
    img_path = os.path.join(output_dir, f"{name}.{ext}")
    gt_path = os.path.join(output_dir, f"{name}.gt.txt")

    # Save Image
    if ext == 'tif':
        img.save(img_path, format='TIFF', compression='tiff_lzw')
    elif ext == 'png':
        img.save(img_path, format='PNG')
    elif ext == 'jpg' or ext == 'jpeg':
        img.convert('RGB').save(img_path, format='JPEG', quality=95)
    else:
        img.save(img_path)

    # Save .gt.txt
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Optionally save .lstmf
    if to_lstmf:
        lstmf_path = os.path.join(output_dir, f"{name}.lstmf")
        lstmf_data = image_to_lstmf(img, text, name)
        with open(lstmf_path, 'wb') as f:
            f.write(lstmf_data)

    return name


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
    print(f"  Corpus: {len(lines)} unique lines from {args.corpus}")

    # Font sizes
    font_sizes = list(range(args.font_size_min, args.font_size_max + 1, 2))
    if not font_sizes:
        font_sizes = [36]

    # Font paths (validate they exist)
    font_paths = [f for f in args.font if os.path.exists(f)]
    if not font_paths:
        # Try to find system fonts
        import glob
        sys_fonts = glob.glob('/usr/share/fonts/**/*.ttf', recursive=True)
        if sys_fonts:
            font_paths = sys_fonts[:3]
            print(f"  No --font specified, using system fonts: {font_paths}")
        else:
            print("  Warning: No fonts found, using default (limited charset)")
            font_paths = ['']

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
        # When using a preset in GPU mode, GPU handles base noise/blur/augment;
        # the extra real-world ops (JPEG, paper texture, etc.) run on CPU per-image.
        _p = _AUGMENTATION_PRESETS[args.preset]
        _noise_prob      = _p['noise_prob']
        _blur_prob       = _p['blur_prob']
        _augment_prob    = _p['augment_prob']
        _noise_intensity = _p['noise_intensity']
        _blur_radius     = _p['blur_radius']
        _brightness_max  = _p['brightness_max']
        _contrast_max    = _p['contrast_max']
        _morph_prob      = _p['morph_prob']
    else:
        _noise_prob      = args.noise
        _blur_prob       = args.blur
        _augment_prob    = args.augment
        _noise_intensity = args.noise_intensity
        _blur_radius     = args.blur_radius
        _brightness_max  = args.brightness_max
        _contrast_max    = args.contrast_max
        _morph_prob      = args.morph_prob

    # Generate
    t0 = time.time()
    done = 0

    if use_gpu:
        # ----- GPU batched mode -----
        import torch
        device = torch.device('cuda')

        # Build lightweight task list (no output_dir/probs, those are passed separately)
        tasks = []
        for i in range(args.count):
            text = random.choice(lines)
            tasks.append((i, text, font_paths, sec_font_paths, font_sizes, args.prefix))

        bs = args.batch_size
        for start in range(0, len(tasks), bs):
            batch_items = tasks[start:start + bs]
            n = generate_batch_gpu(
                batch_items, device,
                noise_prob=_noise_prob, blur_prob=_blur_prob, augment_prob=_augment_prob,
                noise_intensity=_noise_intensity, blur_radius=_blur_radius,
                brightness_max=_brightness_max, contrast_max=_contrast_max,
                morph_prob=_morph_prob,
                to_lstmf=args.to_lstmf, ext=args.ext, output_dir=args.output_dir
            )
            done += n
            if done % 1000 < bs:
                elapsed = time.time() - t0
                sps = done / elapsed
                eta = (args.count - done) / max(sps, 1)
                print(f"  [{done:7d}/{args.count}] {sps:.0f} samples/sec  ETA: {eta:.0f}s")

    else:
        # ----- CPU multiprocessing mode -----
        tasks = []
        for i in range(args.count):
            text = random.choice(lines)
            tasks.append((i, text, font_paths, sec_font_paths, font_sizes, args.output_dir,
                           _noise_prob, _blur_prob, _augment_prob, _noise_intensity,
                           _blur_radius, _brightness_max, _contrast_max,
                           _morph_prob, args.to_lstmf, args.prefix, args.ext))

        workers = min(args.workers, args.count)
        if workers > 1:
            with multiprocessing.Pool(workers) as pool:
                for name in pool.imap_unordered(generate_sample, tasks, chunksize=100):
                    done += 1
                    if done % 1000 == 0:
                        elapsed = time.time() - t0
                        sps = done / elapsed
                        eta = (args.count - done) / sps
                        print(f"  [{done:7d}/{args.count}] {sps:.0f} samples/sec  ETA: {eta:.0f}s")
        else:
            for task in tasks:
                generate_sample(task)
                done += 1
                if done % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{done:7d}/{args.count}] {done/elapsed:.0f} samples/sec")

    elapsed = time.time() - t0
    print(f"\n  ✓ Generated {args.count} samples in {elapsed:.1f}s ({args.count/max(elapsed,0.01):.0f}/sec)")
    print(f"  Output: {args.output_dir}")

    # Count files
    n_img = len([f for f in os.listdir(args.output_dir) if f.endswith(f'.{args.ext}')])
    n_gt = len([f for f in os.listdir(args.output_dir) if f.endswith('.gt.txt')])
    n_lstmf = len([f for f in os.listdir(args.output_dir) if f.endswith('.lstmf')])
    print(f"  Files: {n_img} .{args.ext}, {n_gt} .gt.txt" +
          (f", {n_lstmf} .lstmf" if args.to_lstmf else ""))


if __name__ == '__main__':
    main()
