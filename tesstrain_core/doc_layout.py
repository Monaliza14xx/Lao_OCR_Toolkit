"""Document layout analysis and text-line segmentation for scanned pages.

The same pipeline is used for real scans and for simulated pages, so synthetic
line crops get the same cropping behaviour (margins, skew residue, stamp overlap)
as the real lines a deployed OCR model will see.

Pipeline (segment_page):
  1. Ink mask: dark pixels plus strongly coloured ones (red stamps, blue pen).
  2. Deskew (optional, done by the caller): projection-profile search.
  3. Connected components, classified by size against the page's typical
     letter-body height ("core"):
       body    letters and digits             0.55–3 × core tall
       mark    vowel/tone marks, dots, commas smaller
       graphic emblem, stamp, signature       taller than 3 × core
       rule    ruled lines / long underlines   very wide and thin
  4. Lines: body components are chained when they overlap vertically and sit
     less than `gap_factor` × core apart horizontally (a wider gap starts a new
     line, e.g. a left-aligned name and a right-aligned number on one row).
  5. Marks join the nearest line above or below them. Lao marks above a line
     reach further (stacked vowel + tone) than marks below one, so distances
     are weighted accordingly. Marks are attached whole, never cut.
  6. Crop each line's box with a margin that stops at neighbouring lines.

Tesseract is optional and only used for draft text of the cropped lines.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

PDF_EXTS = ('.pdf',)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')


# ----------------------------------------------------------------------------
#  Loading
# ----------------------------------------------------------------------------

def load_pages(path, dpi=300):
    """Return a list of RGB page images from a PDF or an image file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXTS:
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise SystemExit("Reading PDFs needs pypdfium2: pip install pypdfium2") from e
        pdf = pdfium.PdfDocument(path)
        return [page.render(scale=dpi / 72).to_pil().convert('RGB') for page in pdf]
    if ext in IMAGE_EXTS:
        img = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(img.convert('RGB'))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        return frames
    raise ValueError(f"Unsupported file type: {path}")


# ----------------------------------------------------------------------------
#  Ink mask
# ----------------------------------------------------------------------------

def otsu_threshold(gray):
    """Otsu threshold for a uint8 grayscale array."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    w0 = cum / total
    w1 = 1.0 - w0
    m0 = np.divide(cum_mean, cum, out=np.zeros(256), where=cum > 0)
    m1 = np.divide(cum_mean[-1] - cum_mean, total - cum, out=np.zeros(256), where=(total - cum) > 0)
    return int(np.argmax(w0 * w1 * (m0 - m1) ** 2))


def ink_mask(img):
    """Boolean ink mask of a page (dark or strongly coloured pixels)."""
    rgb = np.asarray(img.convert('RGB')).astype(np.int16)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    thr = min(otsu_threshold(gray), 200)
    sat = rgb.max(axis=2) - rgb.min(axis=2)          # red stamps / blue pen
    return (gray < thr) | (sat > 80)


# ----------------------------------------------------------------------------
#  Deskew
# ----------------------------------------------------------------------------

def estimate_skew(img, max_angle=4.0):
    """Page skew in degrees (PIL convention: rotate(+angle) is counter-clockwise).

    Projection-profile method on a downscaled ink mask: the rotation that makes
    text rows sharpest (highest variance of row sums) is the deskew angle.
    """
    small = Image.fromarray((ink_mask(img) * 255).astype(np.uint8))
    scale = 800 / max(small.size)
    small = small.resize((max(1, int(small.width * scale)), max(1, int(small.height * scale))))

    def sharpness(a):
        rows = np.asarray(small.rotate(a, resample=Image.BILINEAR)).sum(axis=1).astype(np.float64)
        return float(np.var(rows))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.25)
    best = max(coarse, key=sharpness)
    fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05)
    return float(max(fine, key=sharpness))


def deskew(img, max_angle=4.0, min_angle=0.1):
    """Rotate the page upright. Returns (image, angle_applied)."""
    angle = estimate_skew(img, max_angle)
    if abs(angle) < min_angle:
        return img, 0.0
    fill = tuple(int(v) for v in np.median(np.asarray(img.convert('RGB')).reshape(-1, 3), axis=0))
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=fill), angle


def rotate_points(points, angle, size):
    """Where points land after PIL's img.rotate(angle) (expand=False) on an image of `size`."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    a = np.radians(angle)
    pts = np.asarray(points, dtype=np.float64)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([cx + dx * np.cos(a) + dy * np.sin(a),
                     cy - dx * np.sin(a) + dy * np.cos(a)], axis=1)


def rotate_box(box, angle, size):
    """Axis-aligned bounding box of `box` after PIL's img.rotate(angle)."""
    l, t, r, b = box
    p = rotate_points([(l, t), (r, t), (r, b), (l, b)], angle, size)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


# ----------------------------------------------------------------------------
#  Connected components
# ----------------------------------------------------------------------------

def components(mask, min_area=6):
    """Connected ink components as an (n, 5) int array: left, top, right, bottom, area."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros((0, 5), dtype=np.int64)
    objs = ndimage.find_objects(lab)
    area = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    comps = np.array([(s[1].start, s[0].start, s[1].stop, s[0].stop, a)
                      for s, a in zip(objs, area)], dtype=np.int64)
    return comps[comps[:, 4] >= min_area]


def estimate_core_height(comps):
    """Typical letter-body height: median height of the taller half of components."""
    if len(comps) == 0:
        return 30.0
    h = comps[:, 3] - comps[:, 1]
    return float(np.median(h[h >= np.median(h)]))


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# ----------------------------------------------------------------------------
#  Line segmentation
# ----------------------------------------------------------------------------

def colour_graphic_regions(img, core, min_rel_h=4.0, max_aspect=4.0):
    """Boxes of stamps / signatures: clusters of red or blue ink that are tall
    (≥ min_rel_h × core) and not line-shaped. Red text lines stay text."""
    from scipy import ndimage
    rgb = np.asarray(img.convert('RGB')).astype(np.int16)
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > 55   # blur/JPEG wash thin stamp strokes out
    step = 4
    small = coloured[::step, ::step]
    small = ndimage.binary_dilation(small, iterations=max(1, int(0.6 * core / step)))
    lab, n = ndimage.label(small)
    boxes = []
    for s in ndimage.find_objects(lab):
        t, b, l, r = s[0].start * step, s[0].stop * step, s[1].start * step, s[1].stop * step
        h, w = b - t, r - l
        if h >= min_rel_h * core and w / max(h, 1) < max_aspect:
            boxes.append((l, t, r, b))
    return boxes


def segment_lines(img, gap_factor=2.2, min_overlap=0.5, mark_reach_up=1.3, mark_reach_down=0.6):
    """Find text lines on a (deskewed) page.

    Returns (lines, info):
      lines: dicts sorted in reading order with
             'box' (l, t, r, b) of all ink in the line,
             'core' (top, bottom) of its letter bodies,
             'limits' (top, bottom) a crop may use without reaching a neighbour line.
      info:  {'core_height', 'graphics': [boxes], 'rules': [boxes]}
    """
    mask = ink_mask(img)
    comps = components(mask)
    core = estimate_core_height(comps)
    l, t, r, b, comp_area = comps.T if len(comps) else (np.array([]),) * 5
    h, w = b - t, r - l

    is_rule = (w > 12 * core) & (h < 0.5 * core)
    is_graphic = ~is_rule & ((h > 3.0 * core) | (w > 25 * core))
    is_body = ~is_rule & ~is_graphic & (h >= 0.55 * core)
    is_mark = ~is_rule & ~is_graphic & ~is_body

    # 1. Chain body components into lines
    body = np.where(is_body)[0]
    body = body[np.argsort(l[body])]
    uf = _UnionFind(len(body))
    max_gap = gap_factor * core
    bl, bt, br, bb, bh = l[body], t[body], r[body], b[body], h[body]
    for a_i in range(len(body)):
        # candidates to the right whose left edge is within reach (bl is sorted)
        end = np.searchsorted(bl, br[a_i] + max_gap, side='right')
        js = np.arange(a_i + 1, end)
        if js.size == 0:
            continue
        ov = np.minimum(bb[a_i], bb[js]) - np.maximum(bt[a_i], bt[js])
        near = js[(ov >= min_overlap * np.minimum(bh[a_i], bh[js])) & (bl[js] - br[a_i] <= max_gap)]
        for b_i in near:
            uf.union(a_i, int(b_i))
    groups = {}
    for a_i, i in enumerate(body):
        groups.setdefault(uf.find(a_i), []).append(i)

    lines = []
    for members in groups.values():
        m = np.array(members)
        lines.append({
            'members': list(m),
            'box': [int(l[m].min()), int(t[m].min()), int(r[m].max()), int(b[m].max())],
            # robust letter-body band: ignore marks merged into a few bodies
            'core': (float(np.percentile(t[m], 25)), float(np.percentile(b[m], 75))),
        })

    def overlap_area(a, g):
        return max(0, min(a[2], g[2]) - max(a[0], g[0])) * max(0, min(a[3], g[3]) - max(a[1], g[1]))

    def area(a):
        return max(1, (a[2] - a[0]) * (a[3] - a[1]))

    # 2. Graphics: big components (padded a little, to catch loose emblem parts)
    #    and stamp/signature regions of coloured ink. Drop "lines" lying mostly
    #    inside one: lettering in a stamp ring, emblem parts, signature strokes.
    pad = int(0.5 * core)
    # Sparse big components are frames (table grids, boxes around text): the
    # text inside them is still text, so they don't swallow lines.
    density = comp_area / np.maximum(1, w * h)
    big_boxes = [(int(l[i]) - pad, int(t[i]) - pad, int(r[i]) + pad, int(b[i]) + pad)
                 for i in np.where(is_graphic & (density >= 0.1))[0]]
    colour_boxes = colour_graphic_regions(img, core)
    graphic_boxes = big_boxes + colour_boxes
    rgb = np.asarray(img.convert('RGB')).astype(np.int16)
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > 55

    def mostly_coloured(box):
        L, T, R, B = box
        ink = mask[T:B, L:R].sum()
        return ink > 0 and coloured[T:B, L:R].sum() > 0.5 * ink

    def in_graphic(box, share=0.5):
        # A black text line next to a red stamp is still text: coloured regions
        # only swallow lines whose own ink is mostly coloured.
        if any(overlap_area(box, g) >= share * area(box) for g in big_boxes):
            return True
        return (any(overlap_area(box, g) >= share * area(box) for g in colour_boxes)
                and mostly_coloured(box))

    lines = [ln for ln in lines if not in_graphic(ln['box'])]

    # 3. Attach marks to the nearest line (marks reach further above a line than
    #    below). Inside a graphic region only marks right next to a line's letters
    #    join it, so stamp lettering doesn't stretch a nearby line.
    for i in np.where(is_mark)[0]:
        mbox = (int(l[i]), int(t[i]), int(r[i]), int(b[i]))
        limit = 0.35 * core if in_graphic(mbox, share=0.99) else core
        best, best_d = None, None
        for ln in lines:
            L, _, R, _ = ln['box']
            if r[i] < L - 0.5 * core or l[i] > R + 0.5 * core:
                continue
            ct, cb = ln['core']
            if b[i] <= ct:                       # above the line's bodies
                d = (ct - b[i]) / mark_reach_up
            elif t[i] >= cb:                     # below
                d = (t[i] - cb) / mark_reach_down
            else:
                d = 0.0
            if d <= limit and (best_d is None or d < best_d):
                best, best_d = ln, d
        if best is not None:
            bx = best['box']
            best['box'] = [min(bx[0], mbox[0]), min(bx[1], mbox[1]),
                           max(bx[2], mbox[2]), max(bx[3], mbox[3])]

    # 4. Fold fragments lying inside another line's box into that line
    #    (e.g. the tail of a 'g' or ';' that didn't touch its letters)
    lines.sort(key=lambda x: area(x['box']), reverse=True)
    kept = []
    for ln in lines:
        host = next((k for k in kept if overlap_area(ln['box'], k['box']) >= 0.5 * area(ln['box'])), None)
        if host:
            hb, bx = host['box'], ln['box']
            host['box'] = [min(hb[0], bx[0]), min(hb[1], bx[1]), max(hb[2], bx[2]), max(hb[3], bx[3])]
        else:
            kept.append(ln)
    lines = kept

    # 5. Drop specks: a lone body-sized blob is not a line
    lines = [ln for ln in lines if ln['box'][2] - ln['box'][0] >= 0.8 * core]

    # 6. Crop limits: don't extend into a horizontally overlapping neighbour
    #    (but never cut into the line's own ink)
    H = mask.shape[0]
    for ln in lines:
        L, T, R, B = ln['box']
        nb = [o['box'] for o in lines if o is not ln and min(R, o['box'][2]) > max(L, o['box'][0])]
        above = [o[3] for o in nb if (o[1] + o[3]) / 2 < (T + B) / 2]
        below = [o[1] for o in nb if (o[1] + o[3]) / 2 > (T + B) / 2]
        ln['limits'] = (min(T, max(above)) if above else 0, max(B, min(below)) if below else H)

    # 7. Reading order: rows of vertically overlapping lines, top→bottom, then left→right
    lines.sort(key=lambda ln: ln['core'][0])
    rows = []
    for ln in lines:
        if rows and ln['core'][0] < rows[-1][-1]['core'][1]:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    ordered = [ln for row in rows for ln in sorted(row, key=lambda x: x['box'][0])]
    for ln in ordered:
        ln['box'] = tuple(int(v) for v in ln['box'])
        ln['limits'] = tuple(int(v) for v in ln['limits'])
        del ln['members']

    info = {'core_height': core,
            'graphics': [tuple(int(v) for v in g) for g in graphic_boxes],
            'rules': [tuple(int(v) for v in comps[i, :4]) for i in np.where(is_rule)[0]]}
    return ordered, info


def crop_line(img, box, margin, limits=None):
    """Crop a line with a margin; vertically the margin stops at `limits` (if given)
    so the crop never reaches into the neighbouring line."""
    l, t, r, b = box
    W, H = img.size
    top, bottom = (0, H) if limits is None else limits
    return img.crop((max(0, l - margin), max(0, top, t - margin),
                     min(W, r + margin), min(H, bottom, b + margin)))


def draw_layout(img, lines, info, scale=0.4):
    """Page preview: text lines (green, numbered), graphics (red), rules (blue)."""
    from PIL import ImageDraw
    preview = img.convert('RGB').copy()
    d = ImageDraw.Draw(preview)
    for box in info['graphics']:
        d.rectangle(box, outline=(220, 0, 0), width=4)
    for box in info['rules']:
        d.rectangle(box, outline=(0, 0, 220), width=4)
    for i, ln in enumerate(lines, start=1):
        d.rectangle(ln['box'], outline=(0, 170, 0), width=4)
        d.text((max(0, ln['box'][0] - 70), ln['box'][1]), str(i), fill=(0, 120, 0))
    return preview.resize((int(preview.width * scale), int(preview.height * scale)))


# ----------------------------------------------------------------------------
#  Optional draft text (Tesseract)
# ----------------------------------------------------------------------------

def tesseract_available():
    return shutil.which('tesseract') is not None


def draft_text(line_img, lang='lao'):
    """Tesseract's guess for one cropped line (psm 7). Needs manual correction."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'line.png')
        line_img.save(src)
        res = subprocess.run(['tesseract', src, 'stdout', '-l', lang, '--psm', '7'],
                             capture_output=True, text=True)
    return res.stdout.strip()
