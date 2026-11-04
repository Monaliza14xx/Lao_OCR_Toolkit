"""Synthetic Lao document pages with word-level ground truth.

Builds A4 pages laid out like Lao official letters (emblem, national motto,
ministry block, document number and date, title, "ເຖິງ:" / "ເລື່ອງ:" blocks,
"- ອີງຕາມ" references, justified body paragraphs mixing Lao with Latin digits,
phone numbers, e-mails and URLs, optional table, signature block with a red
stamp and a pen signature, ruled footer), then simulates scanning.

Every drawn word is recorded with its box, so lines found later by the same
segmentation used for real scans can be labelled exactly.

Defaults are calibrated on a real 300 dpi scan of a Lao ministry letter:
body ≈ 12.4 pt, line pitch ≈ 1.35 × font size, margins ≈ 3.2 cm left /
1.6 cm right, page skew ≈ 1°, gray ink (median level ≈ 53), clean white paper,
JPEG compression, red stamp ≈ 7 % and blue pen ≈ 2.5 % of the ink.
"""

import io
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from gen_data_from_text import (_HAS_RAQM, _load_font, _script_runs, _split_grapheme_clusters,
                                font_covers, font_shapes_correctly, normalize_lao_text)

# ----------------------------------------------------------------------------
#  Fixed phrases (public, generic wording of Lao official letters)
# ----------------------------------------------------------------------------

MOTTO = ['ສາທາລະນະລັດ ປະຊາທິປະໄຕ ປະຊາຊົນລາວ',
         'ສັນຕິພາບ ເອກະລາດ ປະຊາທິປະໄຕ ເອກະພາບ ວັດທະນະຖາວອນ']
MINISTRIES = ['ກະຊວງສຶກສາທິການ ແລະ ກິລາ', 'ກະຊວງສາທາລະນະສຸກ', 'ກະຊວງການເງິນ',
              'ກະຊວງກະສິກໍາ ແລະ ປ່າໄມ້', 'ກະຊວງການຕ່າງປະເທດ', 'ກະຊວງພາຍໃນ',
              'ກະຊວງແຜນການ ແລະ ການລົງທຶນ', 'ກະຊວງໂຍທາທິການ ແລະ ຂົນສົ່ງ',
              'ກະຊວງຊັບພະຍາກອນທໍາມະຊາດ ແລະ ສິ່ງແວດລ້ອມ', 'ກະຊວງອຸດສາຫະກໍາ ແລະ ການຄ້າ',
              'ກະຊວງເຕັກໂນໂລຊີ ແລະ ການສື່ສານ', 'ກະຊວງແຮງງານ ແລະ ສະຫວັດດີການສັງຄົມ']
DEPARTMENTS = ['ຫ້ອງການ', 'ກົມແຜນການ ແລະ ການຮ່ວມມື', 'ກົມຈັດຕັ້ງ ແລະ ພະນັກງານ', 'ກົມກວດກາ',
               'ກົມຮ່ວມມືກັບຕ່າງປະເທດ', 'ກົມການເງິນ', 'ສູນຂໍ້ມູນຂ່າວສານ', 'ກົມນິຕິກໍາ']
TITLES = ['ແຈ້ງການ', 'ແຈ້ງເຊີນ', 'ຂໍ້ຕົກລົງ', 'ໜັງສືສະເໜີ', 'ບົດລາຍງານ', 'ຄໍາສັ່ງ',
          'ໃບຢັ້ງຢືນ', 'ບົດບັນທຶກ', 'ໜັງສືມອບສິດ', 'ແຜນການ']
PLACES = ['ນະຄອນຫຼວງວຽງຈັນ', 'ແຂວງຫຼວງພະບາງ', 'ແຂວງຈໍາປາສັກ', 'ແຂວງສະຫວັນນະເຂດ',
          'ແຂວງຄໍາມ່ວນ', 'ແຂວງອຸດົມໄຊ', 'ແຂວງຊຽງຂວາງ', 'ແຂວງບໍ່ແກ້ວ', 'ແຂວງວຽງຈັນ']
MONTHS = ['ມັງກອນ', 'ກຸມພາ', 'ມີນາ', 'ເມສາ', 'ພຶດສະພາ', 'ມິຖຸນາ',
          'ກໍລະກົດ', 'ສິງຫາ', 'ກັນຍາ', 'ຕຸລາ', 'ພະຈິກ', 'ທັນວາ']
SIGNERS = ['ຫົວໜ້າຫ້ອງການ', 'ລັດຖະມົນຕີ', 'ຮອງລັດຖະມົນຕີ', 'ຫົວໜ້າກົມ', 'ຜູ້ອໍານວຍການ',
           'ຮອງຫົວໜ້າກົມ', 'ເຈົ້າແຂວງ', 'ຫົວໜ້າພະແນກ']
CLOSINGS = ['ດັ່ງນັ້ນ, ຈຶ່ງແຈ້ງມາຍັງທ່ານເພື່ອຊາບ ແລະ ປະຕິບັດຕາມ.',
            'ດັ່ງນັ້ນ, ຈຶ່ງແຈ້ງມາຍັງທ່ານເພື່ອຊາບ ແລະ ເຂົ້າຮ່ວມຕາມແຈ້ງການດ້ວຍ.',
            'ຈຶ່ງສະເໜີມາຍັງທ່ານເພື່ອພິຈາລະນາ.',
            'ຂໍສະແດງຄວາມນັບຖືຢ່າງສູງ.']
GREETING = ['ທ່ານ', 'ທ່ານ ຫົວໜ້າ', 'ທ່ານ ຜູ້ອໍານວຍການ', 'ທ່ານ ເຈົ້າແຂວງ']
LAO_DIGITS = str.maketrans('0123456789', '໐໑໒໓໔໕໖໗໘໙')


# ----------------------------------------------------------------------------
#  Random content
# ----------------------------------------------------------------------------

class Content:
    """Random text drawn from a corpus plus document-style tokens."""

    def __init__(self, lines, rng):
        self.lines = [l for l in lines if l.strip()]
        self.rng = rng

    def phrase(self, min_words=2, max_words=8):
        """A run of corpus words (spaces are Lao phrase breaks)."""
        words = []
        target = self.rng.randint(min_words, max_words)
        while len(words) < target:
            words += self.rng.choice(self.lines).split()
        return ' '.join(words[:target])

    def date(self):
        r = self.rng
        d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(2015, 2030)
        style = r.random()
        if style < 0.5:
            s = f'{d} {MONTHS[m - 1]} {y}'
        elif style < 0.8:
            s = f'{d:02d}/{m:02d}/{y}'
        else:
            s = f'{d:02d}.{m:02d}.{y}'
        return s.translate(LAO_DIGITS) if r.random() < 0.05 else s

    def token(self):
        """Digits / Latin tokens common in letters."""
        r = self.rng
        kind = r.random()
        if kind < 0.2:
            return f"{r.randint(1, 9999)}/{self.rng.choice(['ສສກ', 'ກສ', 'ຫກ', 'ກງ', 'ສທ', 'ກຕ'])}"
        if kind < 0.35:
            return f"0{r.choice([20, 21, 30])} {r.randint(10, 99)}{r.randint(100000, 999999)}"
        if kind < 0.45:
            user = ''.join(r.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(r.randint(3, 8)))
            return f"{user}@{r.choice(['gmail.com', 'yahoo.com', 'moes.edu.la', 'gov.la', 'mof.gov.la'])}"
        if kind < 0.55:
            site = ''.join(r.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(r.randint(3, 9)))
            return f"www.{site}.{r.choice(['gov.la', 'edu.la', 'org', 'com'])}"
        if kind < 0.65:
            h = r.randint(7, 16)
            return f"{h:02d}:{r.choice(['00', '30'])}-{h + r.randint(1, 3):02d}:{r.choice(['00', '30'])}"
        if kind < 0.8:
            return '(' + ''.join(r.choice('ABCDEFGHIJKLMNOPRSTUVW') for _ in range(r.randint(2, 5))) + ')'
        if kind < 0.9:
            return f"{r.randint(1, 999):,}".replace(',', '.') + r.choice([' ກີບ', ' ໂດລາ', '%', ''])
        return self.date()

    def sentence(self, min_words=8, max_words=60, token_rate=0.08):
        """Body text: corpus phrases with document tokens mixed in."""
        out = []
        target = self.rng.randint(min_words, max_words)
        while len(out) < target:
            if self.rng.random() < token_rate:
                out.append(self.token())
            else:
                out += self.rng.choice(self.lines).split()
        text = ' '.join(out[:target])
        return text + self.rng.choice(['.', '.', ';', ',', ''])


# ----------------------------------------------------------------------------
#  Fonts
# ----------------------------------------------------------------------------

class FontSet:
    """Lao font families (regular + bold when available) and Latin/digit fonts."""

    def __init__(self, font_paths, sec_paths):
        usable = [p for p in font_paths if font_shapes_correctly(p)]
        skipped = [p for p in font_paths if p not in usable]
        self.skipped = skipped
        if not usable:
            raise SystemExit("No Lao font renders correctly with HarfBuzz shaping "
                             "(need Pillow with libraqm and e.g. fonts/saysettha_ot.ttf).")
        self.families = []
        for p in usable:
            name = os.path.basename(p).lower()
            if 'bold' in name:
                continue
            stem = name.replace('-regular', '').replace('_regular', '').split('.')[0]
            bold = next((q for q in usable if 'bold' in os.path.basename(q).lower()
                         and os.path.basename(q).lower().startswith(stem.split('-')[0])), None)
            self.families.append((p, bold))
        if not self.families:
            self.families = [(p, None) for p in usable]
        self.sec = [p for p in sec_paths if os.path.exists(p)] or [usable[0]]
        self.sec_regular = [p for p in self.sec if not any(k in os.path.basename(p).lower()
                                                            for k in ('bold', 'italic'))] or self.sec
        self.sec_bold = [p for p in self.sec if 'bold' in os.path.basename(p).lower()
                         and 'italic' not in os.path.basename(p).lower()]
        self.sec_italic = [p for p in self.sec if 'italic' in os.path.basename(p).lower()]


class Style:
    def __init__(self, lao, sec, size, color=(0, 0, 0), bold=False, fake_bold=False, underline=False):
        self.lao, self.sec, self.size = lao, sec, int(size)
        self.color, self.bold, self.fake_bold, self.underline = color, bold, fake_bold, underline

    def with_(self, **kw):
        s = Style(self.lao, self.sec, self.size, self.color, self.bold, self.fake_bold, self.underline)
        for k, v in kw.items():
            setattr(s, k, v)
        return s


# ----------------------------------------------------------------------------
#  Page builder
# ----------------------------------------------------------------------------

class Page:
    """A4 page with a text cursor; records every drawn word and its box."""

    def __init__(self, dpi, rng, paper=(255, 255, 255)):
        self.dpi = dpi
        self.rng = rng
        self.W, self.H = int(round(8.27 * dpi)), int(round(11.69 * dpi))
        self.img = Image.new('RGB', (self.W, self.H), paper)
        self.draw = ImageDraw.Draw(self.img)
        self.overlays = []   # (layer RGB) multiplied in at the end: stamps, signature, emblem
        self.words = []      # {'text', 'box'}
        self.cm = dpi / 2.54

    # --- text measurement / drawing -------------------------------------
    def _runs(self, text, style):
        """[(run, font, is_lao)] for a word; Latin/digits in the secondary font."""
        out = []
        for lao, run in _script_runs(text):
            path = style.lao if lao else style.sec
            if not font_covers(path, run):
                path = style.lao
            out.append((run, _load_font(path, style.size, _HAS_RAQM), lao))
        return out

    def width(self, text, style):
        return sum(f.getlength(run) for run, f, _ in self._runs(text, style))

    def space(self, style):
        return _load_font(style.lao, style.size, _HAS_RAQM).getlength(' ')

    def draw_word(self, x, baseline, word, style, record=True):
        sw = max(1, style.size // 28) if style.fake_bold else 0
        box = None
        for run, f, lao in self._runs(word, style):
            kw = {'language': 'lo'} if lao and _HAS_RAQM else {}
            self.draw.text((x, baseline), run, font=f, fill=style.color, anchor='ls',
                           stroke_width=sw, stroke_fill=style.color, **kw)
            bb = self.draw.textbbox((x, baseline), run, font=f, anchor='ls', stroke_width=sw, **kw)
            box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]),
                                          max(box[2], bb[2]), max(box[3], bb[3]))
            x += f.getlength(run)
        if style.underline and box:
            y = baseline + max(2, style.size // 9)
            self.draw.line((box[0], y, box[2], y), fill=style.color, width=max(1, style.size // 22))
        if record and box:
            self.words.append({'text': word, 'box': tuple(float(v) for v in box)})
        return x

    def draw_line(self, x, baseline, words, style, justify_to=None, styles=None):
        """Draw words separated by spaces; optionally justify to a right edge."""
        if not words:
            return
        styles = styles or [style] * len(words)
        sp = self.space(style)
        gap = sp
        if justify_to is not None and len(words) > 1:
            natural = sum(self.width(w, s) for w, s in zip(words, styles)) + sp * (len(words) - 1)
            extra = (justify_to - x - natural) / (len(words) - 1)
            if 0 < extra < 1.5 * sp:     # like a word processor; otherwise leave ragged
                gap = sp + extra
        for w, s in zip(words, styles):
            x = self.draw_word(x, baseline, w, s) + gap

    def wrap(self, text, style, width):
        """Greedy wrap at spaces; over-long words are split at Lao cluster boundaries."""
        sp = self.space(style)
        lines, cur, cur_w = [], [], 0.0
        for word in text.split():
            ww = self.width(word, style)
            while ww > width:   # split an over-long word
                head = ''
                for cl in _split_grapheme_clusters(word):
                    if self.width(head + cl, style) > width - (cur_w + sp if cur else 0):
                        break
                    head += cl
                if not head:
                    if cur:
                        lines.append(cur)
                        cur, cur_w = [], 0.0
                        continue
                    head = word[:1]
                cur.append(head)
                lines.append(cur)
                cur, cur_w = [], 0.0
                word = word[len(head):]
                ww = self.width(word, style)
            if cur and cur_w + sp + ww > width:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur_w = cur_w + sp + ww if cur else ww
            cur.append(word)
        if cur:
            lines.append(cur)
        return lines

    # --- graphics (multiplied in, so text under them stays visible) -----
    def _layer(self):
        layer = Image.new('RGB', (self.W, self.H), (255, 255, 255))
        self.overlays.append(layer)
        return layer

    def emblem(self, cx, top, size):
        rng = self.rng
        d = ImageDraw.Draw(self._layer())
        g = rng.randint(70, 140)
        r = size / 2
        cy = top + r
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(g, g, g), width=max(3, int(size / 40)))
        for k in range(rng.randint(20, 40)):          # rays / texture
            a = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0.2, 0.95) * r
            d.line((cx, cy, cx + rr * math.cos(a), cy + rr * math.sin(a)), fill=(g, g, g),
                   width=rng.randint(2, 6))
        d.pieslice((cx - 0.6 * r, cy - 0.3 * r, cx + 0.6 * r, cy + 0.9 * r), 180, 360, fill=(g + 40,) * 3)
        d.rectangle((cx - 0.7 * r, cy + 0.35 * r, cx + 0.7 * r, cy + 0.55 * r), fill=(g,) * 3)

    def stamp(self, cx, cy, radius, colour, ring_text, font_path):
        """Round official stamp: two rings, a star, lettering around the ring."""
        rng = self.rng
        layer = self._layer()
        d = ImageDraw.Draw(layer)
        w = max(3, int(radius / 28))
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=colour, width=w)
        ri = radius * 0.68
        d.ellipse((cx - ri, cy - ri, cx + ri, cy + ri), outline=colour, width=max(2, w - 1))
        star = [(cx + 0.18 * radius * math.cos(math.pi / 2 + k * 4 * math.pi / 5),
                 cy - 0.18 * radius * math.sin(math.pi / 2 + k * 4 * math.pi / 5)) for k in range(5)]
        d.polygon(star, fill=colour)
        f = _load_font(font_path, int(radius * 0.2), _HAS_RAQM)
        chars = [c for c in _split_grapheme_clusters(ring_text) if not c.isspace()]
        span = rng.uniform(1.2, 1.7) * math.pi
        start = math.pi / 2 + span / 2
        for k, ch in enumerate(chars):
            a = start - span * k / max(1, len(chars) - 1)
            tile = Image.new('L', (int(radius * 0.4), int(radius * 0.4)), 0)
            ImageDraw.Draw(tile).text((tile.width / 2, tile.height / 2), ch, font=f, fill=255, anchor='mm')
            tile = tile.rotate(math.degrees(a) - 90, resample=Image.BICUBIC)
            px = cx + 0.84 * radius * math.cos(a) - tile.width / 2
            py = cy - 0.84 * radius * math.sin(a) - tile.height / 2
            layer.paste(Image.new('RGB', tile.size, colour), (int(px), int(py)), tile)

    def signature(self, x0, y0, width, height, colour):
        rng = self.rng
        d = ImageDraw.Draw(self._layer())
        pts = [(x0 + rng.uniform(0, 0.3) * width, y0 + rng.uniform(0.3, 0.7) * height)]
        for _ in range(rng.randint(4, 8)):
            pts.append((pts[-1][0] + rng.uniform(-0.1, 0.3) * width, y0 + rng.uniform(0, 1) * height))
        pts.append((x0 + width, y0 + rng.uniform(0.4, 0.8) * height))
        smooth = []
        for i in range(len(pts) - 1):                   # simple curve through the points
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]
            for s in np.linspace(0, 1, 12):
                smooth.append((xa + (xb - xa) * s, ya + (yb - ya) * s + math.sin(s * math.pi) * rng.uniform(-8, 8)))
        d.line(smooth, fill=colour, width=rng.randint(3, 6), joint='curve')

    def finish(self):
        """Multiply graphics layers into the page (ink under a stamp stays visible)."""
        arr = np.asarray(self.img).astype(np.float32)
        for layer in self.overlays:
            arr *= np.asarray(layer).astype(np.float32) / 255.0
        self.img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return self.img


# ----------------------------------------------------------------------------
#  Letter template
# ----------------------------------------------------------------------------

def make_letter(content, fonts, rng, dpi=300):
    """Lay out one official-letter page. Returns (page_image, words)."""
    page = Page(dpi, rng)
    cm, pt = page.cm, dpi / 72.0
    lao, bold = rng.choice(fonts.families)
    sec = rng.choice(fonts.sec_regular)
    sec_bold = rng.choice(fonts.sec_bold) if fonts.sec_bold else sec
    ink_level = rng.randint(0, 50)          # real scan: ink ≈ 35–67 after scanning
    ink = (ink_level,) * 3
    body_pt = rng.uniform(11.5, 14.0)
    body = Style(lao, sec, body_pt * pt, ink)
    heading = body.with_(size=int(body.size * rng.uniform(1.0, 1.15)))
    title = body.with_(size=int(body.size * rng.uniform(1.15, 1.45)))
    footer = body.with_(size=int(body.size * rng.uniform(0.72, 0.85)))

    def boldify(s):
        if bold and rng.random() < 0.7:
            return s.with_(lao=bold, sec=sec_bold, bold=True)
        return s.with_(fake_bold=True, sec=sec_bold, bold=True)

    pitch = rng.uniform(1.3, 1.6)
    left = rng.uniform(2.5, 3.5) * cm
    right = page.W - rng.uniform(1.5, 2.5) * cm
    bottom = page.H - rng.uniform(1.5, 2.5) * cm
    y = rng.uniform(1.0, 2.0) * cm
    asc = lambda s: _load_font(s.lao, s.size, _HAS_RAQM).getmetrics()[0]

    def newline(s, k=1.0):
        nonlocal y
        y += s.size * pitch * k

    def room(s):
        return y + s.size * 1.6 < bottom - 3.5 * cm   # keep space for the signature

    # Emblem + motto
    if rng.random() < 0.75:
        size = rng.uniform(2.2, 3.2) * cm
        page.emblem(page.W / 2, y, size)
        y += size + 0.2 * cm
    if rng.random() < 0.9:
        s = boldify(heading) if rng.random() < 0.4 else heading
        for text in MOTTO:
            text = normalize_lao_text(text)
            newline(s)
            page.draw_line(page.W / 2 - page.width(text, s) / 2, y, text.split(), s)
        newline(s, 0.6)

    # Ministry block (left) + number and date (right)
    newline(body)
    org = [normalize_lao_text(rng.choice(MINISTRIES))]
    if rng.random() < 0.8:
        org.append(normalize_lao_text(rng.choice(DEPARTMENTS)))
    number = f"{rng.randint(1, 9999)}"
    suffix = rng.choice(['ຫກ', 'ສສກ', 'ກສ', 'ກງ', 'ກຕ', 'ສທ'])
    for i, text in enumerate(org):
        page.draw_line(left, y, text.split(), body)
        if i == 0:
            if rng.random() < 0.4:   # red stamped number above the label
                red = (rng.randint(170, 220), rng.randint(20, 60), rng.randint(30, 70))
                num_style = body.with_(sec=sec_bold, size=int(body.size * 1.3), color=red)
                page.draw_line(right - page.width(number, num_style) - 2 * cm, y, [number], num_style)
                label = [normalize_lao_text('ເລກທີ'), f'/{suffix}']
                newline(body)
                x = right - page.width(' '.join(label), body) - 1.5 * cm
                page.draw_word(x, y, label[0], body)
                page.draw_word(right - page.width(label[1], body), y, label[1], body)
                continue
            text_r = f"{normalize_lao_text('ເລກທີ')} {number}/{suffix}"
            page.draw_line(right - page.width(text_r, body), y, text_r.split(), body)
        newline(body)
    place = f"{normalize_lao_text(rng.choice(PLACES))}, {normalize_lao_text('ວັນທີ')}"
    day, month, year = str(rng.randint(1, 28)), normalize_lao_text(rng.choice(MONTHS)), str(rng.randint(2015, 2030))
    words = place.split() + [day, month, year]
    styles = [body] * len(words)
    if rng.random() < 0.3:      # handwritten day in blue pen
        pen = (rng.randint(20, 60), rng.randint(50, 90), rng.randint(170, 220))
        styles[len(place.split())] = body.with_(sec=rng.choice(fonts.sec_italic or fonts.sec),
                                                 size=int(body.size * 1.25), color=pen)
    width = sum(page.width(w, s) for w, s in zip(words, styles)) + page.space(body) * (len(words) - 1)
    page.draw_line(right - width, y, words, body, styles=styles)
    newline(body, 1.6)

    # Title
    ts = boldify(title)
    t = normalize_lao_text(rng.choice(TITLES))
    page.draw_line(page.W / 2 - page.width(t, ts) / 2, y, t.split(), ts)
    newline(ts, 1.4)

    # ເຖິງ: / ເລື່ອງ: blocks with hanging indent
    label_x = left + rng.uniform(1.5, 3.5) * cm
    text_x = label_x + rng.uniform(1.5, 2.2) * cm
    if rng.random() < 0.7:
        page.draw_word(label_x, y, normalize_lao_text('ເຖິງ:'), body)
        for k in range(rng.randint(1, 4)):
            if not room(body):
                break
            item = f"{normalize_lao_text(rng.choice(GREETING))} {content.phrase(2, 7)}."
            for li, ln in enumerate(page.wrap(item, body, right - text_x - 0.6 * cm)[:2]):
                if li == 0 and rng.random() < 0.8:
                    page.draw_word(text_x - 0.5 * cm, y, '-', body)
                page.draw_line(text_x, y, ln, body)
                newline(body)
    if rng.random() < 0.8 and room(body):
        page.draw_word(label_x, y, normalize_lao_text('ເລື່ອງ:'), body)
        subj = boldify(body) if rng.random() < 0.3 else body
        for ln in page.wrap(content.sentence(6, 25, 0.1), subj, right - text_x)[:3]:
            page.draw_line(text_x, y, ln, subj)
            newline(body)
    newline(body, 0.5)

    # References: "- ອີງຕາມ ..."
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
        if not room(body):
            break
        ref = f"{normalize_lao_text('ອີງຕາມ')} {content.phrase(3, 10)} {normalize_lao_text('ສະບັບເລກທີ')} " \
              f"{content.token()}, {normalize_lao_text('ລົງວັນທີ')} {content.date()};"
        indent = left + rng.uniform(0.5, 1.2) * cm
        page.draw_word(left, y, '-', body)
        for ln in page.wrap(ref, body, right - indent):
            if not room(body):
                break
            page.draw_line(indent, y, ln, body, justify_to=right if rng.random() < 0.5 else None)
            newline(body)
    newline(body, 0.5)

    # Body paragraphs: first-line indent, mostly justified, some bold words
    justify = rng.random() < 0.8
    for _ in range(rng.randint(1, 4)):
        if not room(body):
            break
        para = content.sentence(15, 90, token_rate=rng.uniform(0.03, 0.15))
        first = left + rng.uniform(1.0, 2.0) * cm
        lines = page.wrap(para, body, right - first)
        rest = page.wrap(' '.join(' '.join(l) for l in lines[1:]), body, right - left) if len(lines) > 1 else []
        all_lines = lines[:1] + rest
        for li, ln in enumerate(all_lines):
            if not room(body):
                break
            styles = [boldify(body) if rng.random() < 0.04 else
                      (body.with_(underline=True) if ('@' in w or w.startswith('www.')) and rng.random() < 0.6 else body)
                      for w in ln]
            last = li == len(all_lines) - 1
            page.draw_line(first if li == 0 else left, y, ln, body,
                           justify_to=right if (justify and not last) else None, styles=styles)
            newline(body)
        newline(body, rng.uniform(0.2, 0.6))

    # Optional table
    if rng.random() < 0.25 and room(body):
        cols = rng.randint(2, 4)
        widths = np.array([rng.uniform(1, 3) for _ in range(cols)])
        widths = widths / widths.sum() * (right - left)
        xs = [left] + list(left + np.cumsum(widths))
        row_h = body.size * pitch * 1.3
        top = y
        for r_i in range(rng.randint(2, 6)):
            if not room(body):
                break
            s = boldify(body) if r_i == 0 else body
            for c in range(cols):
                cell = content.phrase(1, 4) if (c or r_i == 0) else str(r_i)
                ln = page.wrap(cell, s, widths[c] - 0.6 * cm)[:1]
                if ln:
                    page.draw_line(xs[c] + 0.3 * cm, y + row_h * 0.7, ln[0], s)
            y += row_h
        for x in xs:
            page.draw.line((x, top, x, y), fill=ink, width=3)
        for k in range(int(round((y - top) / row_h)) + 1):
            page.draw.line((left, top + k * row_h, right, top + k * row_h), fill=ink, width=3)
        newline(body, 0.8)

    # Closing line
    if room(body):
        closing = normalize_lao_text(rng.choice(CLOSINGS))
        for ln in page.wrap(closing, body, right - left - 1.5 * cm):
            page.draw_line(left + rng.uniform(1.0, 2.0) * cm, y, ln, body)
            newline(body)

    # Signature block: title, stamp + signature, name
    y = max(y + body.size, min(y + 1.5 * cm, bottom - 6 * cm))
    sig_cx = right - rng.uniform(3.0, 4.5) * cm
    signer = normalize_lao_text(rng.choice(SIGNERS))
    ss = boldify(body) if rng.random() < 0.6 else body
    page.draw_line(sig_cx - page.width(signer, ss) / 2, y, signer.split(), ss)
    top_sig = y + 0.3 * cm
    if rng.random() < 0.85:
        red = (rng.randint(180, 230), rng.randint(20, 80), rng.randint(40, 90))
        colour = red if rng.random() < 0.85 else (rng.randint(60, 110), rng.randint(40, 80), rng.randint(150, 200))
        radius = rng.uniform(1.6, 2.1) * cm
        page.stamp(sig_cx - rng.uniform(0.8, 1.8) * cm, top_sig + radius * rng.uniform(0.75, 1.0),
                   radius, colour, normalize_lao_text(rng.choice(MINISTRIES)), lao)
    if rng.random() < 0.9:
        pen = (rng.randint(20, 60), rng.randint(50, 90), rng.randint(170, 220)) if rng.random() < 0.7 else ink
        page.signature(sig_cx - 1.5 * cm, top_sig + 0.5 * cm, rng.uniform(3, 5) * cm, 1.8 * cm, pen)
    y = top_sig + rng.uniform(3.6, 4.4) * cm
    name = f"{content.phrase(1, 1)} {content.phrase(1, 1)}"
    name_style = body.with_(color=(rng.randint(170, 220), rng.randint(20, 60), rng.randint(30, 70))) \
        if rng.random() < 0.3 else ss
    page.draw_line(sig_cx - page.width(name, name_style) / 2, y, name.split(), name_style)

    # Footer
    if rng.random() < 0.5:
        fy = page.H - rng.uniform(1.2, 2.0) * cm - 2 * footer.size * 1.3
        if fy > y + body.size * 2:
            page.draw.line((left, fy - footer.size * 1.1, left + (right - left) * rng.uniform(0.6, 1.0),
                            fy - footer.size * 1.1), fill=ink, width=3)
            for k in range(rng.randint(1, 2)):
                text = (content.phrase(4, 10) if k == 0 else
                        f"{normalize_lao_text('ໂທ')} {content.token()}, {content.token()}")
                ln = page.wrap(text, footer, right - left)[:1]
                if ln:
                    page.draw_line(left, fy + k * footer.size * 1.3, ln[0], footer)

    return page.finish(), page.words


# ----------------------------------------------------------------------------
#  Scan simulation
# ----------------------------------------------------------------------------

def scan(img, rng, max_skew=1.5):
    """Simulate printing + scanning. Returns (image, skew_angle) — boxes must be
    rotated with doc_layout.rotate_box(box, skew_angle, size)."""
    W, H = img.size
    arr = np.asarray(img).astype(np.float32)

    # ink spread / print softness
    if rng.random() < 0.35:
        img = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
        arr = np.asarray(img).astype(np.float32)

    # paper tint (multiplicative) and uneven illumination
    d = rng.uniform(0.0, 0.04) if rng.random() < 0.4 else 0.0   # white or slightly warm paper
    tint = np.array([1.0 - 0.2 * d, 1.0 - 0.4 * d, 1.0 - d]) * rng.uniform(0.975, 1.0)
    arr *= tint
    if rng.random() < 0.4:
        g = rng.uniform(0.0, 0.08)
        ramp = np.linspace(1 - g, 1, W if rng.random() < 0.5 else H)
        arr *= ramp[None, :, None] if ramp.size == W else ramp[:, None, None]

    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    paper = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))

    # skew
    angle = rng.uniform(-max_skew, max_skew)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=paper)

    # scanner resolution loss
    if rng.random() < 0.6:
        f = rng.uniform(0.65, 0.95)
        img = img.resize((int(W * f), int(H * f)), Image.BILINEAR).resize((W, H), Image.BILINEAR)

    # sensor noise and specks
    arr = np.asarray(img).astype(np.float32)
    sigma = rng.choice([0, 0, rng.uniform(1, 4), rng.uniform(3, 8)])
    if sigma:
        arr += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, sigma, arr.shape[:2])[..., None]
    n_specks = rng.choice([0, 0, rng.randint(20, 200)])
    for _ in range(n_specks):
        x, y, r = rng.randint(0, W - 4), rng.randint(0, H - 4), rng.randint(1, 3)
        arr[y:y + r, x:x + r] *= rng.uniform(0.2, 0.7)

    # photocopy look: high contrast, gray only
    if rng.random() < 0.1:
        gray = arr.mean(axis=2, keepdims=True)
        arr = np.repeat(np.clip((gray - 128) * 1.6 + 150, 0, 255), 3, axis=2)

    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # JPEG
    if rng.random() < 0.8:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=rng.randint(50, 95))
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
    return img, angle
