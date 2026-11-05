import struct
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class LstmfReader:
    """Reads Tesseract .lstmf binary files."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_bytes(self, n):
        r = self.data[self.offset:self.offset + n]
        self.offset += n
        return r

    def read_uint32(self):
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_int32(self):
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_int16(self):
        return struct.unpack('<h', self.read_bytes(2))[0]

    def read_uint8(self):
        return struct.unpack('<B', self.read_bytes(1))[0]

    def read_int8(self):
        return struct.unpack('<b', self.read_bytes(1))[0]

    def read_string(self):
        length = self.read_uint32()
        return self.read_bytes(length).decode('utf-8', errors='replace') if length else ""

    def read_char_vector(self):
        length = self.read_uint32()
        return self.read_bytes(length) if length else b""

    def read_tbox_vector(self):
        count = self.read_uint32()
        return [(self.read_int16(), self.read_int16(), self.read_int16(), self.read_int16())
                for _ in range(count)]

    def read_string_vector(self):
        count = self.read_uint32()
        return [self.read_string() for _ in range(count)]

    def read_image_data(self):
        return {
            'imagefilename': self.read_string(),
            'page_number': self.read_int32(),
            'image_bytes': self.read_char_vector(),
            'language': self.read_string(),
            'transcription': self.read_string(),
            'boxes': self.read_tbox_vector(),
            'box_texts': self.read_string_vector(),
            'vertical': self.read_int8() != 0,
        }

    def read_document(self):
        count = self.read_uint32()
        pages = []
        for _ in range(count):
            if self.read_uint8():
                pages.append(self.read_image_data())
        return pages

def load_lstmf(filepath):
    with open(filepath, 'rb') as f:
        return LstmfReader(f.read()).read_document()

class TesseractDataset(Dataset):
    def __init__(self, lstmf_files, target_height=48, charset=None):
        self.samples = []
        self.target_height = target_height
        self.charset = charset or {}

        print(f"  Loading {len(lstmf_files)} .lstmf files...")
        for fp in lstmf_files:
            try:
                for page in load_lstmf(fp):
                    if page['image_bytes'] and page['transcription']:
                        self.samples.append((page['image_bytes'], page['transcription']))
            except Exception as e:
                print(f"    Warning: {fp}: {e}")

        if not self.charset:
            chars = set()
            for _, text in self.samples:
                chars.update(text)
            self.charset = {c: i + 1 for i, c in enumerate(sorted(chars))}

        self.inv_charset = {v: k for k, v in self.charset.items()}
        self.num_classes = len(self.charset) + 1
        print(f"  {len(self.samples)} samples, {self.num_classes} classes (incl. CTC blank)")

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return [self.charset[c] for c in text if c in self.charset]

    def decode_labels(self, labels):
        return ''.join(self.inv_charset.get(l, '?') for l in labels if l != 0)

    def __getitem__(self, idx):
        image_bytes, transcription = self.samples[idx]
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert('L')
        except Exception:
            return torch.zeros(1, self.target_height, 100), torch.tensor([1], dtype=torch.long), 100, 1

        w, h = pil.size
        if h != self.target_height and h > 0:
            scale = self.target_height / h
            pil = pil.resize((max(1, int(w * scale)), self.target_height), Image.BILINEAR)

        arr = np.array(pil, dtype=np.float32)
        t = torch.from_numpy((arr - 127.5) / 127.5).unsqueeze(0)
        labels = self.encode_text(transcription) or [1]
        return t, torch.tensor(labels, dtype=torch.long), t.shape[2], len(labels)

# Images are scaled so white paper = +1 and black ink = -1 (see __getitem__).
PAD_WHITE = 1.0
PAD_BLACK = -1.0   # the value used before the P0.1 fix; kept for --padding legacy


class PadCollate:
    """Collate (img, labels, width, label_len) samples, right-padding images to the widest one.

    White padding (the default) continues the paper background, so layers that read the line
    right-to-left (Lrx, Lbx) or look at the whole line (attention, SE) do not start on a black
    bar. Pass PAD_BLACK to reproduce runs made before this fix. A class rather than a closure
    so DataLoader workers can pickle it (spawn start method on macOS / Windows).
    """
    def __init__(self, pad_value=PAD_WHITE):
        self.pad_value = pad_value

    def __call__(self, batch):
        imgs, labels, ilens, tlens = zip(*batch)
        max_w = max(i.shape[2] for i in imgs)
        padded = [F.pad(i, (0, max_w - i.shape[2]), value=self.pad_value) for i in imgs]
        return (torch.stack(padded), torch.cat(labels),
                torch.tensor(ilens, dtype=torch.long), torch.tensor(tlens, dtype=torch.long))


def make_collate_fn(pad_value=PAD_WHITE):
    return PadCollate(pad_value)


collate_fn = make_collate_fn(PAD_WHITE)
legacy_collate_fn = make_collate_fn(PAD_BLACK)
