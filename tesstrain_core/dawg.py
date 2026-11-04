import os
import re
import math
import json
import glob
from collections import defaultdict
import unicodedata

class TrieDAWG:
    """Trie-based DAWG for word lookup and prefix matching."""

    def __init__(self):
        self.root = {}
        self.word_count = 0

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = True 
        self.word_count += 1

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return '$' in node

    def has_prefix(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True

    def get_completions(self, prefix, max_results=10):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        results = []
        self._collect(node, prefix, results, max_results)
        return results

    def _collect(self, node, prefix, results, max_results):
        if len(results) >= max_results:
            return
        if '$' in node:
            results.append(prefix)
        for ch, child in sorted(node.items()):
            if ch != '$':
                self._collect(child, prefix + ch, results, max_results)

    @staticmethod
    def load_from_file(filepath):
        dawg = TrieDAWG()
        if not os.path.exists(filepath):
            return dawg
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    dawg.insert(word)
        return dawg


class NumberDAWG:
    PATTERNS = [
        r'^\d+$',                     # integers: 123
        r'^\d+\.\d+$',               # decimals: 12.34
        r'^\d{1,3}(,\d{3})*$',       # comma-separated: 1,000,000
        r'^\d{1,3}(,\d{3})*\.\d+$',  # comma+decimal: 1,000.50
        r'^\d+%$',                    # percentage: 50%
        r'^\d+/\d+$',                # fractions: 3/4
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',  # dates: 01/02/2024
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',    # ISO dates: 2024-01-02
        r'^\+?\d[\d\-\s]{6,14}$',    # phone numbers
        r'^[\$€£¥฿₭]\s?\d',          # currency prefix
        r'^\d.*[\$€£¥฿₭]$',          # currency suffix
        r'^\d{1,2}:\d{2}(:\d{2})?$', # time: 12:30
        r'^#\d+$',                    # ordinals: #1
    ]

    def __init__(self):
        self._compiled = [re.compile(p) for p in self.PATTERNS]

    def is_number(self, text):
        return any(p.match(text) for p in self._compiled)

    def is_number_prefix(self, text):
        if not text:
            return False
        first = text[0]
        if first.isdigit() or first in '$€£¥฿₭+#':
            return True
        return False


class PuncDAWG:
    OPEN_PUNCS = set(['(', '[', '{', '"', "'"])
    CLOSE_PUNCS = set([')', ']', '}', '"', "'", '.', '!', '?', ';', ':', ','])
    MID_PUNCS = set(['-', '/'])
    NO_REPEAT = set(['.', ',', ';', ':', '!', '?'])

    def validate(self, text):
        if not text:
            return 1.0
        score = 1.0
        for i, ch in enumerate(text):
            if ch in self.NO_REPEAT and i > 0 and text[i-1] == ch:
                if not (ch == '.' and i > 1 and text[i-2] == '.'): 
                    score *= 0.5
            if ch in self.CLOSE_PUNCS and i == 0 and len(text) > 1:
                score *= 0.8
            if ch in self.OPEN_PUNCS and i == len(text) - 1 and len(text) > 1:
                score *= 0.8  
        return score

    def is_valid_boundary(self, prev_char, next_char):
        if prev_char in self.OPEN_PUNCS and next_char == ' ':
            return False
        if next_char in self.CLOSE_PUNCS and prev_char == ' ':
            return False
        return True


class DAWGDecoder:
    def __init__(self, word_dawg=None, number_dawg=None, punc_dawg=None,
                 inv_charset=None, word_bonus=1.0, number_bonus=0.5,
                 punc_penalty=-0.5):
        self.word_dawg = word_dawg or TrieDAWG()
        self.number_dawg = number_dawg or NumberDAWG()
        self.punc_dawg = punc_dawg or PuncDAWG()
        self.inv_charset = inv_charset or {}
        self.word_bonus = word_bonus
        self.number_bonus = number_bonus
        self.punc_penalty = punc_penalty

    def _labels_to_text(self, labels):
        return ''.join(self.inv_charset.get(l, '') for l in labels)

    def _get_current_word(self, labels):
        text = self._labels_to_text(labels)
        words = text.split()
        return words[-1] if words else text

    def score_prefix(self, prefix_labels):
        if not prefix_labels or not self.inv_charset:
            return 0.0

        current_word = self._get_current_word(prefix_labels)
        if not current_word:
            return 0.0

        score = 0.0
        if self.word_dawg.word_count > 0:
            if self.word_dawg.has_prefix(current_word):
                score += self.word_bonus * 0.1
            elif len(current_word) > 2:
                score -= self.word_bonus * 0.05

        if self.number_dawg.is_number_prefix(current_word):
            score += self.number_bonus * 0.05

        text = self._labels_to_text(prefix_labels)
        score += self._score_lm(text) * 0.1

        return score

    def score_final(self, prefix_labels):
        if not prefix_labels or not self.inv_charset:
            return 0.0

        text = self._labels_to_text(prefix_labels)
        words = text.split()
        score = 0.0

        for word in words:
            if not word:
                continue
            if self.word_dawg.word_count > 0 and self.word_dawg.search(word):
                score += self.word_bonus
            elif self.number_dawg.is_number(word):
                score += self.number_bonus
            elif self.word_dawg.word_count > 0 and len(word) > 1:
                score -= self.word_bonus * 0.3

        punc_score = self.punc_dawg.validate(text)
        if punc_score < 1.0:
            score += self.punc_penalty * (1.0 - punc_score)

        score += self._score_lm(text)

        return score

    @classmethod
    def from_files(cls, inv_charset, wordlist=None, numberslist=None,
                   punclist=None, word_bonus=1.0, lm=None, lm_weight=0.5):
        word_dawg = TrieDAWG()
        if wordlist and os.path.exists(wordlist):
            word_dawg = TrieDAWG.load_from_file(wordlist)
            print(f"  Word DAWG: {word_dawg.word_count} words from {wordlist}")

        number_dawg = NumberDAWG()
        punc_dawg = PuncDAWG()

        decoder = cls(word_dawg=word_dawg, number_dawg=number_dawg,
                      punc_dawg=punc_dawg, inv_charset=inv_charset,
                      word_bonus=word_bonus)
        decoder.lm = lm
        decoder.lm_weight = lm_weight
        return decoder

    def _score_lm(self, text):
        if not hasattr(self, 'lm') or self.lm is None:
            return 0.0
        weight = getattr(self, 'lm_weight', 0.5)
        return self.lm.score_sequence(text) * weight


class NgramLM:
    def __init__(self, char_order=3, word_order=2):
        self.char_order = char_order
        self.word_order = word_order
        self.char_ngrams = {}
        for n in range(1, char_order + 1):
            self.char_ngrams[n] = defaultdict(lambda: defaultdict(int))
        self.word_ngrams = {}
        for n in range(1, word_order + 1):
            self.word_ngrams[n] = defaultdict(lambda: defaultdict(int))
        self.char_totals = defaultdict(int) 
        self.word_totals = defaultdict(int)
        self.total_chars = 0
        self.total_words = 0
        self.vocab_size = 0

    def train_from_text(self, text):
        padded = '^' * (self.char_order - 1) + text + '$'
        for i in range(len(padded) - self.char_order + 1):
            for n in range(1, self.char_order + 1):
                if i + n <= len(padded):
                    context = padded[max(0, i):i + n - 1] if n > 1 else ''
                    char = padded[i + n - 1]
                    self.char_ngrams[n][context][char] += 1
                    self.char_totals[context] += 1
        self.total_chars += len(text)

        words = text.split()
        padded_words = ['<s>'] * (self.word_order - 1) + words + ['</s>']
        for i in range(len(padded_words) - self.word_order + 1):
            for n in range(1, self.word_order + 1):
                if i + n <= len(padded_words):
                    context = ' '.join(padded_words[max(0, i):i + n - 1]) if n > 1 else ''
                    word = padded_words[i + n - 1]
                    self.word_ngrams[n][context][word] += 1
                    self.word_totals[context] += 1
        self.total_words += len(words)

        unique = set(text)
        self.vocab_size = max(self.vocab_size, len(unique))

    def train_from_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"  Warning: LM corpus not found: {filepath}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.train_from_text(line)
        print(f"  LM trained: {self.total_chars} chars, {self.total_words} words, "
              f"{self.char_order}-gram chars, {self.word_order}-gram words")

    def train_from_gt_dir(self, gt_dir):
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.gt.txt')))
        if not gt_files:
            print(f"  Warning: No .gt.txt files found in {gt_dir}")
            return
        for f in gt_files:
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read().strip()
                if text:
                    self.train_from_text(text)
        print(f"  LM trained from {len(gt_files)} .gt.txt files: "
              f"{self.total_chars} chars, {self.total_words} words")

    def _char_log_prob(self, char, context):
        for n in range(min(len(context) + 1, self.char_order), 0, -1):
            ctx = context[-(n-1):] if n > 1 else ''
            if ctx in self.char_ngrams[n] and char in self.char_ngrams[n][ctx]:
                count = self.char_ngrams[n][ctx][char]
                total = self.char_totals[ctx]
                if total > 0:
                    k = 0.1
                    vocab = max(self.vocab_size, 100)
                    prob = (count + k) / (total + k * vocab)
                    return math.log(max(prob, 1e-10))

        vocab = max(self.vocab_size, 100)
        return math.log(1.0 / vocab)

    def _word_log_prob(self, word, context):
        for n in range(min(len(context.split()) + 1, self.word_order), 0, -1):
            ctx_words = context.split()
            ctx = ' '.join(ctx_words[-(n-1):]) if n > 1 else ''
            if ctx in self.word_ngrams[n] and word in self.word_ngrams[n][ctx]:
                count = self.word_ngrams[n][ctx][word]
                total = self.word_totals[ctx]
                if total > 0:
                    k = 0.1
                    vocab = max(self.total_words, 1000)
                    prob = (count + k) / (total + k * vocab)
                    return math.log(max(prob, 1e-10))

        return math.log(1e-6)

    def score_sequence(self, text):
        if not text or self.total_chars == 0:
            return 0.0

        total_lp = 0.0
        context = '^' * (self.char_order - 1)
        for char in text:
            total_lp += self._char_log_prob(char, context)
            context = (context + char)[-(self.char_order - 1):]

        return total_lp / max(len(text), 1)

    def score_words(self, text):
        if not text or self.total_words == 0:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        total_lp = 0.0
        context = '<s>'
        for word in words:
            total_lp += self._word_log_prob(word, context)
            context = word

        return total_lp / max(len(words), 1)

    def save(self, filepath):
        data = {
            'char_order': self.char_order,
            'word_order': self.word_order,
            'total_chars': self.total_chars,
            'total_words': self.total_words,
            'vocab_size': self.vocab_size,
            'char_ngrams': {},
            'word_ngrams': {},
        }
        for n in self.char_ngrams:
            data['char_ngrams'][str(n)] = {
                ctx: dict(counts) for ctx, counts in self.char_ngrams[n].items()
            }
        for n in self.word_ngrams:
            data['word_ngrams'][str(n)] = {
                ctx: dict(counts) for ctx, counts in self.word_ngrams[n].items()
            }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  LM saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lm = cls(char_order=data['char_order'], word_order=data['word_order'])
        lm.total_chars = data['total_chars']
        lm.total_words = data['total_words']
        lm.vocab_size = data['vocab_size']
        for n_str, contexts in data['char_ngrams'].items():
            n = int(n_str)
            for ctx, counts in contexts.items():
                for char, count in counts.items():
                    lm.char_ngrams[n][ctx][char] = count
                    lm.char_totals[ctx] += count
        for n_str, contexts in data['word_ngrams'].items():
            n = int(n_str)
            for ctx, counts in contexts.items():
                for word, count in counts.items():
                    lm.word_ngrams[n][ctx][word] = count
                    lm.word_totals[ctx] += count
        print(f"  LM loaded: {filepath} ({lm.total_chars} chars, {lm.total_words} words)")
        return lm

    @classmethod
    def from_gt_dir(cls, gt_dir, char_order=3, word_order=2):
        lm = cls(char_order=char_order, word_order=word_order)
        lm.train_from_gt_dir(gt_dir)
        return lm


class UnicodeRecoder:
    ZERO_WIDTH = set([
        '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff', '\u2060', '\u00ad',
    ])

    LIGATURES = {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl',
        '\u0132': 'IJ', '\u0133': 'ij', '\u00e6': 'ae', '\u0153': 'oe',
    }

    def __init__(self, normalization='NFC'):
        self.normalization = normalization
        self.char_to_id = {}
        self.id_to_char = {}
        self.sequences = {}
        self.next_id = 1

    def normalize(self, text):
        text = ''.join(c for c in text if c not in self.ZERO_WIDTH)
        for lig, decomp in self.LIGATURES.items():
            text = text.replace(lig, decomp)
        text = unicodedata.normalize(self.normalization, text)
        return text

    def build_from_charset(self, charset):
        self.char_to_id = {}
        self.id_to_char = {}
        self.sequences = {}
        for char, cid in charset.items():
            if cid == 0:
                continue
            norm = self.normalize(char)
            if not norm:
                norm = char
            self.char_to_id[norm] = cid
            self.id_to_char[cid] = norm
            if len(norm) > 1:
                self.sequences[norm] = cid
        self.next_id = max(self.id_to_char.keys()) + 1 if self.id_to_char else 1

    def encode(self, text):
        text = self.normalize(text)
        labels = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                seq = text[i:i + length]
                if seq in self.char_to_id:
                    labels.append(self.char_to_id[seq])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self.char_to_id:
                    labels.append(self.char_to_id[ch])
                i += 1
        return labels

    def decode(self, labels):
        chars = []
        for label in labels:
            if label == 0:
                continue
            ch = self.id_to_char.get(label, '')
            if ch:
                chars.append(ch)
        return ''.join(chars)

    def recode_charset(self, charset):
        self.build_from_charset(charset)
        new_charset = {}
        new_id = 1 
        for norm_char in sorted(set(self.char_to_id.keys())):
            if norm_char not in new_charset:
                new_charset[norm_char] = new_id
                new_id += 1
        self.char_to_id = dict(new_charset)
        self.id_to_char = {v: k for k, v in new_charset.items()}
        self.next_id = new_id
        return new_charset

    def get_stats(self):
        multi = sum(1 for c in self.char_to_id if len(c) > 1)
        return {
            'total_chars': len(self.char_to_id),
            'multi_codepoint': multi,
            'single_codepoint': len(self.char_to_id) - multi,
            'normalization': self.normalization,
        }

    def save(self, filepath):
        data = {
            'normalization': self.normalization,
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
            'sequences': self.sequences,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  Recoder saved: {filepath} ({len(self.char_to_id)} entries)")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recoder = cls(normalization=data.get('normalization', 'NFC'))
        recoder.char_to_id = data['char_to_id']
        recoder.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        recoder.sequences = data.get('sequences', {})
        recoder.next_id = max(recoder.id_to_char.keys()) + 1 if recoder.id_to_char else 1
        print(f"  Recoder loaded: {filepath} ({len(recoder.char_to_id)} entries)")
        return recoder

    @classmethod
    def from_charset(cls, charset, normalization='NFC'):
        recoder = cls(normalization=normalization)
        recoder.build_from_charset(charset)
        return recoder
