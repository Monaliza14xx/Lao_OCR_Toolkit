import re
import math
import torch
import torch.nn as nn

class CollapseH(nn.Module):
    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            return x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        return x

def _seq_mask(lengths, T, device):
    """(B, T) bool, True on the real frames of each line."""
    return torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).clamp(max=T).unsqueeze(1)


def _reverse_within(x, lengths):
    """Reverse each sequence of x (B, T, C) inside its own length; padding stays at the end."""
    T = x.size(1)
    t = torch.arange(T, device=x.device).unsqueeze(0)
    L = lengths.to(x.device).clamp(1, T).unsqueeze(1)
    idx = torch.where(t < L, L - 1 - t, t)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


def _packed_lstm(lstm, x, lengths):
    """Run a (bidirectional) LSTM on the real frames only; padded outputs are 0."""
    T = x.size(1)
    L = lengths.cpu().clamp(1, T)
    packed = nn.utils.rnn.pack_padded_sequence(x, L, batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    return nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)[0]


class BiLSTM(nn.Module):
    uses_lengths = True

    def __init__(self, inp, hid):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True, bidirectional=True)
    def forward(self, x, lengths=None):
        if lengths is not None:
            return _packed_lstm(self.lstm, x, lengths)
        return self.lstm(x)[0]

class BiLSTMProjection(nn.Module):
    """Lbpx<hid>,<out>: BiLSTM followed by a linear projection (no activation).

    Matches the CRNN BidirectionalLSTM pattern:
        LSTM(in, hid, bidirectional=True) → Linear(hid*2, out)

    Input : (B, W, in)
    Output: (B, W, out)
    """
    uses_lengths = True

    def __init__(self, inp, hid, out):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hid * 2, out)
    def forward(self, x, lengths=None):
        h = _packed_lstm(self.lstm, x, lengths) if lengths is not None else self.lstm(x)[0]
        return self.projection(h)

class FwdLSTM(nn.Module):
    def __init__(self, inp, hid):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True)
    def forward(self, x):
        return self.lstm(x)[0]

class RevLSTM(nn.Module):
    uses_lengths = True

    def __init__(self, inp, hid):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True)
    def forward(self, x, lengths=None):
        if lengths is not None:      # start at each line's own last frame, not at the padding
            return _reverse_within(self.lstm(_reverse_within(x, lengths))[0], lengths)
        return torch.flip(self.lstm(torch.flip(x, [1]))[0], [1])

class SumLSTM(nn.Module):
    """Lfys: collapse height via LSTM summary."""
    def __init__(self, inp, hid):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True)
    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
            _, (hn, _) = self.lstm(x)
            return hn.squeeze(0).view(B, W, -1)
        return self.lstm(x)[0]


class AttentionCollapseY(nn.Module):
    """Atys<D>: Collapse height via additive (Bahdanau) attention over vertical positions.

    For each horizontal column, the H vertical feature vectors are treated as a
    sequence. A small MLP scores each position; softmax-normalised weights then
    produce a weighted-sum context vector of dimension D.

    This gives the model a learnable, input-dependent selection of *which* rows
    to emphasise per column — critical for the morphological intersection problem
    where lower vowels and consonant descenders occupy the same vertical span.

    Input : (B, C, H, W)  — CNN feature map
    Output: (B, W, D)     — same shape as SumLSTM, drop-in compatible
    """
    def __init__(self, inp_c: int, out_d: int):
        super().__init__()
        self.value_proj = nn.Linear(inp_c, out_d)
        self.query      = nn.Linear(inp_c, out_d)
        self.key        = nn.Linear(inp_c, out_d)
        self.energy     = nn.Linear(out_d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Treat each column as an independent sequence of H vertical positions
            # x_col: (B*W, H, C)
            x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)

            # Additive attention scoring
            q   = torch.tanh(self.query(x_col))              # (B*W, H, D)
            k   = torch.tanh(self.key(x_col))                # (B*W, H, D)
            e   = self.energy(q + k).squeeze(-1)             # (B*W, H)
            alpha = torch.softmax(e, dim=-1).unsqueeze(-1)   # (B*W, H, 1)

            # Weighted-sum collapse
            v       = self.value_proj(x_col)                 # (B*W, H, D)
            context = (alpha * v).sum(dim=1)                 # (B*W, D)
            return context.view(B, W, -1)                    # (B, W, D)
        return x


class MultiHeadAttentionCollapseY(nn.Module):
    """Matys<D>: Collapse height via multi-head self-attention over vertical positions.

    Uses `num_heads` independent attention heads, each learning a different
    vertical focus pattern. In the Lao script context, separate heads can
    specialise for each of the four tiers: lower vowel, consonant baseline,
    upper vowel, and tone mark — allowing explicit disentanglement of
    morphologically overlapping components.

    A learnable collapse query (one per head) attends over the H key vectors,
    producing a single weighted context per head; the heads are concatenated
    and projected to dimension D.

    Input : (B, C, H, W)  — CNN feature map
    Output: (B, W, D)     — drop-in compatible with SumLSTM / AttentionCollapseY
    """
    def __init__(self, inp_c: int, out_d: int, num_heads: int = 4):
        super().__init__()
        if out_d % num_heads != 0:
            raise ValueError(
                f"MultiHeadAttentionCollapseY: out_d ({out_d}) must be divisible "
                f"by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim  = out_d // num_heads
        self.out_d     = out_d
        self.scale     = math.sqrt(self.head_dim)

        self.kv_proj  = nn.Linear(inp_c, 2 * out_d)          # keys + values
        self.out_proj = nn.Linear(out_d, out_d)
        # Learnable per-head collapse query: (1, 1, num_heads, head_dim)
        self.collapse_q = nn.Parameter(
            torch.empty(1, 1, num_heads, self.head_dim)
        )
        nn.init.xavier_uniform_(self.collapse_q.view(1, -1, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            # x_col: (B*W, H, C)
            x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)

            BW = B * W
            kv  = self.kv_proj(x_col)                          # (BW, H, 2*D)
            k, v = kv.split(self.out_d, dim=-1)                # each: (BW, H, D)

            # Reshape for multi-head: (BW, H, heads, head_dim)
            k = k.view(BW, H, self.num_heads, self.head_dim)
            v = v.view(BW, H, self.num_heads, self.head_dim)

            # Expand learnable query: (BW, 1, heads, head_dim)
            cq = self.collapse_q.expand(BW, -1, -1, -1)

            # Attention scores over H positions: (BW, H, heads)
            attn = (cq * k).sum(dim=-1) / self.scale
            alpha = torch.softmax(attn, dim=1).unsqueeze(-1)   # (BW, H, heads, 1)

            # Weighted sum and flatten heads: (BW, D)
            context = (alpha * v).sum(dim=1)                   # (BW, heads, head_dim)
            context = context.view(BW, self.out_d)
            context = self.out_proj(context)                   # (BW, D)
            return context.view(B, W, -1)                      # (B, W, D)
        return x


class HybridCollapseY(nn.Module):
    """Hybys<D>: Hybrid Recurrent + Multi-Head Attention Vertical Collapse.

    Combines vertical recurrent sequence context (BiLSTM over height H) with
    tier-specialized multi-head attention pooling across H positions:
        h_y  = BiLSTM(x_y)              [Injects spatial continuity & tier order]
        c_sum = sum_{y=0}^{H-1} h_y       [Shift-robust recurrent sum path]
        c_att = MultiHeadAttn(h_y)      [Tier-specialized attention pooling]
        out  = LayerNorm(c_sum + c_att) [Residual fusion]

    Input : (B, C, H, W)
    Output: (B, W, D)
    """
    def __init__(self, inp_c: int, out_d: int, num_heads: int = 4):
        super().__init__()
        lstm_hidden = out_d // 2
        self.bilstm = nn.LSTM(inp_c, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)

        if out_d % num_heads != 0:
            raise ValueError(f"HybridCollapseY: out_d ({out_d}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = out_d // num_heads
        self.out_d = out_d
        self.scale = math.sqrt(self.head_dim)

        self.kv_proj = nn.Linear(out_d, 2 * out_d)
        self.out_proj = nn.Linear(out_d, out_d)
        self.collapse_q = nn.Parameter(torch.empty(1, 1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.collapse_q.view(1, -1, self.head_dim))
        self.norm = nn.LayerNorm(out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)

            # Step 1: Vertical Recurrent Context
            h_y, _ = self.bilstm(x_col)                      # (BW, H, D)
            c_sum = h_y.sum(dim=1)                           # (BW, D)

            # Step 2: Tier-Specialized Multi-Head Attention Pooling over h_y
            BW = B * W
            kv = self.kv_proj(h_y)                           # (BW, H, 2*D)
            k, v = kv.split(self.out_d, dim=-1)
            k = k.view(BW, H, self.num_heads, self.head_dim)
            v = v.view(BW, H, self.num_heads, self.head_dim)
            cq = self.collapse_q.expand(BW, -1, -1, -1)

            attn = (cq * k).sum(dim=-1) / self.scale
            alpha = torch.softmax(attn, dim=1).unsqueeze(-1)
            c_att = (alpha * v).sum(dim=1).view(BW, self.out_d)
            c_att = self.out_proj(c_att)                     # (BW, D)

            # Step 3: Residual Fusion
            out = self.norm(c_sum + c_att)                   # (BW, D)
            return out.view(B, W, -1)                        # (B, W, D)
        return x



class HorizontalSelfAttention(nn.Module):
    """SelfAttnX<heads>: Multi-Head Self-Attention over the horizontal sequence (W dimension).

    Applies multi-head self-attention across the width steps of a 1D sequence (B, W, C),
    allowing global context modeling across the entire text line.
    """
    uses_lengths = True

    def __init__(self, inp_c: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        gcd_heads = math.gcd(inp_c, num_heads)
        self.mha = nn.MultiheadAttention(embed_dim=inp_c, num_heads=gcd_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(inp_c)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        if x.dim() == 3:
            pad = None if lengths is None else ~_seq_mask(lengths, x.size(1), x.device)
            attn_out, _ = self.mha(x, x, x, key_padding_mask=pad)
            return self.norm(x + attn_out)
        return x


class MonotonicConformerBlock(nn.Module):
    """ConfX<heads>: Monotonic Conformer (ConvViT) block for 1D sequence (B, W, C).

    Combines Multi-Head Self-Attention (global context) with a Depthwise 1D Convolution
    (local positional monotonicity) and Feed-Forward Network:
        1. Local Monotonic Conv Module (Depthwise 1D conv k=7)
        2. Multi-Head Self-Attention (Global context)
        3. FFN + Residual LayerNorms
    """
    def __init__(self, inp_c: int, num_heads: int = 4, conv_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        gcd_heads = math.gcd(inp_c, num_heads)
        padding = conv_kernel // 2
        self.conv_norm = nn.LayerNorm(inp_c)
        self.conv_1d = nn.Sequential(
            nn.Conv1d(inp_c, inp_c * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(inp_c, inp_c, kernel_size=conv_kernel, padding=padding, groups=inp_c),
            nn.BatchNorm1d(inp_c),
            nn.SiLU(),
            nn.Conv1d(inp_c, inp_c, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(inp_c)
        self.mha = nn.MultiheadAttention(embed_dim=inp_c, num_heads=gcd_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(inp_c),
            nn.Linear(inp_c, inp_c * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inp_c * 2, inp_c),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(inp_c)

    uses_lengths = True

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        if x.dim() == 3:
            keep = None if lengths is None else _seq_mask(lengths, x.size(1), x.device)
            h = self.conv_norm(x).permute(0, 2, 1)
            for i, layer in enumerate(self.conv_1d):
                if i == 2 and keep is not None:          # depthwise conv: zeros past the line end
                    h = h * keep.unsqueeze(1)
                h = layer(h)
            x = x + h.permute(0, 2, 1)
            x_norm2 = self.attn_norm(x)
            attn_out, _ = self.mha(x_norm2, x_norm2, x_norm2,
                                   key_padding_mask=None if keep is None else ~keep)
            x = x + attn_out
            x = x + self.ffn(x)
            return self.out_norm(x)
        return x



class Permute1DForBN(nn.Module):
    """Wrap BatchNorm1d so it works on (B, T, C) sequence tensors.
    BatchNorm1d expects (B, C, T), so we permute before and after."""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _conv_like_width(w, kernel, stride, padding, dilation=1):
    """Width after a Conv2d / MaxPool2d (floor mode), given the input width tensor w."""
    kx, sx, px, dx = _pair(kernel)[1], _pair(stride)[1], _pair(padding)[1], _pair(dilation)[1]
    return torch.div(w + 2 * px - dx * (kx - 1) - 1, sx, rounding_mode='floor') + 1


def module_out_width(module, w):
    """Width (sequence length) that `module` produces from input width(s) w (LongTensor).

    Handles Conv2d, MaxPool2d and any module that defines ``out_width(w)`` (the Lcn4 block).
    Containers are walked in order; every other layer keeps the width unchanged.
    """
    if hasattr(module, 'out_width'):
        return module.out_width(w)
    if isinstance(module, nn.Conv2d):
        return _conv_like_width(w, module.kernel_size, module.stride, module.padding, module.dilation)
    if isinstance(module, nn.MaxPool2d):
        if module.ceil_mode:
            raise NotImplementedError('ceil_mode pooling is not used by the VGSL parser')
        return _conv_like_width(w, module.kernel_size, module.stride or module.kernel_size,
                                module.padding, module.dilation)
    if isinstance(module, nn.Sequential):
        for child in module:
            w = module_out_width(child, w)
    return w


def output_lengths(model, input_widths, max_len=None):
    """Real number of output frames per line, for CTC and decoding.

    input_widths: the unpadded image widths of the batch (the ``ilens`` from collate_fn).
    Frames that only see padding are excluded, so a line's loss and its decoded text do
    not depend on how wide the other lines in its batch are. Clamped to [1, max_len].
    """
    w = torch.as_tensor(input_widths, dtype=torch.long).cpu()
    w = module_out_width(model, w)
    w = w.clamp(min=1)
    if max_len is not None:
        w = w.clamp(max=max_len)
    return w


def _mask_width(x, w, fill):
    """Set the columns of a (B, C, H, W) map beyond each line's width w to `fill`."""
    keep = (torch.arange(x.size(3), device=x.device).unsqueeze(0) < w.to(x.device).unsqueeze(1))
    return x.masked_fill(~keep[:, None, None, :], fill)


class VGSLNet(nn.Sequential):
    """nn.Sequential that can also take the real width of every line in the batch.

    ``model(x)`` behaves exactly like nn.Sequential (old behaviour, ONNX export, single images).
    ``model(x, widths)`` makes each line's output independent of the padding (P0.1):
    convolutions and pools see the same zeros / -inf past the line end as at batch size 1,
    right-to-left and bidirectional LSTMs start at the line's own end, attention ignores
    padded frames. State-dict keys are the same as a plain nn.Sequential.
    """

    def forward(self, x, widths=None):
        if widths is None:
            return super().forward(x)
        w = torch.as_tensor(widths, dtype=torch.long).cpu()
        for layer in self:
            if x.dim() == 4 and w.numel():
                if isinstance(layer, nn.Conv2d) and _pair(layer.padding)[1] > 0:
                    x = _mask_width(x, w, 0.0)
                elif isinstance(layer, nn.MaxPool2d) and _pair(layer.padding)[1] > 0:
                    x = _mask_width(x, w, float('-inf'))
            if getattr(layer, 'uses_lengths', False):
                T = x.size(3) if x.dim() == 4 else x.size(1)
                x = layer(x, lengths=w.clamp(1, T))
            else:
                x = layer(x)
            w = module_out_width(layer, w)
        return x


def build_model(spec, num_classes):
    """Parse VGSL NET_SPEC → VGSLNet (an nn.Sequential; see VGSLNet for the widths argument)."""
    spec = spec.strip().strip('[]')
    tokens = spec.split()
    layers, in_c, cur_h, is_seq, rnn_in = [], 1, 48, False, 0

    for tok in tokens:
        if re.match(r'^\d+,\d+,\d+,\d+$', tok):
            parts = tok.split(',')
            in_c, cur_h = int(parts[3]), int(parts[1])
            continue
        m = re.match(r'^C([tsr]?)(\d+),(\d+),(\d+)$', tok)
        if m:
            md, ky, kx, oc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            layers += [nn.Conv2d(in_c, oc, (ky, kx), padding=(ky//2, kx//2)),
                       nn.Tanh() if md == 't' else nn.ReLU(inplace=True)]
            in_c = oc
            continue
        # ── Conv2d with explicit padding: Cp<act><ky>,<kx>,<oc>,<py>,<px> ──
        m = re.match(r'^Cp([tsr]?)(\d+),(\d+),(\d+),(\d+),(\d+)$', tok)
        if m:
            md = m.group(1)
            ky, kx, oc = int(m.group(2)), int(m.group(3)), int(m.group(4))
            py, px = int(m.group(5)), int(m.group(6))
            layers += [nn.Conv2d(in_c, oc, (ky, kx), padding=(py, px)),
                       nn.Tanh() if md == 't' else nn.ReLU(inplace=True)]
            in_c = oc
            cur_h = (cur_h + 2 * py - ky) // 1 + 1
            continue
        m = re.match(r'^Mp(\d+),(\d+)$', tok)
        if m:
            py, px = int(m.group(1)), int(m.group(2))
            layers.append(nn.MaxPool2d((py, px), stride=(py, px)))
            cur_h //= py
            continue
        # ── Asymmetric MaxPool: Mpa<ky>,<kx>,<sy>,<sx>,<py>,<px> ──
        m = re.match(r'^Mpa(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)$', tok)
        if m:
            ky, kx = int(m.group(1)), int(m.group(2))
            sy, sx = int(m.group(3)), int(m.group(4))
            py, px = int(m.group(5)), int(m.group(6))
            layers.append(nn.MaxPool2d((ky, kx), stride=(sy, sx), padding=(py, px)))
            cur_h = (cur_h + 2 * py - ky) // sy + 1
            continue
        # ── Additive Attention Y-collapse: Atys<D> ──
        m = re.match(r'^Atys(\d+)$', tok)
        if m:
            hid = int(m.group(1))
            layers.append(AttentionCollapseY(in_c, hid))
            is_seq, rnn_in, in_c = True, hid, hid
            continue
        # ── Multi-Head Attention Y-collapse: Matys<D> ──
        m = re.match(r'^Matys(\d+)$', tok)
        if m:
            hid = int(m.group(1))
            layers.append(MultiHeadAttentionCollapseY(in_c, hid))
            is_seq, rnn_in, in_c = True, hid, hid
            continue
        # ── Hybrid Recurrent + Multi-Head Attention Y-collapse: Hybys<D> ──
        m = re.match(r'^Hybys(\d+)$', tok)
        if m:
            hid = int(m.group(1))
            layers.append(HybridCollapseY(in_c, hid))
            is_seq, rnn_in, in_c = True, hid, hid
            continue
        # ── Horizontal Multi-Head Self-Attention: SelfAttnX<heads> ──
        m = re.match(r'^SelfAttnX(\d+)$', tok)
        if m:
            num_heads = int(m.group(1))
            if not is_seq:
                layers.append(CollapseH())
                rnn_in, is_seq = in_c * cur_h, True
            inp = rnn_in or in_c
            layers.append(HorizontalSelfAttention(inp, num_heads))
            rnn_in = inp
            in_c = inp
            continue
        # ── Monotonic Conformer (ConvViT) block: ConfX<heads> ──
        m = re.match(r'^ConfX(\d+)$', tok)
        if m:
            num_heads = int(m.group(1))
            if not is_seq:
                layers.append(CollapseH())
                rnn_in, is_seq = in_c * cur_h, True
            inp = rnn_in or in_c
            layers.append(MonotonicConformerBlock(inp, num_heads))
            rnn_in = inp
            in_c = inp
            continue
        # ── BiLSTM with Linear projection: Lbpx<hid>,<out> ──
        m = re.match(r'^Lbpx(\d+),(\d+)$', tok)
        if m:
            hid, out_d = int(m.group(1)), int(m.group(2))
            if out_d == num_classes:
                pass  # allow ### replacement
            if not is_seq:
                layers.append(CollapseH())
                rnn_in, is_seq = in_c * cur_h, True
            inp = rnn_in or in_c
            layers.append(BiLSTMProjection(inp, hid, out_d))
            rnn_in = out_d; in_c = out_d
            continue
        m = re.match(r'^L([frbm])([xy])([s]?)(\d+)$', tok)
        if m:
            d, _ax, sm, hid = m.group(1), m.group(2), m.group(3), int(m.group(4))
            if sm == 's':
                layers.append(SumLSTM(in_c, hid))
                is_seq, rnn_in, in_c = True, hid, hid
                continue
            if not is_seq:
                layers.append(CollapseH())
                rnn_in, is_seq = in_c * cur_h, True
            inp = rnn_in or in_c
            if d == 'b':
                layers.append(BiLSTM(inp, hid)); rnn_in = hid * 2; in_c = hid * 2
            elif d == 'r':
                layers.append(RevLSTM(inp, hid)); rnn_in = hid; in_c = hid
            else:
                layers.append(FwdLSTM(inp, hid)); rnn_in = hid; in_c = hid
            continue
        # ── Batch Normalization: Bn ──
        if tok == 'Bn':
            if is_seq:
                layers.append(Permute1DForBN(rnn_in or in_c))
            else:
                layers.append(nn.BatchNorm2d(in_c))
            continue
        # ── Dropout: D<rate>  (rate is an integer 0–99, e.g. D50 = 50%) ──
        m = re.match(r'^D(\d+)$', tok)
        if m:
            p = int(m.group(1)) / 100.0
            if is_seq:
                layers.append(nn.Dropout(p))
            else:
                layers.append(nn.Dropout2d(p))
            continue
        # ── Dense / Fully-connected: Fc<units>  (Linear + ReLU) ──
        m = re.match(r'^Fc(\d+)$', tok)
        if m:
            units = int(m.group(1))
            inp = rnn_in or in_c
            layers += [nn.Linear(inp, units), nn.ReLU(inplace=True)]
            rnn_in = units
            in_c = units
            continue
        # ── Output: O<dim>c<classes> ──
        m = re.match(r'^O(\d+)c(.+)$', tok)
        if m:
            n_out = num_classes if m.group(2) == '###' else int(m.group(2))
            if not is_seq:
                layers.append(CollapseH()); rnn_in = in_c * cur_h; is_seq = True
            layers.append(nn.Linear(rnn_in or in_c, n_out))
            continue
        print(f"  Warning: Unknown token '{tok}'")

    return VGSLNet(*layers)
