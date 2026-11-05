"""P0.1: a line's prediction must not depend on the other lines in its batch."""
import pytest
import torch
import torch.nn.functional as F

from tesstrain_core.dataset import PAD_BLACK, PAD_WHITE, collate_fn, legacy_collate_fn
from tesstrain_core.metrics import ctc_greedy_decode
from tesstrain_core.model import build_model, output_lengths

H = 32
# Each has a layer that reads the padding if nothing stops it: right-to-left LSTM (B0 layout),
# true BiLSTM + attention + conformer, an x-stride-1 pool with attention collapse, Lcn4 + SE.
SPECS = [
    '[1,32,0,1 Ct3,3,8 Mp3,3 Lfys16 Lfx32 Lrx32 Lfx48 O1c20]',
    '[1,32,0,1 Ct3,3,8 Mp3,3 Lfys16 Lbx32 SelfAttnX4 ConfX4 O1c20]',
    '[1,32,0,1 Ct5,5,8 Mpa3,3,3,1,0,1 Atys16 Lbpx16,16 Lrx16 O1c20]',
    '[1,32,0,1 Lcn4s,8,1 Mp3,3 Lcn4s,16,1,2,1 Lfys16 Lrx32 O1c20]',     # Lcn4 with SE (mean over the line)
]


def synthetic_lines(n=64, seed=0):
    """White lines (+1) with black 'glyph' blocks (-1) and a little noise, widths 40–300 px."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n):
        w = int(torch.randint(40, 300, (1,), generator=g))
        img = torch.ones(1, H, w)
        x = 4
        while x < w - 8:
            gw = int(torch.randint(3, 9, (1,), generator=g))
            top = int(torch.randint(4, 12, (1,), generator=g))
            bot = int(torch.randint(20, 29, (1,), generator=g))
            img[:, top:bot, x:x + gw] = -1.0
            x += gw + int(torch.randint(2, 10, (1,), generator=g))
        img += 0.05 * torch.randn(img.shape, generator=g)
        samples.append((img.clamp(-1, 1), torch.tensor([1]), w, 1))
    return samples


def seeded_model(spec):
    """Seeded random model. Weights are scaled ×4 so the output changes along the line
    (at the default init every frame predicts the same class and any test would pass)."""
    torch.manual_seed(0)
    model = build_model(spec, 20).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4)
    return model


def predict(model, samples, batch_size, collate, real_lengths=True):
    """Per-frame argmax on each line's own frames, and the decoded labels, for every sample."""
    frames, texts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            imgs, _, ilens, _ = collate(samples[i:i + batch_size])
            out = model(imgs, ilens) if real_lengths else model(imgs)
            T = out.shape[1]
            own = output_lengths(model, ilens, T)
            olens = own if real_lengths else torch.full((len(ilens),), T)
            for o, k, n in zip(out, own, olens):
                frames.append(o[:k].argmax(-1))
                texts.append(tuple(ctc_greedy_decode(o[:n])))
    return frames, texts


def agreement(a, b):
    same = sum(int((x == y).sum()) for x, y in zip(a, b))
    return same / sum(len(x) for x in a)


def test_output_lengths_follow_the_pooling():
    w = torch.tensor([40, 41, 299, 300])
    assert output_lengths(build_model(SPECS[0], 20), w).tolist() == [13, 13, 99, 100]   # Mp3,3: w // 3
    assert output_lengths(build_model(SPECS[2], 20), w).tolist() == w.tolist()          # x-stride 1
    model = build_model('[1,32,0,1 Ct3,3,8 Mp2,2 Ct3,3,8 Mp2,2 Lfys16 O1c20]', 20)
    assert output_lengths(model, torch.tensor([7, 8, 9])).tolist() == [1, 2, 2]
    assert output_lengths(model, torch.tensor([1000]), max_len=50).tolist() == [50]


def test_padding_is_white_by_default():
    a, b = torch.zeros(1, H, 5), torch.zeros(1, H, 9)
    batch = [(a, torch.tensor([1]), 5, 1), (b, torch.tensor([1]), 9, 1)]
    imgs, _, ilens, _ = collate_fn(batch)
    assert imgs.shape == (2, 1, H, 9) and ilens.tolist() == [5, 9]
    assert torch.all(imgs[0, :, :, 5:] == PAD_WHITE)
    imgs, *_ = legacy_collate_fn(batch)
    assert torch.all(imgs[0, :, :, 5:] == PAD_BLACK)


@pytest.mark.parametrize('spec', SPECS)
def test_batch_size_1_and_64_agree(spec):
    model, samples = seeded_model(spec), synthetic_lines()
    f1, t1 = predict(model, samples, 1, collate_fn)
    f64, t64 = predict(model, samples, 64, collate_fn)
    assert len({int(c) for f in f1 for c in f}) > 3, 'model output is constant; the test would be vacuous'
    assert agreement(f1, f64) >= 0.99
    same_lines = sum(a == b for a, b in zip(t1, t64)) / len(t1)
    assert same_lines >= 0.99, f'only {same_lines:.1%} of lines decode the same'


@pytest.mark.parametrize('spec', SPECS)
def test_legacy_black_padding_does_leak(spec):
    """The old behaviour, kept for --padding legacy. Shows the test above can see a leak."""
    model, samples = seeded_model(spec), synthetic_lines()
    _, t1 = predict(model, samples, 1, legacy_collate_fn, real_lengths=False)
    _, t64 = predict(model, samples, 64, legacy_collate_fn, real_lengths=False)
    assert sum(a == b for a, b in zip(t1, t64)) / len(t1) < 0.9


def test_model_without_widths_is_unchanged():
    """model(x) is plain nn.Sequential (ONNX export, single images); widths only matter for padding."""
    model = seeded_model(SPECS[0])
    x = synthetic_lines(1)[0][0].unsqueeze(0)
    with torch.no_grad():
        plain = torch.nn.Sequential(*model)(x)
        assert torch.allclose(model(x), plain)
        assert torch.allclose(model(x, torch.tensor([x.shape[3]])), plain, atol=1e-5)


def test_ctc_loss_ignores_padding_frames():
    """With real lengths, the loss of a line is the same alone or next to a much wider line."""
    model = seeded_model(SPECS[0])
    short, wide = sorted(synthetic_lines(2, seed=3), key=lambda s: s[2])
    wide = (F.pad(wide[0], (0, 200), value=PAD_WHITE), wide[1], wide[2] + 200, 1)
    ctc = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

    def loss_of_first(batch):
        imgs, labels, ilens, tlens = collate_fn(batch)
        lp = F.log_softmax(model(imgs, ilens), 2).permute(1, 0, 2)
        return ctc(lp, labels, output_lengths(model, ilens, lp.shape[0]), tlens)[0]

    with torch.no_grad():
        alone, together = loss_of_first([short]), loss_of_first([short, wide])
    assert torch.isclose(alone, together, rtol=1e-4)
