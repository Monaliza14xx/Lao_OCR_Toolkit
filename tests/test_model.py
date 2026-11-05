import pytest
import torch

from tesstrain_core.model import build_model


@pytest.mark.parametrize('spec', [
    '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lbx96 O1c43]',
    '[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx192 O1c43]',
])
def test_model_maps_a_line_image_to_per_column_class_scores(spec):
    model = build_model(spec, 43)
    out = model(torch.zeros(2, 1, int(spec.split(',')[1]), 120))
    if out.dim() == 4:                       # (B, C, H, W) → (B, W, C·H), as in training
        b, c, h, w = out.shape
        out = out.permute(0, 3, 1, 2).reshape(b, w, c * h)
    assert out.shape[0] == 2 and out.shape[-1] == 43
    assert out.shape[1] > 0


# ── Lcn4: LCNetV4-style CNN stage ────────────────────────────────────────────────────────
from tesstrain_core.model import Lcn4Stage, RepDWConv, output_lengths, reparameterize_model  # noqa: E402

LCN4_SPEC = '[1,64,0,1 Lcn4,48,1 Mp2,2 Lcn4,96,2 Lcn4,160,2,2,1 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]'


def _randomise_bn(model):
    """Give every BN non-trivial statistics, so fusing is really tested (not 0-mean / 1-var)."""
    g = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.3)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.randn(m.num_features, generator=g))
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)


def test_lcn4_spec_builds_runs_and_keeps_height():
    torch.manual_seed(0)
    model = build_model(LCN4_SPEC.replace('###', '80'), 80).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n  {LCN4_SPEC}: {n_params:,} params (80 classes)')   # visible with pytest -s
    assert 500_000 < n_params < 3_000_000

    seen = {}
    collapse = next(m for m in model if m.__class__.__name__ == 'SumLSTM')
    collapse.register_forward_hook(lambda m, inp, out: seen.update(shape=inp[0].shape))
    with torch.no_grad():
        out = model(torch.randn(2, 1, 64, 240))
    b, c, h, w = seen['shape']
    assert (b, c) == (2, 160) and h >= 8, f'only {h} rows reach the height collapse'
    assert h == 16 and w == 120                       # 64 → Mp2,2 → 32 → stride (2,1) → 16; W/2
    assert out.shape == (2, 120, 80)
    assert output_lengths(model, torch.tensor([240, 101])).tolist() == [120, 50]


def test_lcn4_output_stays_4d():
    stage = Lcn4Stage(1, 32, 2, stride=(2, 1), se=True).eval()
    out = stage(torch.randn(3, 1, 48, 77))
    assert out.shape == (3, 32, 24, 77)


@pytest.mark.parametrize('se', [False, True])
def test_reparameterized_lcn4_matches_unfused(se):
    torch.manual_seed(0)
    spec = f'[1,32,0,1 Lcn4{"s" if se else ""},16,2 Mp2,2 Lcn4,24,1,2,1 Lfys16 Lfx32 O1c20]'
    model = build_model(spec, 20)
    _randomise_bn(model)
    for m in model.modules():                    # undo the zero-init so the channel mixer counts too
        if isinstance(m, torch.nn.BatchNorm2d) and torch.all(m.weight == 0):
            m.weight.data.fill_(0.5)
    model.eval()
    x = torch.randn(2, 1, 32, 90)
    with torch.no_grad():
        before = model(x)
        before_w = model(x, torch.tensor([90, 61]))
        reparameterize_model(model)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for r in model.modules()
                       if isinstance(r, RepDWConv) for m in r.modules())
        assert all(r.fused is not None for r in model.modules() if isinstance(r, RepDWConv))
        after = model(x)
        after_w = model(x, torch.tensor([90, 61]))
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)
    assert torch.allclose(before_w, after_w, atol=1e-4, rtol=1e-4)
    n_unfused = sum(p.numel() for p in build_model(spec, 20).parameters())
    assert sum(p.numel() for p in model.parameters()) < n_unfused


def test_rep_dw_conv_fuses_to_one_depthwise_conv():
    conv = RepDWConv(8)
    _randomise_bn(conv)
    conv.eval()
    x = torch.randn(1, 8, 10, 10)
    with torch.no_grad():
        ref = conv(x)
        conv.reparameterize()
        assert conv.fused.groups == 8 and conv.fused.kernel_size == (3, 3)
        assert torch.allclose(conv(x), ref, atol=1e-5)
