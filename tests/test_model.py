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
