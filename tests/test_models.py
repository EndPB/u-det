"""U-Net 单元测试（CPU 即可运行）。"""

from __future__ import annotations

import torch

from udet.models import UNet1D, UNet2D, build_unet, list_unets, register_unet


def test_registry_contents():
    assert "unet1d" in list_unets()
    assert "unet2d" in list_unets()


def test_build_unet_from_dict():
    unet = build_unet({"name": "unet1d", "in_channels": 16, "out_channels": 3, "features": [8, 16]})
    assert isinstance(unet, UNet1D)
    assert unet.out_channels == 3


def test_unet1d_shape_even_and_odd():
    unet = UNet1D(in_channels=32, out_channels=2, features=(8, 16, 32))
    unet.eval()
    for length in (64, 65, 63, 37):  # 覆盖奇偶长度
        x = torch.randn(2, 32, length)
        mask = torch.ones(2, length, dtype=torch.long)
        mask[0, length // 2 :] = 0
        with torch.no_grad():
            logits = unet(x, mask=mask)
        assert logits.shape == (2, 2, length), f"长度 {length} 输出形状错误: {logits.shape}"
        # mask 位置被置零
        assert torch.allclose(logits[0, :, length // 2 :], torch.zeros_like(logits[0, :, length // 2:]))


def test_unet1d_deep_supervision_and_features():
    unet = UNet1D(in_channels=8, out_channels=1, features=(4, 8, 16), deep_supervision=True)
    x = torch.randn(2, 8, 50)
    outs = unet(x)
    assert isinstance(outs, list) and len(outs) == 3
    assert all(o.shape == (2, 1, 50) for o in outs)

    unet2 = UNet1D(in_channels=8, out_channels=1, features=(4, 8))
    logits, feats = unet2(x, return_features=True)
    assert logits.shape == (2, 1, 50)
    assert len(feats) == 1


def test_unet1d_backward():
    unet = UNet1D(in_channels=16, out_channels=2, features=(8, 16))
    x = torch.randn(2, 16, 33, requires_grad=True)
    logits = unet(x)
    loss = logits.mean()
    loss.backward()
    grads = [p.grad for p in unet.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_unet2d_shapes():
    unet = UNet2D(in_channels=3, out_channels=5, features=(8, 16, 32))
    unet.eval()
    for h, w in ((64, 64), (65, 37), (100, 100)):
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = unet(x)
        assert y.shape == (1, 5, h, w)


def test_unet2d_bilinear_and_ds():
    unet = UNet2D(in_channels=1, out_channels=2, features=(4, 8), bilinear=True, deep_supervision=True)
    x = torch.randn(2, 1, 32, 24)
    outs = unet(x)
    assert isinstance(outs, list) and len(outs) == 2
    assert all(o.shape == (2, 2, 32, 24) for o in outs)


def test_register_custom_unet():
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return x

    register_unet("dummy_test")(Dummy)
    assert "dummy_test" in list_unets()
    assert isinstance(build_unet("dummy_test"), Dummy)
