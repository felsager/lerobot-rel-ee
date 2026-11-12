from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class _FakePolicy(nn.Module):
    """Stand-in exposing only what `_preprocess_images` reads, so the test needs no weights."""

    def __init__(self, dropout: float):
        super().__init__()
        keys = ("observation.images.top", "observation.images.wrist")
        self.config = SimpleNamespace(
            image_features=keys, image_resolution=(224, 224), wrist_camera_dropout=dropout
        )
        self.anchor = nn.Parameter(torch.zeros(1))

    _preprocess_images = PI05Policy._preprocess_images


def _masks(dropout: float, training: bool):
    policy = _FakePolicy(dropout).train(training)
    batch = {
        key: torch.rand(4000, 3, 224, 224)[:1].expand(4000, -1, -1, -1)
        for key in policy.config.image_features
    }
    torch.manual_seed(0)
    return policy._preprocess_images(batch)[1]


def test_default_is_off():
    assert PI05Config.__dataclass_fields__["wrist_camera_dropout"].default == 0.0


def test_training_masks_only_the_wrist_camera():
    top, wrist = _masks(0.1, training=True)
    assert top.all()
    assert 0.07 < 1 - wrist.float().mean().item() < 0.13


def test_inference_keeps_every_camera():
    top, wrist = _masks(0.1, training=False)
    assert top.all() and wrist.all()
