import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.helpers import is_valid_image
from utils.preprocessing import (
    letterbox_resize,
    flip_image,
    adjust_brightness,
    scale_image,
    crop_image,
    apply_all_augmentations,
)


def make_image(w=200, h=100):
    return Image.fromarray(np.full((h, w, 3), 120, dtype=np.uint8), mode="RGB")


def test_letterbox_resize():
    out = letterbox_resize(make_image(), 416)
    assert out.size == (416, 416)
    assert out.mode == "RGB"


def test_flip_and_brightness():
    img = make_image()
    assert flip_image(img).size == img.size
    bright = adjust_brightness(img, factor=1.2)
    assert bright.size == img.size


def test_scale_and_crop():
    img = letterbox_resize(make_image(), 416)
    assert scale_image(img, factor=1.1, target_size=416).size == (416, 416)
    assert crop_image(img, crop_fraction=0.1, target_size=416).size == (416, 416)


def test_apply_all_augmentations():
    config = {"augmentation": {"horizontal_flip": True, "brightness_range": [0.9, 1.1], "scale_range": [1.0, 1.0], "crop_fraction": 0.1}}
    out = apply_all_augmentations(make_image(), config, target_size=416)
    assert isinstance(out, dict)
    assert out
    assert all(isinstance(v, Image.Image) for v in out.values())


def test_invalid_image():
    valid, message = is_valid_image("/definitely/not/a/real/file.jpg")
    assert valid is False
    assert message
