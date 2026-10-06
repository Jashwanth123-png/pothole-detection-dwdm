"""
utils/preprocessing.py
=======================
Image preprocessing and augmentation utilities for the pothole
detection pipeline.

Provides:
- letterbox_resize : resize an image to a square while preserving aspect ratio
- flip_image       : horizontal flip augmentation
- adjust_brightness: brightness jitter augmentation
- scale_image      : random scale augmentation
- crop_image       : centre-crop augmentation

All functions accept and return PIL Image objects, making them easy to
chain together.

Beginner note:
  PIL (Pillow) is Python's standard image library.
  Every function here takes an image and returns a NEW image — the
  original is never modified.
"""

import random
from pathlib import Path

try:
    from PIL import Image, ImageEnhance
except ImportError as e:
    raise ImportError(
        "Pillow is required for image preprocessing.\n"
        "Install it with:  pip install pillow"
    ) from e


# ──────────────────────────────────────────────────────────────
# 1.  LETTERBOX RESIZE
# ──────────────────────────────────────────────────────────────

def letterbox_resize(image: Image.Image, target_size: int = 416) -> Image.Image:
    """
    Resize an image to a square canvas while preserving its aspect ratio.

    This technique is called "letterboxing":
    - The image is scaled down (or up) so its longest side equals target_size.
    - The remaining area is padded with grey (128, 128, 128) pixels.

    This is the standard preprocessing step used by YOLO models so that
    objects are not distorted by stretching.

    Parameters
    ----------
    image : PIL.Image.Image
        The source image to resize.
    target_size : int
        The desired square output size in pixels (default 416).

    Returns
    -------
    PIL.Image.Image
        A new RGB image of shape (target_size × target_size).

    Example
    -------
    >>> from PIL import Image
    >>> img = Image.open("road.jpg")
    >>> resized = letterbox_resize(img, 416)
    >>> resized.size   # (416, 416)
    """
    # Convert to RGB (handles RGBA, grayscale, palette images)
    image = image.convert("RGB")
    original_w, original_h = image.size

    # --- Compute scale factor ---
    scale = target_size / max(original_w, original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # --- Resize the image (high-quality Lanczos filter) ---
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # --- Create a grey square canvas ---
    canvas = Image.new("RGB", (target_size, target_size), color=(128, 128, 128))

    # --- Paste the resized image centred on the canvas ---
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


# ──────────────────────────────────────────────────────────────
# 2.  HORIZONTAL FLIP
# ──────────────────────────────────────────────────────────────

def flip_image(image: Image.Image) -> Image.Image:
    """
    Apply a horizontal (left-right) flip to an image.

    This is one of the most common and effective augmentations for
    object detection — it doubles the dataset with no quality loss.

    Note on YOLO labels after a horizontal flip:
      For a bounding box (x_center, y_center, width, height) in YOLO format:
      - The new x_center = 1.0 - x_center
      - y_center, width, height remain unchanged

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    PIL.Image.Image
        Horizontally flipped image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


# ──────────────────────────────────────────────────────────────
# 3.  BRIGHTNESS ADJUSTMENT
# ──────────────────────────────────────────────────────────────

def adjust_brightness(
    image: Image.Image,
    brightness_range: tuple = (0.7, 1.3),
    factor: float = None,
) -> Image.Image:
    """
    Randomly adjust the brightness of an image.

    Simulates different lighting conditions (overcast vs. sunny roads).
    A factor of 1.0 returns the original brightness; < 1.0 darkens;
    > 1.0 brightens.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    brightness_range : tuple[float, float]
        The (min, max) range from which a random factor is drawn.
        Defaults to (0.7, 1.3) as configured in config.yaml.
    factor : float, optional
        If provided, use this exact factor instead of a random one.
        Useful for deterministic testing.

    Returns
    -------
    PIL.Image.Image
        Brightness-adjusted image.
    """
    if factor is None:
        factor = random.uniform(brightness_range[0], brightness_range[1])

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


# ──────────────────────────────────────────────────────────────
# 4.  SCALE (ZOOM IN / OUT)
# ──────────────────────────────────────────────────────────────

def scale_image(
    image: Image.Image,
    scale_range: tuple = (0.8, 1.2),
    target_size: int = 416,
    factor: float = None,
) -> Image.Image:
    """
    Randomly scale (zoom) an image and resize back to target_size × target_size.

    - scale > 1.0 : zooms in (crops edges)
    - scale < 1.0 : zooms out (adds grey padding)

    Parameters
    ----------
    image : PIL.Image.Image
        Source image (should already be target_size × target_size).
    scale_range : tuple[float, float]
        The (min, max) range for the random scale factor.
    target_size : int
        Output size in pixels.
    factor : float, optional
        Fixed scale factor; overrides random selection.

    Returns
    -------
    PIL.Image.Image
        Scaled image of size (target_size × target_size).
    """
    if factor is None:
        factor = random.uniform(scale_range[0], scale_range[1])

    w, h = image.size
    new_w = int(w * factor)
    new_h = int(h * factor)

    # Scale the image
    scaled = image.resize((new_w, new_h), Image.LANCZOS)

    # Create a grey canvas and paste centred
    canvas = Image.new("RGB", (target_size, target_size), color=(128, 128, 128))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # When zoomed in (new size > target), we crop instead of paste
    if new_w > target_size or new_h > target_size:
        crop_x = max(0, (new_w - target_size) // 2)
        crop_y = max(0, (new_h - target_size) // 2)
        cropped = scaled.crop((
            crop_x, crop_y,
            crop_x + target_size, crop_y + target_size
        ))
        return cropped
    else:
        canvas.paste(scaled, (paste_x, paste_y))
        return canvas


# ──────────────────────────────────────────────────────────────
# 5.  CENTRE CROP
# ──────────────────────────────────────────────────────────────

def crop_image(
    image: Image.Image,
    crop_fraction: float = 0.1,
    target_size: int = 416,
) -> Image.Image:
    """
    Remove a fraction of pixels from each edge, then resize back to target_size.

    This simulates the camera capturing slightly different field-of-view
    angles and helps the model generalise.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    crop_fraction : float
        Fraction of the image to remove from each side.
        0.1 means 10% is cropped from each edge → 80% of image remains.
    target_size : int
        Output size in pixels.

    Returns
    -------
    PIL.Image.Image
        Cropped and resized image.
    """
    w, h = image.size
    crop_px_w = int(w * crop_fraction)
    crop_py_h = int(h * crop_fraction)

    # Crop box: (left, upper, right, lower)
    left   = crop_px_w
    upper  = crop_py_h
    right  = w - crop_px_w
    lower  = h - crop_py_h

    # Guard: crop box must be valid
    if right <= left or lower <= upper:
        return image.resize((target_size, target_size), Image.LANCZOS)

    cropped = image.crop((left, upper, right, lower))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


# ──────────────────────────────────────────────────────────────
# 6.  CONVENIENCE: APPLY ALL AUGMENTATIONS
# ──────────────────────────────────────────────────────────────

def apply_all_augmentations(
    image: Image.Image,
    config: dict,
    target_size: int = 416,
) -> dict:
    """
    Apply each augmentation and return a dictionary of augmented images.

    Parameters
    ----------
    image : PIL.Image.Image
        The source image (should already be letterbox-resized).
    config : dict
        Augmentation config block from config.yaml (config["augmentation"]).
    target_size : int
        Output size for scale/crop augmentations.

    Returns
    -------
    dict[str, PIL.Image.Image]
        Keys are augmentation names, values are resulting PIL images:
        {
          "flip"       : <flipped image>,
          "brightness" : <brightness-adjusted image>,
          "scale"      : <scaled image>,
          "crop"       : <cropped image>,
        }
        Only includes augmentations that are enabled in config.yaml.
    """
    aug_cfg = config.get("augmentation", {})
    results = {}

    # Horizontal flip
    if aug_cfg.get("horizontal_flip", True):
        results["flip"] = flip_image(image)

    # Brightness jitter
    br = aug_cfg.get("brightness_range", [0.7, 1.3])
    results["brightness"] = adjust_brightness(image, brightness_range=tuple(br))

    # Scale
    sr = aug_cfg.get("scale_range", [0.8, 1.2])
    results["scale"] = scale_image(
        image, scale_range=tuple(sr), target_size=target_size
    )

    # Crop
    cf = aug_cfg.get("crop_fraction", 0.1)
    results["crop"] = crop_image(image, crop_fraction=cf, target_size=target_size)

    return results

# ---------------------------------------------------------------------------
# Compatibility helpers for the ML inference modules
# ---------------------------------------------------------------------------

def load_image(image_path: str):
    """Load an image as a BGR NumPy array for OpenCV/YOLO inference."""
    import numpy as np
    try:
        import cv2
        image = cv2.imread(str(image_path))
        return image
    except Exception:
        try:
            image = Image.open(image_path).convert("RGB")
            return np.array(image)[:, :, ::-1].copy()
        except Exception:
            return None


def resize_image(image, target_size: int = 416):
    """Compatibility wrapper that resizes a PIL image or NumPy image to a square."""
    if isinstance(image, Image.Image):
        return letterbox_resize(image, target_size)
    import numpy as np
    arr = np.asarray(image)
    if arr.ndim == 3:
        pil = Image.fromarray(arr[:, :, ::-1] if arr.shape[2] == 3 else arr)
        out = letterbox_resize(pil, target_size)
        return np.array(out)[:, :, ::-1].copy()
    raise TypeError("resize_image expects a PIL image or a 3-channel NumPy array")
