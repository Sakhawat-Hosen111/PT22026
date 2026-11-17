
"""Consistent, muted colours for semantic segmentation masks."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping

import numpy as np
from PIL import Image


# Fixed colours used for prominent ADE20K classes (RGB).
SPECIAL_COLORS = {
    "person": (230, 105, 99),
    "animal": (234, 166, 83),
    "tree": (69, 151, 114),
    "grass": (156, 188, 104),
    "plant": (99, 173, 137),
    "sky": (102, 162, 215),
    "water": (75, 169, 181),
    "sea": (64, 125, 176),
    "road": (126, 137, 151),
    "sidewalk": (186, 174, 151),
    "wall": (173, 150, 183),
    "floor": (193, 178, 151),
    "ceiling": (216, 190, 153),
    "building": (139, 151, 175),
    "windowpane": (130, 189, 199),
    "door": (192, 139, 107),
    "chair": (183, 121, 104),
    "sofa": (135, 158, 191),
    "table": (186, 153, 111),
    "cabinet": (152, 126, 105),
    "earth": (167, 143, 116),
    "mountain": (122, 145, 124),
    "rock": (133, 134, 147),
    "fence": (180, 141, 175),
}

ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "deer", "rabbit", "monkey",
    "fish", "animal",
}


def build_palette(
    class_names: Mapping[int | str, str] | None = None,
    num_classes: int = 150,
) -> np.ndarray:
    """Assign stable colours to classes independent of the input image."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    names = class_names or {}

    palette = np.empty(
        (num_classes, 3),
        dtype=np.uint8
    )

    for class_id in range(num_classes):

        # Stable fallback colour, gently saturated.
        hue = (class_id * 0.61803398875) % 1.0

        rgb = colorsys.hsv_to_rgb(
            hue,
            0.43,
            0.82
        )

        color = tuple(
            int(round(value * 255))
            for value in rgb
        )

        label = str(
            names.get(
                class_id,
                names.get(str(class_id), "")
            )
        ).strip().lower()

        if label in ANIMALS:
            color = SPECIAL_COLORS["animal"]

        elif label in SPECIAL_COLORS:
            color = SPECIAL_COLORS[label]

        palette[class_id] = color

    return palette


def create_color_mask(
    label_map: np.ndarray,
    palette: np.ndarray,
) -> Image.Image:
    """Convert a 2D class-ID array to an RGB segmentation image."""

    labels = np.asarray(label_map)
    colors = np.asarray(palette)

    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(
            "Label map must be a non-empty 2D array."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Label map must contain integer IDs."
        )

    if (
        colors.ndim != 2
        or colors.shape[1] != 3
        or len(colors) == 0
    ):
        raise ValueError(
            "Palette must have shape (N, 3)."
        )

    if (
        np.any(labels < 0)
        or np.any(labels >= len(colors))
    ):
        raise ValueError(
            "Label map contains an invalid class ID."
        )

    rgb = colors[
        labels.astype(np.int64)
    ].astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def create_overlay(
    image: Image.Image,
    color_mask: Image.Image,
    opacity: float = 0.35,
) -> Image.Image:
    """Blend the segmentation colours on top of the original image."""

    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(
            "Opacity must be between 0 and 1."
        )

    original = image.convert("RGB")
    mask = color_mask.convert("RGB")

    if original.size != mask.size:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    return Image.blend(
        original,
        mask,
        float(opacity)
    )
