
"""
SceneSense - Segmentation Visualization

This file assigns colours to semantic classes,
creates the segmentation mask and blends it
with the original image.
"""

from __future__ import annotations

# colorsys is used to generate different colours for semantic classes.
import colorsys
from collections.abc import Mapping

import numpy as np
from PIL import Image


# These colours are assigned to some common semantic categories
# so they always appear with the same colour in different images.
# Each tuple contains Red, Green and Blue values.
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


# Different animal classes use the same colour in the mask.
# Their actual category names remain separate in the results table.
ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "deer", "rabbit", "monkey",
    "fish", "animal",
}


# ============================================================
# CREATE COLOUR PALETTE
# ============================================================

def build_palette(
    class_names: Mapping[int | str, str] | None = None,
    num_classes: int = 150,
) -> np.ndarray:
    """Create consistent colours for the semantic classes."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    names = class_names or {}

    # Create an empty NumPy array to store three colour values
    # for each semantic class.
    palette = np.empty(
        (num_classes, 3),
        dtype=np.uint8
    )

    # Assign a colour to each class using its class identifier.
    for class_id in range(num_classes):

        # Generate a consistent colour using the class identifier.
        # The same class will receive the same colour each time.
        hue = (class_id * 0.61803398875) % 1.0

        # Convert Hue, Saturation and Value into Red, Green and Blue.
        # Lower saturation creates softer colours for the mask.
        rgb = colorsys.hsv_to_rgb(
            hue,
            0.43,
            0.82
        )

        # Convert colour values from the 0–1 range
        # into the 0–255 range used by image pixels.
        color = tuple(
            int(round(value * 255))
            for value in rgb
        )

        # Find the category name using its class identifier.
        # Convert it to lowercase for easier matching.
        label = str(
            names.get(
                class_id,
                names.get(str(class_id), "")
            )
        ).strip().lower()

        # Use a fixed colour for animal categories.
        if label in ANIMALS:
            color = SPECIAL_COLORS["animal"]

        # Use the predefined colour if the category is listed.
        elif label in SPECIAL_COLORS:
            color = SPECIAL_COLORS[label]

        # Store the final colour for this semantic class.
        palette[class_id] = color

    return palette


# ============================================================
# CREATE SEGMENTATION MASK
# ============================================================

def create_color_mask(
    label_map: np.ndarray,
    palette: np.ndarray,
) -> Image.Image:
    """Convert the predicted class identifiers into a coloured mask."""

    # Convert the class map and colour palette into NumPy arrays.
    labels = np.asarray(label_map)
    colors = np.asarray(palette)

    # The label map must be a non-empty two-dimensional array.
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(
            "Label map must be a non-empty 2D array."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Label map must contain integer IDs."
        )

    # Every class must have three colour values:
    # Red, Green and Blue.
    if (
        colors.ndim != 2
        or colors.shape[1] != 3
        or len(colors) == 0
    ):
        raise ValueError(
            "Palette must have shape (N, 3)."
        )

    # Make sure all predicted class identifiers
    # have corresponding colours in the palette.
    if (
        np.any(labels < 0)
        or np.any(labels >= len(colors))
    ):
        raise ValueError(
            "Label map contains an invalid class ID."
        )

    # Replace each numerical class identifier
    # with its corresponding Red, Green and Blue colour.
    rgb = colors[
        labels.astype(np.int64)
    ].astype(np.uint8)

    # Convert the coloured NumPy array into a Pillow image.
    return Image.fromarray(rgb, mode="RGB")


# ============================================================
# CREATE SEGMENTATION OVERLAY
# ============================================================

def create_overlay(
    image: Image.Image,
    color_mask: Image.Image,
    opacity: float = 0.35,
) -> Image.Image:
    """Blend the coloured segmentation mask with the original image."""

    # Opacity must be between 0 and 1.
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(
            "Opacity must be between 0 and 1."
        )

    # Make sure both images use Red, Green and Blue colour mode.
    original = image.convert("RGB")
    mask = color_mask.convert("RGB")

    # Both images must have the same dimensions
    # so their pixels can be blended correctly.
    if original.size != mask.size:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    # Blend the segmentation colours with the original photograph.
    # Opacity 0 shows the original image, while 1 shows only the mask.
    return Image.blend(
        original,
        mask,
        float(opacity)
    )
