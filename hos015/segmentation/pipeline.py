
"""
SceneSense - Semantic Segmentation Pipeline

Processes uploaded images using MaskFormer
and prepares the results for the Gradio interface.
"""

from __future__ import annotations

import time

from PIL import Image, ImageOps

from .analytics import summarize_mask

from .config import (
    DEFAULT_OVERLAY_OPACITY,
    MAX_IMAGE_SIDE,
    MAX_INPUT_PIXELS,
    MIN_IMAGE_SIDE,
)

from .model import (
    load_model,
    predict_label_map,
)

from .visualize import (
    build_palette,
    create_color_mask,
    create_overlay,
)


def prepare_image(image: Image.Image) -> Image.Image:
    """Validate, convert, and resize an uploaded image."""

    if image is None:
        raise ValueError("Please upload an image.")

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    width, height = image.size

    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValueError(
            f"Image must be at least {MIN_IMAGE_SIDE} pixels per side."
        )

    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            "This image is too large. Please use a smaller image."
        )

    prepared = ImageOps.exif_transpose(image).convert("RGB")

    prepared.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS
    )

    return prepared


def run_segmentation(
    image: Image.Image,
    opacity: float = DEFAULT_OVERLAY_OPACITY,
):
    """Run semantic segmentation and return the results."""

    if not 0 <= float(opacity) <= 1:
        raise ValueError(
            "Overlay strength must be between 0 and 1."
        )

    # Prepare the image.
    prepared = prepare_image(image)

    # Load the pretrained MaskFormer model.
    processor, model = load_model()

    # Perform semantic segmentation.
    start = time.perf_counter()

    label_map = predict_label_map(
        prepared,
        processor,
        model
    )

    elapsed = time.perf_counter() - start

    # Get class names.
    class_names = {
        int(class_id): str(name).strip()
        for class_id, name in model.config.id2label.items()
    }

    # Create the color palette.
    palette = build_palette(
        class_names,
        int(model.config.num_labels)
    )

    # Generate the semantic mask.
    color_mask = create_color_mask(
        label_map,
        palette
    )

    # Generate the overlay.
    overlay = create_overlay(
        prepared,
        color_mask,
        float(opacity)
    )

    # Calculate category statistics.
    rows = summarize_mask(
        label_map,
        class_names
    )

    # Display each result on a separate line.
    summary = (
        f"**Image:** {prepared.width} × {prepared.height}  \n"
        f"**Detected:** {len(rows)}  \n"
        f"**Inference:** {elapsed:.2f}s"
    )

    return (
        color_mask,
        overlay,
        rows,
        summary
    )
