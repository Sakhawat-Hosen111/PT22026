
"""
SceneSense - Semantic Segmentation Pipeline

This file processes uploaded images using MaskFormer
and prepares the results for the Gradio interface.
"""

from __future__ import annotations

# Used to measure how long the model takes to make a prediction.
import time

# Pillow handles images, while ImageOps helps correct image orientation.
from PIL import Image, ImageOps

# Calculate pixel counts and percentages for detected categories.
from .analytics import summarize_mask

# Import image size limits and the default overlay strength.
from .config import (
    DEFAULT_OVERLAY_OPACITY,
    MAX_IMAGE_SIDE,
    MAX_INPUT_PIXELS,
    MIN_IMAGE_SIDE,
)

# Load the pretrained model and predict pixel-level categories.
from .model import (
    load_model,
    predict_label_map,
)

# Create colours, segmentation masks and overlay images.
from .visualize import (
    build_palette,
    create_color_mask,
    create_overlay,
)


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image: Image.Image) -> Image.Image:
    """Validate, convert and resize an uploaded image."""

    if image is None:
        raise ValueError("Please upload an image.")

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    # Get the original image dimensions.
    width, height = image.size

    # Reject images that are too small for processing.
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValueError(
            f"Image must be at least {MIN_IMAGE_SIDE} pixels per side."
        )

    # Reject extremely large images to avoid excessive memory usage.
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            "This image is too large. Please use a smaller image."
        )

    # Correct the image orientation using its Exchangeable Image
    # File Format metadata, then convert it to Red, Green and Blue.
    prepared = ImageOps.exif_transpose(image).convert("RGB")

    # Resize large images while keeping their original aspect ratio.
    # LANCZOS is used to maintain image quality during resizing.
    prepared.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS
    )

    return prepared


# ============================================================
# SEGMENTATION PIPELINE
# ============================================================

def run_segmentation(
    image: Image.Image,
    opacity: float = DEFAULT_OVERLAY_OPACITY,
):
    """
    Run semantic segmentation and prepare the results
    for the SceneSense interface.
    """

    # Overlay strength must be between 0 and 1.
    if not 0 <= float(opacity) <= 1:
        raise ValueError(
            "Overlay strength must be between 0 and 1."
        )

    # Validate and prepare the uploaded photograph.
    prepared = prepare_image(image)

    # Load the cached MaskFormer model and its image processor.
    processor, model = load_model()

    # Measure only the model prediction and postprocessing time.
    start = time.perf_counter()

    # Predict a semantic class identifier for every pixel.
    label_map = predict_label_map(
        prepared,
        processor,
        model
    )

    elapsed = time.perf_counter() - start

    # Get the category names from the pretrained model configuration.
    class_names = {
        int(class_id): str(name).strip()
        for class_id, name in model.config.id2label.items()
    }

    # Assign consistent colours to the model's semantic categories.
    palette = build_palette(
        class_names,
        int(model.config.num_labels)
    )

    # Convert the predicted class identifiers into a coloured mask.
    color_mask = create_color_mask(
        label_map,
        palette
    )

    # Blend the segmentation mask with the original photograph.
    overlay = create_overlay(
        prepared,
        color_mask,
        float(opacity)
    )

    # Calculate detected category names, pixel counts and percentages.
    rows = summarize_mask(
        label_map,
        class_names
    )

    # Prepare the image dimensions, detected category count
    # and inference time for display in the Gradio interface.
    summary = (
        f"**Image:** {prepared.width} × {prepared.height}  \n"
        f"**Detected:** {len(rows)}  \n"
        f"**Inference:** {elapsed:.2f}s"
    )

    # Return the four results in the order expected by app.py.
    return (
        color_mask,
        overlay,
        rows,
        summary
    )
