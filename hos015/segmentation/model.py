
"""
SceneSense - MaskFormer Model Integration

Model:
    facebook/maskformer-resnet50-coco-stuff

Task:
    Semantic image segmentation

Features:
    - 171 predefined semantic classes
    - CPU inference
    - Cached pretrained model
    - Pixel-level class predictions
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

from .config import (
    DEVICE,
    MODEL_ID,
    NUM_CLASSES,
)


# ==========================================================
# LOAD THE PRETRAINED MODEL
# ==========================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the pretrained MaskFormer model once.

    The cached model is reused for later predictions.
    """

    # Limit CPU threads to reduce unnecessary overhead.
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    print("\n" + "=" * 55)
    print("LOADING SCENESENSE MASKFORMER")
    print("=" * 55)

    print("Model:", MODEL_ID)
    print("Device:", DEVICE)

    # Load the model's image processor.
    print("\nLoading image processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False
    )

    # Load the pretrained model.
    print("Loading pretrained MaskFormer...")

    model = (
        MaskFormerForInstanceSegmentation
        .from_pretrained(MODEL_ID)
    )

    # Use the local CPU.
    model.to(torch.device(DEVICE))

    # Disable training behavior.
    model.eval()

    # Verify the expected class vocabulary.
    actual_classes = int(model.config.num_labels)

    if actual_classes != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"but the model contains {actual_classes}."
        )

    print("\nMaskFormer loaded successfully!")
    print("Classes:", actual_classes)
    print("Device:", DEVICE)

    return processor, model


# ==========================================================
# PREDICT SEMANTIC SEGMENTATION
# ==========================================================

def predict_label_map(
    image: Image.Image,
    processor,
    model,
) -> np.ndarray:
    """
    Predict a semantic class for every pixel.

    Returns:
        A 2D NumPy array with shape (height, width).
    """

    # Validate the input.
    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL image."
        )

    if image.width < 1 or image.height < 1:
        raise ValueError(
            "The image must have positive dimensions."
        )

    # Ensure the image uses RGB.
    image = image.convert("RGB")

    # Prepare the input tensor.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Run inference without calculating gradients.
    with torch.inference_mode():

        outputs = model(**inputs)

        # Convert MaskFormer predictions into
        # semantic class IDs at image resolution.
        segmentation = (
            processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[
                    (image.height, image.width)
                ]
            )[0]
        )

    # Convert the predicted mask to NumPy.
    labels = (
        segmentation
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Verify the output shape.
    expected_shape = (
        image.height,
        image.width
    )

    if labels.shape != expected_shape:
        raise RuntimeError(
            "Mask dimensions do not match "
            "the processed image."
        )

    # Verify the class IDs.
    if (
        labels.size == 0
        or int(labels.max()) >= NUM_CLASSES
    ):
        raise RuntimeError(
            "The model returned invalid semantic class IDs."
        )

    return labels


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def get_model_information():
    """Return information about the selected model."""

    return {
        "name": "MaskFormer ResNet-50 COCO-Stuff",
        "checkpoint": MODEL_ID,
        "classes": NUM_CLASSES,
        "device": DEVICE,
        "task": "semantic segmentation",
    }
