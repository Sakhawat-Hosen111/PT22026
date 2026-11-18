
"""
SceneSense - MaskFormer Model Integration

This file loads the pretrained MaskFormer model
and uses it to predict semantic labels for images.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from PIL import Image

# Transformers provides the image processor and the MaskFormer
# model implementation used for semantic segmentation.
from transformers import (
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

# Import the model name, processing device and number of classes.
from .config import (
    DEVICE,
    MODEL_ID,
    NUM_CLASSES,
)


# ==========================================================
# LOAD THE PRETRAINED MODEL
# ==========================================================

# Cache the loaded model so it can be reused for other images
# instead of loading the pretrained weights every time.
@lru_cache(maxsize=1)
def load_model():
    """Load the pretrained MaskFormer model and image processor."""

    # Limit PyTorch to a maximum of four processing threads.
    # This helps control Central Processing Unit usage.
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    print("\n" + "=" * 55)
    print("LOADING SCENESENSE MASKFORMER")
    print("=" * 55)

    print("Model:", MODEL_ID)
    print("Device:", DEVICE)

    print("\nLoading image processor...")

    # Load the image processor that prepares photographs
    # in the format expected by the pretrained model.
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False
    )

    print("Loading pretrained MaskFormer...")

    # Load the MaskFormer architecture and its pretrained weights.
    model = (
        MaskFormerForInstanceSegmentation
        .from_pretrained(MODEL_ID)
    )

    # Move the model to the selected device, which is the CPU.
    model.to(torch.device(DEVICE))

    # Use evaluation mode because the model is only making
    # predictions and is not being trained.
    model.eval()

    # Check whether the loaded model supports the expected
    # number of semantic categories.
    actual_classes = int(model.config.num_labels)

    if actual_classes != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"but the model contains {actual_classes}."
        )

    print("\nMaskFormer loaded successfully!")
    print("Classes:", actual_classes)
    print("Device:", DEVICE)

    # Return the processor and model for image prediction.
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
    Predict a semantic category for every pixel.

    Returns a two-dimensional NumPy array containing
    the predicted class identifiers.
    """

    # Make sure the input is a valid Pillow image.
    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL image."
        )

    if image.width < 1 or image.height < 1:
        raise ValueError(
            "The image must have positive dimensions."
        )

    # Convert the image into Red, Green and Blue colour mode.
    image = image.convert("RGB")

    # The image processor resizes and normalizes the image
    # according to the model settings and creates PyTorch tensors.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move the input tensors to the same device as the model.
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Inference mode avoids unnecessary gradient calculations,
    # since we are using the model for prediction only.
    with torch.inference_mode():

        # Pass the processed image through MaskFormer.
        outputs = model(**inputs)

        # Convert the raw model predictions into a semantic
        # class map matching the uploaded image dimensions.
        segmentation = (
            processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[
                    (image.height, image.width)
                ]
            )[0]
        )

    # Convert the PyTorch output into a NumPy array.
    # Each pixel stores a class identifier from 0 to 170.
    labels = (
        segmentation
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # The segmentation mask must have the same
    # height and width as the processed image.
    expected_shape = (
        image.height,
        image.width
    )

    if labels.shape != expected_shape:
        raise RuntimeError(
            "Mask dimensions do not match "
            "the processed image."
        )

    # Make sure the output is not empty and contains
    # only valid semantic class identifiers.
    if (
        labels.size == 0
        or int(labels.max()) >= NUM_CLASSES
    ):
        raise RuntimeError(
            "The model returned invalid semantic class IDs."
        )

    # Return the predicted class map to the segmentation pipeline.
    return labels


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def get_model_information():
    """Return basic information about the selected model."""

    return {
        "name": "MaskFormer ResNet-50 COCO-Stuff",
        "checkpoint": MODEL_ID,
        "classes": NUM_CLASSES,
        "device": DEVICE,
        "task": "semantic segmentation",
    }
