
"""
SceneSense - MaskFormer Model Verification

This script checks whether the pretrained MaskFormer
model loads correctly and produces valid predictions.

Run:
    python -m scripts.verify_model

Test one photograph:
    python -m scripts.verify_model --image "test_images/dogs.png"
"""

from __future__ import annotations

# argparse allows us to provide an image path through the terminal.
import argparse

# Used to measure how long the model takes to make a prediction.
import time

# Path is used to work with image file paths.
from pathlib import Path

import numpy as np

from PIL import Image, ImageOps

from segmentation.config import (
    DEVICE,
    MODEL_ID,
    NUM_CLASSES,
)

# Import the functions needed to load MaskFormer
# and predict semantic categories for the image.
from segmentation.model import (
    load_model,
    predict_label_map,
)

from segmentation.pipeline import prepare_image

from segmentation.analytics import summarize_mask


def main():

    # Create a command-line argument parser so we can
    # optionally provide an image when running this script.
    parser = argparse.ArgumentParser(
        description="Verify SceneSense MaskFormer."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None
    )

    # Read the arguments provided through the terminal.
    args = parser.parse_args()

    # If no image is provided, create a simple test image
    # to check whether the model can complete inference.
    if args.image is None:

        image = Image.new(
            "RGB",
            (480, 320),
            (120, 160, 190)
        )

        print(
            "Using a synthetic image for the inference check."
        )

    else:

        # Check whether the provided image file exists.
        if not args.image.is_file():
            parser.error(
                f"Image not found: {args.image}"
            )

        # Open the image, correct its orientation
        # and convert it to Red, Green and Blue mode.
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    # Prepare the image using the same preprocessing
    # function used by the main SceneSense application.
    prepared = prepare_image(image)

    # Load the pretrained MaskFormer model and image processor.
    processor, model = load_model()

    # Record the starting time before model inference.
    start = time.perf_counter()

    # Run MaskFormer to predict a semantic class for every pixel.
    labels = predict_label_map(
        prepared,
        processor,
        model
    )

    # Calculate the time taken for inference and postprocessing.
    elapsed = time.perf_counter() - start

    # Check whether the predicted mask matches the image dimensions.
    assert labels.shape == (
        prepared.height,
        prepared.width
    )

    # Check that the output uses eight-bit unsigned integers.
    assert labels.dtype == np.uint8

    # Make sure the predicted class identifiers are valid.
    assert int(labels.max()) < NUM_CLASSES

    # Get the class names from the pretrained model configuration.
    names = {
        int(index): str(label).strip()
        for index, label in model.config.id2label.items()
    }

    # Calculate the number of pixels and percentage
    # of image area covered by each detected category.
    rows = summarize_mask(
        labels,
        names
    )

    print("\n" + "=" * 55)
    print("SCENESENSE MASKFORMER VERIFICATION")
    print("=" * 55)

    # Display the model information and inference time.
    print("Checkpoint:", MODEL_ID)
    print("Device:", DEVICE)
    print("Classes:", NUM_CLASSES)
    print("Image size:", prepared.size)
    print(f"Inference: {elapsed:.2f} seconds")

    print("\nDetected categories:")

    # Display the first eight categories, sorted by pixel coverage.
    for name, pixels, percentage in rows[:8]:
        print(
            f"{name:20s} "
            f"{pixels:>9,} pixels "
            f"{percentage:6.2f}%"
        )

    print("\nMODEL VERIFICATION PASSED")

    # This confirms the model runs correctly,
    # but it does not measure segmentation accuracy.
    print(
        "This verifies inference, "
        "not segmentation accuracy."
    )


# Run the verification only when this script is executed directly.
if __name__ == "__main__":
    main()
