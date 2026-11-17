
"""
SceneSense - MaskFormer Model Verification

Run:
    python -m scripts.verify_model

Test one photograph:
    python -m scripts.verify_model --image "test_images/dogs.png"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from PIL import Image, ImageOps

from segmentation.config import (
    DEVICE,
    MODEL_ID,
    NUM_CLASSES,
)

from segmentation.model import (
    load_model,
    predict_label_map,
)

from segmentation.pipeline import prepare_image

from segmentation.analytics import summarize_mask


def main():

    parser = argparse.ArgumentParser(
        description="Verify SceneSense MaskFormer."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None
    )

    args = parser.parse_args()

    # Use a real image if one is provided.
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

        if not args.image.is_file():
            parser.error(
                f"Image not found: {args.image}"
            )

        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    # Prepare image and load model.
    prepared = prepare_image(image)

    processor, model = load_model()

    # Measure inference time.
    start = time.perf_counter()

    labels = predict_label_map(
        prepared,
        processor,
        model
    )

    elapsed = time.perf_counter() - start

    # Verify the prediction.
    assert labels.shape == (
        prepared.height,
        prepared.width
    )

    assert labels.dtype == np.uint8

    assert int(labels.max()) < NUM_CLASSES

    # Get detected category statistics.
    names = {
        int(index): str(label).strip()
        for index, label in model.config.id2label.items()
    }

    rows = summarize_mask(
        labels,
        names
    )

    print("\n" + "=" * 55)
    print("SCENESENSE MASKFORMER VERIFICATION")
    print("=" * 55)

    print("Checkpoint:", MODEL_ID)
    print("Device:", DEVICE)
    print("Classes:", NUM_CLASSES)
    print("Image size:", prepared.size)
    print(f"Inference: {elapsed:.2f} seconds")

    print("\nDetected categories:")

    for name, pixels, percentage in rows[:8]:
        print(
            f"{name:20s} "
            f"{pixels:>9,} pixels "
            f"{percentage:6.2f}%"
        )

    print("\nMODEL VERIFICATION PASSED")

    print(
        "This verifies inference, "
        "not segmentation accuracy."
    )


if __name__ == "__main__":
    main()
