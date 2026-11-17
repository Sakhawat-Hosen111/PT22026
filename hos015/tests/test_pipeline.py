
"""
SceneSense - Pipeline Tests

Tests image preprocessing and application outputs
without loading the real pretrained model.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from PIL import Image

import segmentation.pipeline as pipeline


def test_rgb_conversion():

    result = pipeline.prepare_image(
        Image.new("RGBA", (100, 80))
    )

    assert result.mode == "RGB"
    assert result.size == (100, 80)


def test_resize_keeps_aspect_ratio():

    result = pipeline.prepare_image(
        Image.new("RGB", (2000, 1000))
    )

    assert result.size == (640, 320)


def test_small_image_rejected():

    with pytest.raises(ValueError):

        pipeline.prepare_image(
            Image.new("RGB", (20, 20))
        )


def test_missing_image_rejected():

    with pytest.raises(ValueError):

        pipeline.prepare_image(None)


def test_invalid_input_rejected():

    with pytest.raises(TypeError):

        pipeline.prepare_image(
            "not an image"
        )


def test_invalid_opacity_rejected():

    with pytest.raises(ValueError):

        pipeline.run_segmentation(
            Image.new("RGB", (100, 80)),
            opacity=1.5
        )


def test_full_pipeline_with_fake_model(monkeypatch):
    """
    Verify pipeline integration without downloading
    or running the real MaskFormer model.
    """

    # Create the class vocabulary.
    names = {
        i: f"category_{i}"
        for i in range(171)
    }

    names[0] = "person"
    names[16] = "dog"

    # Create a fake model.
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            id2label=names,
            num_labels=171
        )
    )

    # Replace real model loading in this test.
    monkeypatch.setattr(
        pipeline,
        "load_model",
        lambda: (object(), fake_model)
    )

    # Return a synthetic prediction:
    # left half = person, right half = dog.
    def fake_prediction(image, processor, model):

        labels = np.zeros(
            (image.height, image.width),
            dtype=np.uint8
        )

        labels[:, image.width // 2:] = 16

        return labels

    monkeypatch.setattr(
        pipeline,
        "predict_label_map",
        fake_prediction
    )

    # Run the complete pipeline.
    output = pipeline.run_segmentation(
        Image.new("RGB", (100, 80)),
        0.35
    )

    assert len(output) == 4

    color_mask, overlay, rows, summary = output

    assert color_mask.size == (100, 80)

    assert overlay.size == (100, 80)

    assert {
        row[0]
        for row in rows
    } == {
        "person",
        "dog"
    }

    assert sum(
        row[1]
        for row in rows
    ) == 8000

    assert "MaskFormer" in summary
