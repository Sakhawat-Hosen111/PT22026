
"""Tests for SceneSense semantic class statistics."""

import numpy as np
import pytest

from segmentation.analytics import summarize_mask


def test_class_pixel_counts():
    """Check whether pixels are counted correctly."""

    mask = np.array(
        [
            [12, 12, 0, 0],
            [12, 12, 0, 2],
        ],
        dtype=np.uint8
    )

    names = {
        0: "wall",
        2: "sky",
        12: "person",
    }

    result = summarize_mask(mask, names)

    assert result[0] == ["person", 4, 50.0]
    assert result[1] == ["wall", 3, 37.5]
    assert result[2] == ["sky", 1, 12.5]


def test_single_class_coverage():
    """A single predicted class should cover the entire image."""

    mask = np.full(
        (10, 10),
        2,
        dtype=np.uint8
    )

    result = summarize_mask(
        mask,
        {2: "sky"}
    )

    assert result == [
        ["sky", 100, 100.0]
    ]


def test_missing_category_name():
    """Unknown class IDs should have fallback names."""

    mask = np.full(
        (4, 4),
        25,
        dtype=np.uint8
    )

    result = summarize_mask(mask, {})

    assert result[0][0] == "class_25"
    assert result[0][1] == 16
    assert result[0][2] == 100.0


def test_invalid_mask_dimensions():
    """The analytics module must reject non-2D masks."""

    with pytest.raises(ValueError):
        summarize_mask(
            np.zeros((10, 10, 3), dtype=np.uint8),
            {}
        )


def test_empty_mask():
    """Empty predictions are invalid."""

    with pytest.raises(ValueError):
        summarize_mask(
            np.empty((0, 0), dtype=np.uint8),
            {}
        )


def test_area_percentages():
    """Reported area percentages should total approximately 100%."""

    mask = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.uint8
    )

    result = summarize_mask(
        mask,
        {0: "wall", 1: "building", 2: "sky"}
    )

    total = sum(row[2] for row in result)

    assert total == pytest.approx(
        100.0,
        abs=0.02
    )
