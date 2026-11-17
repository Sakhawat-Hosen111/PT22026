
"""Statistical analysis utilities for SceneSense."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def summarize_mask(
    label_map: np.ndarray,
    class_names: Mapping
) -> list[list]:
    """
    Calculate pixel counts and area percentages
    for every semantic category detected.

    Returns:
        List of rows:
        [class_name, pixel_count, area_percentage]
    """

    labels = np.asarray(label_map)

    if labels.ndim != 2:
        raise ValueError(
            "Label map must be a 2-D array."
        )

    if labels.size == 0:
        raise ValueError(
            "Label map cannot be empty."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Label map must contain integer class IDs."
        )

    if not isinstance(class_names, Mapping):
        raise TypeError(
            "class_names must be a mapping of class IDs to names."
        )

    unique_ids, counts = np.unique(
        labels,
        return_counts=True
    )

    total_pixels = int(labels.size)

    results = []

    for class_id, pixel_count in zip(unique_ids, counts):

        class_id = int(class_id)
        pixel_count = int(pixel_count)

        name = class_names.get(
            class_id,
            class_names.get(str(class_id), f"class_{class_id}")
        )

        percentage = (
            pixel_count / total_pixels
        ) * 100.0

        results.append([
            str(name),
            pixel_count,
            round(percentage, 2)
        ])

    # Sort categories by their pixel area.
    results.sort(
        key=lambda row: row[1],
        reverse=True
    )

    return results
