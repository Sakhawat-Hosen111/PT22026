
"""
SceneSense - Segmentation Statistics

This file calculates the number of pixels and the
percentage of image area covered by each detected category.
"""

from __future__ import annotations

# Mapping is used to check whether the class names
# are provided in a dictionary-like format.
from collections.abc import Mapping

# NumPy is used to work with the predicted pixel class array.
import numpy as np


def summarize_mask(
    label_map: np.ndarray,
    class_names: Mapping
) -> list[list]:
    """
    Calculate pixel counts and area percentages
    for every semantic category detected.

    Returns:
        A list containing category names,
        pixel counts and area percentages.
    """

    # Convert the predicted segmentation map into a NumPy array.
    labels = np.asarray(label_map)

    # A semantic segmentation map should have two dimensions:
    # image height and image width.
    if labels.ndim != 2:
        raise ValueError(
            "Label map must be a 2-D array."
        )

    # An empty array cannot be used to calculate statistics.
    if labels.size == 0:
        raise ValueError(
            "Label map cannot be empty."
        )

    # Every pixel must contain an integer class identifier.
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Label map must contain integer class IDs."
        )

    # Make sure the category names are provided
    # using a dictionary or another mapping object.
    if not isinstance(class_names, Mapping):
        raise TypeError(
            "class_names must be a mapping of class IDs to names."
        )

    # Find all unique class identifiers in the image
    # and count how many pixels belong to each class.
    unique_ids, counts = np.unique(
        labels,
        return_counts=True
    )

    # Count the total number of pixels in the segmentation map.
    total_pixels = int(labels.size)

    # Store the statistics for each detected category.
    results = []

    # Process every detected class and its corresponding pixel count.
    for class_id, pixel_count in zip(unique_ids, counts):

        class_id = int(class_id)
        pixel_count = int(pixel_count)

        # Find the category name using its class identifier.
        # Use a default name if the category is missing.
        name = class_names.get(
            class_id,
            class_names.get(str(class_id), f"class_{class_id}")
        )

        # Calculate what percentage of the image
        # is covered by this predicted semantic category.
        percentage = (
            pixel_count / total_pixels
        ) * 100.0

        # Store the category name, pixel count and percentage.
        results.append([
            str(name),
            pixel_count,
            round(percentage, 2)
        ])

    # Show categories with the largest pixel coverage first.
    results.sort(
        key=lambda row: row[1],
        reverse=True
    )

    return results
