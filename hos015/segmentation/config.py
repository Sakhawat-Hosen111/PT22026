
"""Configuration for the SceneSense MaskFormer application."""

MODEL_ID = "facebook/maskformer-resnet50-coco-stuff"

DEVICE = "cpu"

NUM_CLASSES = 171

DEFAULT_OVERLAY_OPACITY = 0.35

# Tested CPU inference used a maximum image side of 640 pixels.
MAX_IMAGE_SIDE = 640

MAX_INPUT_PIXELS = 40_000_000

MIN_IMAGE_SIDE = 32
