
"""Configuration for the SceneSense MaskFormer application."""

MODEL_ID = "facebook/maskformer-resnet50-coco-stuff" #This is the pretrained model i used for segmentation

DEVICE = "cpu"

NUM_CLASSES = 171  # number of classes in the COCO-Stuff dataset

DEFAULT_OVERLAY_OPACITY = 0.35 # overlay means the segmentation mask is drawn on top of the original image with this opacity

MAX_IMAGE_SIDE = 640 # maximum image side for inference

MAX_INPUT_PIXELS = 40_000_000 # maximum number of pixels in the input image

MIN_IMAGE_SIDE = 32 # minimum image side for inference
