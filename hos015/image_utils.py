
import numpy as np
from PIL import Image


MAX_IMAGE_PIXELS = 16_000_000


def validate_image(image: Image.Image) -> None:
    # Check if the input is a Pillow image
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload a valid image.")

    width, height = image.size

    # Check the image dimensions
    if width <= 0 or height <= 0:
        raise ValueError("The image dimensions are invalid.")

    # Avoid processing very large images
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "The image is too large. "
            "Please use an image with 16 million pixels or less."
        )


def get_image_info(image: Image.Image) -> str:
    # Get the original image information
    width, height = image.size
    color_mode = image.mode

    info = (
        f"Width: {width} pixels\n"
        f"Height: {height} pixels\n"
        f"Original color mode: {color_mode}"
    )

    return info


def convert_to_rgb(image: Image.Image) -> Image.Image:
    # Convert the image to three color channels
    rgb_image = image.convert("RGB")

    return rgb_image


def resize_image(
    image: Image.Image,
    max_size: int = 768
) -> Image.Image:
    # Check the selected image size
    if max_size <= 0:
        raise ValueError("Image size must be greater than zero.")

    # Keep the original image unchanged
    resized_image = image.copy()

    # Reduce the size while keeping the aspect ratio
    resized_image.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    return resized_image


def get_pixel_info(image: Image.Image) -> str:
    # Convert the RGB image into a NumPy array
    pixels = np.asarray(image)

    # Get the height, width, and number of channels
    height, width, channels = pixels.shape

    # Check the RGB values of the first pixel
    first_pixel = pixels[0, 0].tolist()

    info = (
        f"Array shape: {pixels.shape}\n"
        f"Number of channels: {channels}\n"
        f"Data type: {pixels.dtype}\n"
        f"Top-left pixel (RGB): {first_pixel}"
    )

    return info
