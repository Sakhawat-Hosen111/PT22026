
from PIL import Image


def get_image_info(image: Image.Image) -> str:
    # Get the image size
    width, height = image.size

    # Check the original color format
    color_mode = image.mode

    info = (
        f"Width: {width} pixels\n"
        f"Height: {height} pixels\n"
        f"Original color mode: {color_mode}"
    )

    return info


def convert_to_rgb(image: Image.Image) -> Image.Image:
    # Convert the image to RGB
    rgb_image = image.convert("RGB")

    return rgb_image


def resize_image(
    image: Image.Image,
    max_size: int = 768
) -> Image.Image:
    # Check that the selected size is valid
    if max_size <= 0:
        raise ValueError("Image size must be greater than zero.")

    # Make a copy so the original image stays unchanged
    resized_image = image.copy()

    # Resize without stretching the image
    resized_image.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    return resized_image
