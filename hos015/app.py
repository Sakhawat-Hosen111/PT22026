
import gradio as gr

from image_utils import (
    validate_image,
    get_image_info,
    convert_to_rgb,
    resize_image,
    get_pixel_info
)


def show_image(image, max_size):
    # Make sure the user uploaded an image
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Check the image before processing
    try:
        validate_image(image)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    # Get the original image details
    image_info = get_image_info(image)

    # Convert the image to RGB
    rgb_image = convert_to_rgb(image)

    # Resize the image
    resized_image = resize_image(
        rgb_image,
        max_size=int(max_size)
    )

    # Get the resized dimensions
    new_width, new_height = resized_image.size

    # Read the numerical pixel information
    pixel_info = get_pixel_info(resized_image)

    # Combine all the image information
    image_info += (
        f"\nProcessed color mode: {resized_image.mode}"
        f"\nResized width: {new_width} pixels"
        f"\nResized height: {new_height} pixels"
        f"\n\n{pixel_info}"
    )

    return resized_image, image_info


app = gr.Interface(
    fn=show_image,

    inputs=[
        gr.Image(
            type="pil",
            label="Input Image"
        ),

        gr.Slider(
            minimum=256,
            maximum=1024,
            value=768,
            step=128,
            label="Maximum Image Size"
        )
    ],

    outputs=[
        gr.Image(label="Processed Image"),
        gr.Textbox(
            label="Image and Pixel Information",
            lines=10
        )
    ],

    title="SceneSense - Image Segmentation",

    description=(
        "Upload an image to check its dimensions, "
        "convert it to RGB, resize it, "
        "and inspect its pixel information."
    )
)


if __name__ == "__main__":
    app.launch()
