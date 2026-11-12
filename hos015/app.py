
import gradio as gr

from image_utils import (
    get_image_info,
    convert_to_rgb,
    resize_image
)


def show_image(image, max_size):
    # Check if an image was uploaded
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Get the original image information
    image_info = get_image_info(image)

    # Convert the image to RGB
    rgb_image = convert_to_rgb(image)

    # Resize the RGB image
    resized_image = resize_image(
        rgb_image,
        max_size=int(max_size)
    )

    # Get the new dimensions
    new_width, new_height = resized_image.size

    # Add the processed image details
    image_info += (
        f"\nProcessed color mode: {resized_image.mode}"
        f"\nResized width: {new_width} pixels"
        f"\nResized height: {new_height} pixels"
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
        gr.Image(label="Resized Image"),
        gr.Textbox(label="Image Details")
    ],

    title="SceneSense - Image Segmentation",

    description=(
        "Upload an image and choose the maximum size. "
        "The image will be resized while keeping "
        "its original proportions."
    )
)


if __name__ == "__main__":
    app.launch()
