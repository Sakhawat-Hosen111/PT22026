
import gradio as gr


def show_image(image):
    # Check if the user uploaded an image
    if image is None:
        raise gr.Error("Please upload an image first.")

    # For now, just display the uploaded image
    return image


app = gr.Interface(
    fn=show_image,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Image(label="Image Preview"),
    title="SceneSense - Image Segmentation",
    description="Upload an image to preview it."
)


if __name__ == "__main__":
    app.launch()
