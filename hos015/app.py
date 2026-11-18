
"""
SceneSense
Semantic Image Segmentation using MaskFormer
"""

# Gradio is used to create the web interface for SceneSense.
import gradio as gr

# This function handles image processing, model prediction and results.
from segmentation.pipeline import run_segmentation


# ============================================================
# APPLICATION STYLE
# ============================================================

# CSS is used to control the page width, spacing and title appearance.
CSS = """
.gradio-container {
    width: 100% !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px 24px !important;
    box-sizing: border-box !important;
}

#header {
    text-align: center;
    margin-bottom: 20px;
}

#header h1 {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 0;
}
"""


# ============================================================
# SEGMENTATION FUNCTION
# ============================================================

def segment_image(image, opacity):
    """
    Run semantic segmentation on the uploaded image
    and return the mask, overlay, statistics and summary.
    """

    # Show an error if the user clicks Segment Image without uploading a photo.
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Send the image and overlay strength to the segmentation pipeline.
    return run_segmentation(image, opacity)


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_results():
    """Clear the uploaded image and all segmentation results."""

    # Return empty values for the input image and four result components.
    return None, None, None, [], ""


# ============================================================
# GRADIO APPLICATION
# ============================================================

# Blocks allows us to create a custom interface using rows,
# columns, buttons and other interactive components.
with gr.Blocks(
    title="SceneSense",
    fill_width=True
) as demo:

    gr.Markdown(
        "# SceneSense",
        elem_id="header"
    )

    # Row places the three main image sections horizontally.
    with gr.Row(equal_height=False):

        # First column contains the image upload and controls.
        with gr.Column(
            scale=1,
            min_width=260
        ):

            # type="pil" returns the uploaded photo as a Pillow image.
            input_image = gr.Image(
                type="pil",
                label="Upload Photograph",
                height=350
            )

            # The slider controls how visible the segmentation
            # colours are over the original photograph.
            opacity_slider = gr.Slider(
                minimum=0.0,
                maximum=0.8,
                value=0.35,
                step=0.05,
                label="Overlay Strength"
            )

            with gr.Row():

                segment_button = gr.Button(
                    "Segment Image",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "Clear",
                    variant="secondary"
                )

        # Second column displays the coloured semantic segmentation mask.
        with gr.Column(
            scale=1,
            min_width=260
        ):

            mask_output = gr.Image(
                type="pil",
                label="Semantic Mask",
                height=350,
                interactive=False
            )

        # Third column displays the mask blended with the original image.
        with gr.Column(
            scale=1,
            min_width=260
        ):

            overlay_output = gr.Image(
                type="pil",
                label="Segmentation Overlay",
                height=350,
                interactive=False
            )

    # --------------------------------------------------------
    # SEGMENTATION RESULTS
    # --------------------------------------------------------

    gr.Markdown("### Segmentation Results")

    # Markdown is used to show image dimensions, detected
    # categories and the time taken for model inference.
    summary_output = gr.Markdown()

    # Dataframe displays the detected class names,
    # number of pixels and percentage of image area.
    class_table = gr.Dataframe(
        headers=[
            "Category",
            "Pixels",
            "Area (%)"
        ],
        datatype=[
            "str",
            "number",
            "number"
        ],
        row_count=0,
        column_count=3,
        interactive=False,
        label="Detected Categories",
        max_height=350
    )

    # --------------------------------------------------------
    # BUTTON EVENTS
    # --------------------------------------------------------

    # When Segment Image is clicked, Gradio passes the image
    # and opacity to segment_image(), then displays its four results.
    segment_button.click(
        fn=segment_image,
        inputs=[
            input_image,
            opacity_slider
        ],
        outputs=[
            mask_output,
            overlay_output,
            class_table,
            summary_output
        ]
    )

    # The Clear button calls clear_results() to reset the interface.
    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[
            input_image,
            mask_output,
            overlay_output,
            class_table,
            summary_output
        ]
    )


# ============================================================
# START APPLICATION
# ============================================================

# Only launch the application when this file is executed directly.
if __name__ == "__main__":

    # Allow one queued segmentation request to run at a time
    # to avoid unnecessary Central Processing Unit and memory usage.
    demo.queue(
        default_concurrency_limit=1
    )

    # Start the application locally with the selected theme and CSS.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Default(
            primary_hue="teal",
            neutral_hue="slate"
        ),
        css=CSS,
        footer_links=[]
    )
