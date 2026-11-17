
"""
SceneSense
Semantic Image Segmentation using MaskFormer
"""

import gradio as gr

from segmentation.pipeline import run_segmentation


# ============================================================
# APPLICATION STYLE
# ============================================================

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
    Run MaskFormer semantic segmentation.

    Returns:
        - Colored segmentation mask
        - Segmentation overlay
        - Detected category statistics
        - Result summary
    """

    if image is None:
        raise gr.Error("Please upload an image first.")

    return run_segmentation(image, opacity)


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_results():
    """Reset all image inputs and outputs."""

    return None, None, None, [], ""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="SceneSense",
    fill_width=True
) as demo:

    # Main title
    gr.Markdown(
        "# SceneSense",
        elem_id="header"
    )

    # --------------------------------------------------------
    # THREE-COLUMN IMAGE LAYOUT
    # --------------------------------------------------------

    with gr.Row(equal_height=False):

        # COLUMN 1: INPUT IMAGE AND CONTROLS

        with gr.Column(
            scale=1,
            min_width=260
        ):

            input_image = gr.Image(
                type="pil",
                label="Upload Photograph",
                height=350
            )

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

        # COLUMN 2: SEMANTIC SEGMENTATION MASK

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

        # COLUMN 3: SEGMENTATION OVERLAY

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

    summary_output = gr.Markdown()

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

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1
    )

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
