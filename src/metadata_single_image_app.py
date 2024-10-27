import gradio as gr
from PIL import Image
import tempfile

from generate_matadta_single_image import get_metadata

def generate_metadata(image):
    # Assuming input_json is a list of dictionaries
    # You can add your processing logic here
    image = Image.fromarray(image)
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        image.save(f"{tmpdirname}/temp_image.png")
        mt_data = get_metadata(f"{tmpdirname}/temp_image.png")

    return mt_data

# Set up the Gradio interface
iface = gr.Interface(
    fn=generate_metadata,
    inputs=gr.Image(label="Input Image"),
    outputs=gr.JSON(label="Output JSON"),
    title="List of JSON Dictionaries Example",
    description="Input a list of JSON objects and see the output."
)

# Launch the interface
iface.launch(share=True)
