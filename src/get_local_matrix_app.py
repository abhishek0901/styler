import gradio as gr
import json

from get_local_matrix import get_local_matrix

class Matrix:
    def __init__(self, color_to_index,cosine_similarity_exp):
        self.color_to_index = color_to_index
        self.cosine_similarity_exp = cosine_similarity_exp

# Function to process the JSON input
def get_matrix(input_json):
    # Process list
    local_matrix, image_order = get_local_matrix(input_json)
    return {
        "message": "Received!", 
        "local_matrix": local_matrix,
        "image_order":image_order
        }

# Set up the Gradio interface
iface = gr.Interface(
    fn=get_matrix,
    inputs=gr.JSON(label="Input JSON"),
    outputs=gr.JSON(label="Output JSON"),
    title="Local Matrix generator",
    description="API to get local matrix"
)

# Launch the interface
iface.launch()