import gradio as gr

# Function to process the list of JSON dictionaries
def process_json_list(input_json):
    # Assuming input_json is a list of dictionaries
    # You can add your processing logic here
    return {"message": "Received!"}

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_json_list,
    inputs=gr.Image(label="Input Image"),
    outputs=gr.JSON(label="Output JSON"),
    title="List of JSON Dictionaries Example",
    description="Input a list of JSON objects and see the output."
)

# Launch the interface
iface.launch(share=True)
