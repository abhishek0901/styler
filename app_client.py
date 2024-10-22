from gradio_client import Client

client = Client("abhi995/styler")  # connecting to a Hugging Face Space
print(client.predict("Abhishek", api_name="/predict"))