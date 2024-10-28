import requests

url = "https://bewajafarwah--hf-stackup-image-classifier-fastapi-app.modal.run/metadata"
file = {
    "file": open("/Users/farwah/Downloads/inv_fullxfull.3426179582_ad4qvtov.webp", "rb")
}
resp = requests.get(url=url, files=file)
print(resp.json())
