import requests

url = " https://bewajafarwah--stackup-image-classifier-fastapi-app.modal.run/metadata"
file = {"file": open("../src/tmp_dir/blue_shirt.png", "rb")}
resp = requests.get(url=url, files=file)
print(resp.json())


