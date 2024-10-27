from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
meta_data_list = []
elem1 = {
    'image_name':'khaki_pant', 'bbox': [], 'label': 'Pants', 'condifence': 0.9, 'color': 'peru', 'top_category': 'Bottom'
}
elem2 = {
    'image_name':'blue_shirt', 'bbox': [], 'label': 'Upper-clothes', 'condifence': 0.9, 'color': 'darkslategrey', 'top_category': 'Top'
}
elem3 = {
    'image_name':'green_shirt', 'bbox': [], 'label': 'Upper-clothes', 'condifence': 0.9, 'color': 'mediumseagreen', 'top_category': 'Top'
}
elem4 = {
    'image_name':'light_pant', 'bbox': [], 'label': 'Pants', 'condifence': 0.9, 'color': 'silver', 'top_category': 'Bottom'
}
meta_data_list.append(elem1)
meta_data_list.append(elem2)
meta_data_list.append(elem3)
meta_data_list.append(elem4)
result = client.predict(
		input_json=meta_data_list,
		api_name="/predict"
)
print(result)