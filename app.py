import gradio as gr
import pandas as pd
import numpy as np
import joblib, os

def get_str(name):
    return f"Hello {name}"

def predict(name):
    response = get_str(name)

    return response


demo = gr.Interface(fn=predict, inputs="textbox", outputs="textbox")

if __name__ == "__main__":
    demo.launch()


# curl -X POST -H 'Content-type: application/json' --data '{ "data": ["Jill"] }' https://abhi995-styler.hf.space//gradio_api/queue/join