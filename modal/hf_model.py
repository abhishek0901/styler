####################################
# Author: Abhishek Srivastava
# Author: Harman Farwah
# Description : This file hosts endpoint that will generate clothes segmentation metadata for single image
####################################
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import webcolors
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

import modal

logger = logging.getLogger(__name__)

CLOTH_MAP = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "DRESSES",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

CLOTH_OF_INTEREST = [4, 5, 6, 7]


def download_classification_model(repo_id, local_dir):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.pt", "*.bin"],
        local_dir=local_dir,
    )


repo_id = "sayeed99/segformer_b3_clothes"
local_dir = "/classification_model"

image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "python3-opencv")
    .pip_install(
        "pillow",
        "torch",
        "torchvision",
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "onnxruntime",
        "onnx",
        "huggingface_hub",
        "webcolors",
        "numpy==2.1.1",
        "transformers",
        "scikit-learn",
    )
    .run_function(
        download_classification_model,
        kwargs={"repo_id": repo_id, "local_dir": local_dir},
    )
)

app = modal.App("hf-stackup-image-classifier", image=image)

IDLE_TIMEOUT_TIME = 5 * 60  # 5 MINS


@app.cls(gpu="T4", container_idle_timeout=IDLE_TIMEOUT_TIME)
class MaskNClassifier:
    @modal.enter()
    def setup(self):
        self.processor = SegformerImageProcessor.from_pretrained(local_dir)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(local_dir)

    def rgb_to_color(self, requested_colour):
        min_colours = {}
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    @modal.method()
    def get_metadata(self, image_array):
        """
        Description
        --------------
            1. Load model
            2. Get Predictions
            3. Get Type
            4. Get Color
            5. Fill Values
            6. Return Dict
        """
        image = Image.fromarray(image_array)

        inputs = self.processor(images=image, return_tensors="pt")

        # get predictions
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get type
        cloth_type_predicted_index = torch.mean(
            upsampled_logits[0, CLOTH_OF_INTEREST], dim=[1, 2]
        ).argmax()
        cloth_type = CLOTH_MAP[CLOTH_OF_INTEREST[cloth_type_predicted_index]]

        # get color
        t1 = (
            upsampled_logits[0, CLOTH_OF_INTEREST[cloth_type_predicted_index]]
            .detach()
            .numpy()
        )
        t1[t1 < t1.mean()] = 0
        t1[t1 >= t1.mean()] = 1
        t2 = t1.reshape((t1.shape[0], t1.shape[1], 1))
        original_image = image_array.copy()  # cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_copy = original_image * t2
        rgb_values = original_image_copy[np.nonzero(t1)].mean(axis=0)

        # convert them to int
        rgb_values = rgb_values.astype(int)
        color = self.rgb_to_color(rgb_values)

        # Meta data
        metadata = {
            "bbox": [],
            "label": "default",
            "condifence": 0.9,
            "color": "default",
            "top_category": "default",
        }

        metadata["label"] = cloth_type
        metadata["color"] = color
        metadata["top_category"] = (
            "Top" if cloth_type in ["Upper-clothes", "DRESSES"] else "Bottom"
        )

        return metadata


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    web_app = FastAPI()
    mask_classifier = MaskNClassifier()

    @web_app.get("/metadata")
    async def create_upload_file(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            nparr = np.fromstring(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            metadata = mask_classifier.get_metadata.remote(image_array=img)
        except Exception as e:
            logger.error(e, exc_info=True)
            return {"message": "error", "body": str(e), "status": 500}
        finally:
            file.file.close()

        return JSONResponse(
            content={"message": "successfull", "metadata": metadata, "status": 200}
        )

    return web_app
