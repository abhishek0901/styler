import logging
import os

import constants
import cv2
import numpy as np
import torch
import webcolors
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import KMeans
from transformers import ViTForImageClassification, ViTImageProcessor
from utils import CATEGORY_MAP, download_classification_model

import modal

logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "python3-opencv")
    .pip_install(
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
        "git+https://github.com/facebookresearch/segment-anything.git",
    )
    .run_commands(
        f"wget -P {constants.SAM_CHECKPOINT_PATH} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    .run_function(
        download_classification_model,
        kwargs={
            "repo_id": constants.CLASSIFICATION_REPO_ID,
            "local_dir": constants.CLASSIFICATION_MODEL_PATH,
        },
    )
)

app = modal.App("stackup-image-classifier", image=image)


@app.cls(gpu="T4")
class MaskNClassifier:
    @modal.enter()
    def setup(self):

        logger.info("SETUP STARTED")

        sam = sam_model_registry[constants.SAM_MODEL_TYPE](
            checkpoint=os.path.join(
                constants.SAM_CHECKPOINT_PATH, constants.SAM_CHECKPOINT
            )
        )
        sam.to(device="cuda")

        self.mask_generator = SamAutomaticMaskGenerator(sam)

        self.processor = ViTImageProcessor.from_pretrained(
            constants.CLASSIFICATION_MODEL_PATH
        )
        self.classifier = ViTForImageClassification.from_pretrained(
            constants.CLASSIFICATION_MODEL_PATH
        )

        logger.info("SETUP ENDED")

    def get_masks(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def classify(self, image: np.ndarray) -> tuple:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.classifier(**inputs).logits

        logits = torch.softmax(logits, -1)
        confidence, idx = logits.max(-1)
        confidence, idx = confidence.item(), idx.item()
        label = self.classifier.config.id2label[idx]

        return label, confidence

    def _average_color(self, segmented_image):
        pixels = segmented_image.reshape(-1, 3).astype(float)

        non_black = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        if len(non_black) > 0:
            return np.mean(non_black, axis=0).astype(int)
        else:
            return None

    def closest_colour(self, requested_colour):
        min_colours = {}
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def _nearest_color(self, segmented_image, k=3):
        pixels = segmented_image.reshape(-1, 3)

        non_black = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        if len(non_black) > 0:
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(non_black)

            # Get the colors
            colors = kmeans.cluster_centers_

            # Sort colors by their frequency
            labels = kmeans.labels_
            counts = np.bincount(labels)
            sorted_indices = np.argsort(counts)[::-1]

            return colors[sorted_indices].astype(int)[0]
        else:
            return None

    def get_dominant_color(self, segmented_image, type):
        if type == "average":
            color = self._average_color(segmented_image)
        elif type == "nearest":
            color = self._nearest_color(segmented_image)
        else:
            raise NotImplementedError

        if color is None:
            raise ValueError

        color_name = self.closest_colour(color)
        return color_name

    @modal.method()
    def get_metadata(self, image: np.ndarray, find_color="nearest"):
        masks = self.get_masks(image)

        package = []

        for mask in masks:
            x, y, w, h = mask["bbox"]
            cropped_image = image[y : y + h, x : x + w].copy()
            if cropped_image.shape[0] < 2 or cropped_image.shape[1] < 2:
                continue

            cropped_image[~mask["segmentation"][y : y + h, x : x + w]] = 0.0
            label, confidence = self.classify(cropped_image)
            color = self.get_dominant_color(cropped_image, find_color)
            tc = CATEGORY_MAP.get(label, "NA")

            metadata = {
                "bbox": mask["bbox"],
                "label": label,
                "condifence": confidence,
                "color": color,
                "category": tc,
            }

            package.append(metadata)

        logger.info(package)
        return package


@app.local_entrypoint()
def cli(path: str):
    import cv2

    mask_classifier = MaskNClassifier()

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    metadata = mask_classifier.get_metadata.remote(image)  # .remote()
    print(metadata)


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
            metadata = mask_classifier.get_metadata.remote(image=img)

        except Exception as e:
            return {"message": "error", "body": str(e), "status": 500}
        finally:
            file.file.close()

        return JSONResponse(
            content={"message": "successfull", "metadata": metadata, "status": 200}
        )

    return web_app
