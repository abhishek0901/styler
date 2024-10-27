import constants
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from sam_model import MaskNClassifier
from utils import download_classification_model

import modal

app = modal.App("stackup-image-classifier")


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
        "numpy",
        "scikit-learn",
        "git+https://github.com/facebookresearch/segment-anything.git",
    )
    .run_commands(
        "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    .run_function(
        download_classification_model,
        kwargs={
            "repo_id": constants.CLASSIFICATION_REPO_ID,
            "local_dir": constants.CLASSIFICATION_MODEL_PATH,
        },
    )
)


web_app = FastAPI()
mask_classifier = MaskNClassifier()


@web_app.get("/metadata")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        metadata = mask_classifier.get_metadata(image=img)

    except Exception as e:
        return {"message": "error", "body": str(e), "status": 500}
    finally:
        file.file.close()

    return {"message": "successfull", "metadata": metadata, "status": 200}
