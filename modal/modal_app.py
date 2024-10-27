import constants
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
        f"wget -P {constants.SAM_CHECKPOINT} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    .run_function(
        download_classification_model,
        kwargs={
            "repo_id": constants.CLASSIFICATION_REPO_ID,
            "local_dir": constants.CLASSIFICATION_MODEL_PATH,
        },
    )
)
