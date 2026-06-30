import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import time


class CaptionModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[INFO] Using device:", self.device)

        # self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/git-base"
        # ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
        "microsoft/git-base",
        local_files_only=True

    )

        self.model = AutoModelForCausalLM.from_pretrained(
    "microsoft/git-base",
    local_files_only=True
    ).to(self.device)
        self.model.eval()

        print("[INFO] Warming up model...")
        self.caption(Image.new("RGB", (224, 224)))
        print("[INFO] Warm-up complete")

    def caption(self, image_pil):
        start = time.time()

        image_pil = image_pil.resize((224, 224))

        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.batch_decode(
            output,
            skip_special_tokens=True
        )[0].strip()

        print(f"[GIT] {time.time() - start:.2f}s -> {caption}")

        return caption
