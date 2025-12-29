# CaptionGen.py
# ----------------------------------------
# Utilizes a Vision-Language Model to generate captions based on cropped images and annotations.
# ----------------------------------------

import json
import os
import torch
import argparse
import tarfile
from pathlib import Path
from io import BytesIO
from typing import Optional

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils.caption_utils import caption_generate
from utils.img_utils import resize_with_aspect_ratio

class UniversalCaptioner:
    """A smart captioner that infers data_root and tar_path from the JSONL location."""
    def __init__(self, model_id: str, PROMPT: str, SYSTEM: str, device: str = "auto"):
        print(f"[INFO] Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.prompt = PROMPT
        self.system = SYSTEM
        self.tar_handle: Optional[tarfile.TarFile] = None

    def load_image(self, rel_path: str, data_root: Path) -> Image.Image:
        """Attempts to load from an auto-detected TAR first, then falls back to disk."""
        if self.tar_handle:
            try:
                member = self.tar_handle.getmember(rel_path)
                return Image.open(BytesIO(self.tar_handle.extractfile(member).read())).convert("RGB")
            except (KeyError, AttributeError):
                pass
        
        # Fallback to local file system: data_root / image_path
        img_path = data_root / rel_path
        return Image.open(img_path).convert("RGB")

    def clean_caption(self, text: str) -> str:
        """Extracts the first factual sentence from model output."""
        if "assistant" in text.lower():
            text = text.split("assistant")[-1]
        return text.strip().split("\n")[0].strip()

    def process(self, jsonl_path_str: str):
        # 1. Automatic Path Inference
        jsonl_path = Path(jsonl_path_str).resolve()
        data_root = jsonl_path.parent
        print(f"[INFO] Target JSONL: {jsonl_path}")
        print(f"[INFO] Inferred Data Root: {data_root}")

        # 2. Automatic TAR Detection (looks for any .tar file in the same folder)
        tar_files = list(data_root.glob("*.tar"))
        if tar_files:
            # Prioritize a tar file that contains 'image' in the name if multiple exist
            selected_tar = next((t for t in tar_files if "image" in t.name.lower()), tar_files[0])
            print(f"[INFO] TAR archive detected: {selected_tar.name}")
            self.tar_handle = tarfile.open(selected_tar, "r")

        # 3. Checkpoint Setup
        tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
        processed_count = 0
        if tmp_path.exists():
            with open(tmp_path, "r", encoding="utf-8") as f:
                processed_count = sum(1 for _ in f)
            print(f"[INFO] Resuming from record {processed_count}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        # 4. Main Processing Loop
        with open(tmp_path, "a", encoding="utf-8") as out_f:
            for i, line in enumerate(tqdm(all_lines[processed_count:], desc="Captioning"), start=processed_count):
                record = json.loads(line)
                
                # Skip if already captioned
                if record.get("caption"):
                    out_f.write(json.dumps(record) + "\n")
                    continue

                try:
                    img = self.load_image(record["image_path"], data_root)
                    resized_img = resize_with_aspect_ratio(img, 256)
                    
                    context = f"This image shows a {record.get('label', 'object')} in a driving scene."
                    raw_out = caption_generate(context, self.prompt, self.system, resized_img, self.processor, self.model)
                    record["caption"] = self.clean_caption(raw_out)
                    
                except Exception as e:
                    print(f"\n[ERROR] Failed at {record.get('image_path')}: {e}")
                    record["caption"] = "" 

                out_f.write(json.dumps(record) + "\n")
                if i % 20 == 0: out_f.flush()

        # 5. Finalize
        if self.tar_handle: self.tar_handle.close()
        os.replace(tmp_path, jsonl_path)
        print(f"[SUCCESS] Updated {jsonl_path.name}")

def main():
    PROMPT = "Provide one factual sentence describing its visual attributes, including color, geometry, relative scale, and any unique features visible."
    SYSTEM = "You are an assistant that generates concise, factual object captions. Do not mention 'image' or 'scene'."

    parser = argparse.ArgumentParser(description="Smart VLM Captioner")
    parser.add_argument("--jsonl_path", type=str, help="Path to the JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Pretrained model ID")
    args = parser.parse_args()

    worker = UniversalCaptioner(model_id=args.model, PROMPT=PROMPT, SYSTEM=SYSTEM, device="auto")
    worker.process(args.jsonl_path)

if __name__ == "__main__":
    main()
