from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from tqdm import tqdm
import json, os, torch
import tarfile
from io import BytesIO
from pathlib import Path
from PIL import Image

from utils.caption_utils import caption_generate
from utils.processing import resize_with_aspect_ratio

prompt = "Provide one factual sentence describing its visual attributes, including color, geometry, relative scale, and any unique features visible."
system = "You are an assistant that generates concise, factual object captions. Do not mention 'image' or 'scene'."

# Automatically locate tar based on jsonl path
def get_tar_for_jsonl(jsonl_path):
    data_dir = Path(jsonl_path).parent
    tar_path = data_dir / "nuscenes_images.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"Cannot find TAR archive at {tar_path}")
    print(f"[INFO] Using TAR archive: {tar_path}")
    return tarfile.open(tar_path, "r")

# Read image from TAR directly
def read_image_from_tar(tar, rel_path):
    # Remove possible prefix like "data/nuscenes_images/"
    rel_path = rel_path.replace("data/nuscenes_images/", "")
    try:
        member = tar.getmember(rel_path)
    except KeyError:
        raise FileNotFoundError(f"Image {rel_path} not found in tar archive")
    file = tar.extractfile(member)
    return Image.open(BytesIO(file.read())).convert("RGB")

def generate_triplet_captions_inplace(
    jsonl_path,
    processor,
    model,
    skip_existing=True,
    save_every=100,
):
    # load existing records
    with open(jsonl_path, "r") as f:
        records = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(records)} records from {jsonl_path}")

    # check whether a tmp file exists
    tmp_path = jsonl_path + ".tmp"
    processed = 0
    if os.path.exists(tmp_path):
        with open(tmp_path, "r") as f:
            existing = [json.loads(line.strip()) for line in f]
        processed = len(existing)
        print(f"Resuming from checkpoint: already processed {processed} samples.")
    else:
        existing = []

    updated = 0
    processed_records = []

    # tar = get_tar_for_jsonl(jsonl_path)

    for i, rec in enumerate(tqdm(records[processed:], desc="Generating Captions", initial=processed, total=len(records))):
        if skip_existing and rec.get("caption", "").strip():
            processed_records.append(rec)
            continue

        label = rec["label"]
        image_path = rec["image_path"]
        image = Image.open(image_path).convert("RGB")
        # image = read_image_from_tar(tar, rec["image_path"])

        describe = f"This image shows a {label} object in an autonomous driving scene."
        resized_image = resize_with_aspect_ratio(image, 256)
        
        caption = caption_generate(describe, prompt, system, resized_image, processor, model)
        if "assistant" in caption:
            caption = caption.split("assistant")[-1].strip()
        if "\n" in caption:
            caption = caption.split("\n")[0].strip()
        rec["caption"] = caption
        updated += 1

        processed_records.append(rec)

        # Write to temp file periodically
        if (i + 1) % save_every == 0:
            with open(tmp_path, "a") as f:
                for r in processed_records:
                    f.write(json.dumps(r) + "\n")
            processed_records.clear()
            print(f"Saved {i+1}/{len(records)} samples to temp file.")

    # Write remaining records
    if processed_records:
        with open(tmp_path, "a") as f:
            for r in processed_records:
                f.write(json.dumps(r) + "\n")

     # Replace original file with updated temp file
    os.replace(tmp_path, jsonl_path)
    print(f"Finished! Updated {updated} captions in {jsonl_path}")

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

generate_triplet_captions_inplace(
    jsonl_path="/home/ximeng/Documents/SparseCLIP/data/nuscenes_triplet_val.jsonl",
    processor=processor,
    model=model,
    skip_existing=True,   # skip existing captions
)
