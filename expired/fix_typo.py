import json
from tqdm import tqdm

jsonl_path = "data/nuscenes_triplet_val.jsonl"
fixed_lines = []

with open(jsonl_path, "r") as f:
    for line in tqdm(f, desc="Fixing image paths"):
        obj = json.loads(line)
        obj["image_path"] = obj["image_path"].replace("nuscenses_images", "nuscenes_images")
        fixed_lines.append(json.dumps(obj))

with open(jsonl_path, "w") as f:
    for line in tqdm(fixed_lines, desc="Writing fixed JSONL"):
        f.write(line + "\n")
