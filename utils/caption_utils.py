import torch
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from collections import Counter

def describe_camera_annotations(nusc, sample_record, camera_name, box_vis_level=1):
    cam_token = sample_record["data"][camera_name]
    _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=box_vis_level)

    if len(boxes) == 0:
        description = "No annotated objects are visible in this camera view."
        return description

    # find categories and counts
    categories = [
        nusc.get("sample_annotation", box.token)["category_name"].split(".")[-1]
        for box in boxes
    ]
    counter = Counter(categories)

    # build description
    parts = [f"{count} {cat}{'s' if count > 1 else ''}" for cat, count in counter.items()]
    if len(parts) == 1:
        description = f"This scene includes {parts[0]}."
    elif len(parts) == 2:
        description = f"This scene includes {parts[0]} and {parts[1]}."
    else:
        description = f"This scene includes {', '.join(parts[:-1])}, and {parts[-1]}."

    return description

def lidar2camera_fov(nusc, points_ego, token, camera_name):
    sample_record = nusc.get('sample', token)
    cam_token = nusc.get('sample_data', sample_record['data'][camera_name])
    cam_calib = nusc.get('calibrated_sensor', cam_token['calibrated_sensor_token'])

    # Camera extrinsics (ego -> camera)
    q = Quaternion(cam_calib['rotation'])
    t = torch.tensor(cam_calib['translation'], dtype=torch.float32)  # [3]
    R = torch.tensor(q.rotation_matrix, dtype=torch.float32)         # [3,3]
    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

    pc_cam = R.T @ (points_ego[:, :3].T - t.view(3, 1))  # [3, N]
    pc_cam_np = pc_cam.detach().cpu().numpy()  # to numpy

    points_img = view_points(pc_cam_np, cam_intrinsic, normalize=True)

    W, H = (1600, 900) # image size for nuScenes camera
    mask = (points_img[0, :] > 0) & (points_img[0, :] < W) & \
           (points_img[1, :] > 0) & (points_img[1, :] < H) & \
           (pc_cam_np[2, :] > 0)

    visible_points = points_ego[mask]
    return visible_points, mask

def caption_generate(describe, prompt, system, image, processor, model):
    text = (
        f"{describe}\n"
        f"{prompt}"
    )

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=77)

    caption = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return caption
