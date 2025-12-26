from nuscenes.utils.geometry_utils import view_points
from PIL import Image

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py, render_annotation
def crop_annotation(nusc, ann_token, sample_record, margin=5, min_ratio=0.8):
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1, selected_anntokens=[ann_token])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    if len(boxes) == 0:
        return None  # skip if not visible

    cam_token = sample_record['data'][cam]

    # Plot CAMERA view.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
    im = Image.open(data_path)

    # Crop the box from the image
    box = boxes[0]
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    x_min, y_min = corners.min(axis=1)
    x_max, y_max = corners.max(axis=1)
    
    # Calculate the area inside the image, if too small, skip this box
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), im.width)
    y_max_clip = min(int(y_max), im.height)

    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    if ratio_inside < min_ratio:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min) - margin, 0)
    y_min_final = max(int(y_min) - margin, 0)
    x_max_final = min(int(x_max) + margin, im.width)
    y_max_final = min(int(y_max) + margin, im.height)

    cropped_im = im.crop((x_min_final, y_min_final, x_max_final, y_max_final)) 
    
    return cropped_im

