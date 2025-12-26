import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def resize_with_aspect_ratio(img, size=256):
    """Resize image while maintaining aspect ratio and then center crop to (size, size).
    Args:
        img: PIL Image, numpy array, or torch tensor
        size: Desired output size (size x size)
    Returns:
        Resized and center-cropped PIL Image
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        img = Image.fromarray(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    new_img = Image.new("RGB", (size, size))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

def resize_long_edge(image, long_edge=256):
    """
    Resize image so that the longer edge matches long_edge, maintaining aspect ratio.
    Args:
        image: PIL Image or numpy array
        long_edge: Desired size of the longer edge
    Returns:
        Resized PIL Image
    """
    # Convert numpy → PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    if w >= h:
        new_w = long_edge
        new_h = int(h * long_edge / w)
    else:
        new_h = long_edge
        new_w = int(w * long_edge / h)

    # Use high-quality downsampling
    image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

image_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_aspect_ratio(img, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP mean
        std=(0.26862954, 0.26130258, 0.27577711)   # CLIP std
    )
])

def inverse_image_transform(img_tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)

    return transforms.ToPILImage()(img)
