# src/preprocess.py
# Preprocessing pipeline for Deep Learning CNN inference and training.

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.denoise_dl import UNetDenoise

# ==============================
# IMAGE SIZE CONSTANTS
# ==============================

IMG_SIZE = 128  # Standard input size for the CNN


import random

# ==============================
# CUSTOM AUGMENTATIONS
# ==============================

class RandomOcclusion(object):
    """Randomly covers 10%-30% of the image with a noise patch to simulate occlusion/accessories."""
    def __init__(self, p=0.5, scale=(0.1, 0.3)):
        self.p = p
        self.scale = scale
        
    def __call__(self, tensor):
        if random.random() < self.p:
            c, h, w = tensor.size()
            area = h * w
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(0.5, 2.0)
            
            h_occ = int(round((target_area * aspect_ratio) ** 0.5))
            w_occ = int(round((target_area / aspect_ratio) ** 0.5))
            
            if w_occ < w and h_occ < h:
                y = random.randint(0, h - h_occ)
                x = random.randint(0, w - w_occ)
                
                # Fill with random noise patch to simulate complex occlusion
                noise_patch = torch.rand((c, h_occ, w_occ))
                tensor[:, y:y+h_occ, x:x+w_occ] = noise_patch
                
        return tensor

class AddGaussianNoise(object):
    """Adds Gaussian noise to the tensor image."""
    def __init__(self, p=0.5, std=0.05):
        self.p = p
        self.std = std
        
    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std
            return torch.clamp(tensor + noise, 0., 1.)
        return tensor

class AddSaltPepperNoise(object):
    """Adds Salt and Pepper noise to the tensor image."""
    def __init__(self, p=0.5, amount=0.04):
        self.p = p
        self.amount = amount
        
    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.rand(tensor.size())
            tensor[noise < (self.amount / 2)] = 1.0  # Salt
            tensor[noise > (1 - self.amount / 2)] = 0.0  # Pepper
        return tensor

# ==============================
# TRANSFORMS (for PyTorch tensors)
# ==============================

def get_train_transforms():
    """Robust augmentation + normalization transforms used during training."""
    return transforms.Compose([
        # 1. Resize
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        
        # 2. Geometric & Color Augmentations (PIL)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        
        # 3. Blur Augmentation (PIL)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        
        # 4. Tensor conversion
        transforms.ToTensor(),
        
        # 5. Tensor-based Augmentations (Occlusion, Erasing, Noise)
        RandomOcclusion(p=0.4, scale=(0.1, 0.3)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0), # Solid patch
        AddGaussianNoise(p=0.3, std=0.05),
        AddSaltPepperNoise(p=0.2, amount=0.04),
        
        # 6. Normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Normalization-only transforms used during validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ==============================
# RAW IMAGE LOADING (for GUI display)
# ==============================

def load_image_for_display(image_path, size=(256, 256)):
    """Load and resize an image for GUI display (returns BGR cv2 image)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.resize(img, size)


def load_image_pil(image_path):
    """Load an image as a PIL RGB Image (for model inference)."""
    img = Image.open(image_path).convert("RGB")
    return img

# ==============================
# DEEP LEARNING DENOISING
# ==============================

_denoise_model = None
_device = None
_ae_transform = None

def _load_denoise_model(model_dir="Model"):
    global _denoise_model, _device, _ae_transform
    if _denoise_model is not None:
        return
        
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _denoise_model = UNetDenoise().to(_device)
    
    weights_path = os.path.join(model_dir, "denoise_model.pth")
    if os.path.exists(weights_path):
        _denoise_model.load_state_dict(torch.load(weights_path, map_location=_device))
    else:
        print(f"[Warn] {weights_path} not found! Using untrained Autoencoder weights.")
        
    _denoise_model.eval()
    
    _ae_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

# ==============================
# TRADITIONAL CV HELPERS (Edges)
# ==============================

def auto_canny(image, sigma=0.33):
    """Compute Canny edges automatically based on median pixel intensity."""
    # Convert to grayscale if it isn't already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    median = np.median(gray)
    lower  = int(max(0,   (1.0 - sigma) * median))
    upper  = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)

def clean_edges(edges):
    """Clean up noise in edge detection using morphology."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# ==============================
# PIPELINE FOR GUI VISUALIZATION
# ==============================
import os

def extract_pokemon_debug(image_path):
    """
    Returns 4 stages of the preprocessing pipeline for UI display:
    1. Original (cv2 BGR)
    2. DL Denoised (cv2 BGR)
    3. Edges (cv2 BGR)
    4. Final (cv2 BGR)
    """
    _load_denoise_model()
    
    # 1. Original
    original = cv2.imread(image_path)
    if original is None:
        return None
        
    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    
    # 2. DL Denoise
    try:
        pil_img = Image.open(image_path).convert("RGB")
        input_tensor = _ae_transform(pil_img).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            output_tensor = _denoise_model(input_tensor)
            
        # Convert tensor [0, 1] back to cv2 BGR [0, 255]
        out_np = output_tensor.squeeze().cpu().numpy()
        out_np = np.transpose(out_np, (1, 2, 0)) # CHW -> HWC
        out_np = (out_np * 255).astype(np.uint8)
        denoised = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error in DL Denoise: {e}")
        denoised = original_resized.copy()
        
    # 3. Edges
    edges = auto_canny(denoised)
    edges = clean_edges(edges)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 4. Final (We just return the denoised version as final to feed into CNN)
    final = denoised.copy()
    
    return {
        "original": original_resized,
        "blur": denoised,      # "blur" name kept for backwards compatibility with UI logic
        "edges": edges_bgr,
        "combine": final       # "combine" name kept for backwards compatibility
    }