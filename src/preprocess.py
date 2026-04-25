# src/preprocess.py
# Preprocessing pipeline for Deep Learning CNN inference and training.

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ==============================
# IMAGE SIZE CONSTANTS
# ==============================

IMG_SIZE = 128  # Standard input size for the CNN


# ==============================
# TRANSFORMS (for PyTorch tensors)
# ==============================

def get_train_transforms():
    """Augmentation + normalization transforms used during training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
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