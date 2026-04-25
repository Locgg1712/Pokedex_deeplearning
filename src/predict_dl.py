# src/predict_dl.py
# Inference pipeline — loads the trained CNN and predicts Pokémon from an image.

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image

from src.model_dl import build_model
from src.preprocess import get_val_transforms

# ==============================
# SINGLETON MODEL LOADER
# ==============================

_model = None
_class_names = None
_device = None
_transform = None


def _load_model(model_dir="Model"):
    """Lazy-load the model and metadata once."""
    global _model, _class_names, _device, _transform

    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    _class_names = metadata["class_names"]
    num_classes  = metadata["num_classes"]

    # Build model and load weights
    _model = build_model(num_classes=num_classes, device=_device)
    weights_path = os.path.join(model_dir, "best_model.pth")
    _model.load_state_dict(torch.load(weights_path, map_location=_device))
    _model.eval()

    _transform = get_val_transforms()

    print(f"[predict_dl] Model loaded on {_device} — {num_classes} classes")


# ==============================
# PREDICT FUNCTION
# ==============================

def predict(image_path, model_dir="Model"):
    """
    Predict the Pokémon class for a given image.

    Returns:
        predicted_name (str): Pokémon name.
        confidence (float): Probability of the top prediction (0–1).
        top3 (list of tuples): Top-3 predictions as [(name, probability), ...].
    """
    _load_model(model_dir)

    # Load and preprocess image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return "Không đọc được ảnh", 0.0, []

    input_tensor = _transform(img).unsqueeze(0).to(_device)

    # Inference
    with torch.no_grad():
        logits = _model(input_tensor)
        probs  = F.softmax(logits, dim=1)[0]

    # Top-3 predictions
    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(_class_names)))
    top3 = [
        (_class_names[idx.item()], prob.item())
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    predicted_name = top3[0][0]
    confidence     = top3[0][1]

    return predicted_name, confidence, top3


if __name__ == "__main__":
    path = input("Ảnh: ")
    name, conf, top3 = predict(path)
    print(f"\n==== KẾT QUẢ ====")
    print(f"  {name} — {conf*100:.1f}%")
    print(f"\n  Top-3:")
    for n, p in top3:
        print(f"    {n}: {p*100:.1f}%")
