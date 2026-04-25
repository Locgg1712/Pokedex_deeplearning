# src/train_dl.py
# Deep Learning training pipeline for the Pokémon CNN classifier.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
import json

from src.dataset import PokemonDataset
from src.model_dl import build_model
from src.preprocess import get_train_transforms, get_val_transforms


def train(data_dir="data", model_dir="Model", epochs=15, batch_size=32, lr=0.001):
    """Full training pipeline: load data → train → evaluate → save."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ==============================
    # 1. DATASET & DATALOADERS
    # ==============================

    full_dataset = PokemonDataset(data_dir, transform=None)
    class_names = full_dataset.class_names
    num_classes = len(class_names)

    print(f"\nFound {len(full_dataset)} images across {num_classes} classes:", flush=True)
    for name in class_names:
        print(f"  • {name}", flush=True)

    # 80/20 split
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms via wrapper datasets
    train_set = TransformSubset(train_subset, get_train_transforms())
    val_set   = TransformSubset(val_subset,   get_val_transforms())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain: {len(train_set)} | Val: {len(val_set)}", flush=True)

    # ==============================
    # 2. MODEL, LOSS, OPTIMIZER
    # ==============================

    model = build_model(num_classes=num_classes, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # ==============================
    # 3. TRAINING LOOP
    # ==============================

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total_samples = 0.0, 0, 0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 5 == 0 or batch_idx == num_batches:
                print(f"  [Epoch {epoch:02d}] batch {batch_idx}/{num_batches}  loss={loss.item():.4f}", flush=True)

        train_loss = running_loss / total_samples
        train_acc  = correct / total_samples

        # --- Validate ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} — "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%", flush=True)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

    # ==============================
    # 4. SAVE METADATA & PLOT
    # ==============================

    os.makedirs(model_dir, exist_ok=True)

    # Save class names mapping
    metadata = {
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": 128,
        "best_val_acc": best_val_acc,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Training complete!  Best Val Accuracy: {best_val_acc*100:.1f}%", flush=True)
    print(f"   Model saved to {model_dir}/best_model.pth", flush=True)

    # Plot training curves
    plot_training_history(history, model_dir)


def plot_training_history(history, model_dir):
    """Plot and save loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]],   label="Val Acc")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"   Training curves saved to {model_dir}/training_curves.png", flush=True)


# ==============================
# HELPER: Apply transforms to Subset
# ==============================

class TransformSubset:
    """Wraps a torch Subset to apply a transform at __getitem__ time."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    train()
