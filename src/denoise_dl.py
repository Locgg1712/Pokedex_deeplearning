# src/denoise_dl.py
# U-Net Deep Learning Autoencoder for Image Denoising

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import numpy as np
import random

from src.dataset import PokemonDataset

IMG_SIZE = 128

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetDenoise(nn.Module):
    """
    A U-Net architecture for optimal image denoising.
    Skip connections preserve sharp edges while filtering noise.
    """
    def __init__(self):
        super().__init__()
        
        # Lighter Encoder
        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2) # 128 -> 64
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2) # 64 -> 32
        
        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)  # 32 + 32 = 64
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)   # 16 + 16 = 32
        
        # Final output
        self.out_conv = nn.Conv2d(16, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        d2 = self.upconv2(b)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return self.sigmoid(out)

def add_noise_to_tensor(img_tensor):
    """
    Randomly injects Gaussian and/or Salt & Pepper noise.
    Input: Tensor (C, H, W) in [0, 1]
    Output: Noisy Tensor
    """
    noisy_img = img_tensor.clone()
    
    # 1. Random Gaussian Noise
    if random.random() < 0.8:
        noise_factor = random.uniform(0.1, 0.3)
        noise = torch.randn_like(noisy_img) * noise_factor
        noisy_img = noisy_img + noise
        
    # 2. Random Salt & Pepper Noise
    if random.random() < 0.5:
        prob = random.uniform(0.02, 0.08)
        mask_salt = torch.rand_like(noisy_img[0:1, ...]) < (prob / 2)
        mask_pepper = torch.rand_like(noisy_img[0:1, ...]) < (prob / 2)
        noisy_img[:, mask_salt.squeeze()] = 1.0
        noisy_img[:, mask_pepper.squeeze()] = 0.0

    return torch.clamp(noisy_img, 0., 1.)

def get_ae_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor() # Converts to [0, 1]
    ])

class AutoencoderDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        clean_img = self.transform(image)
        # Apply random noise augmentation on the fly
        noisy_img = add_noise_to_tensor(clean_img)
        return noisy_img, clean_img

def train_unet_denoise(data_dir="data", model_dir="Model", epochs=15, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNet Denoise] Device: {device}", flush=True)

    full_dataset = PokemonDataset(data_dir, transform=None)
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    ae_transform = get_ae_transform()
    train_set = AutoencoderDataset(train_subset, ae_transform)
    val_set   = AutoencoderDataset(val_subset,   ae_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model = UNetDenoise().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"[UNet Denoise] Train: {len(train_set)} | Val: {len(val_set)}", flush=True)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader, 1):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
            
            if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                print(f"  [UNet Epoch {epoch:02d}] batch {batch_idx}/{len(train_loader)}  loss={loss.item():.4f}", flush=True)

        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item() * noisy_imgs.size(0)

        val_loss /= len(val_set)
        
        scheduler.step(val_loss)

        print(f"UNet Epoch {epoch:02d}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "denoise_model.pth"))

    print(f"\n[DONE] U-Net Denoising Training Complete! Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"   Model saved to {model_dir}/denoise_model.pth", flush=True)

if __name__ == "__main__":
    train_unet_denoise()
