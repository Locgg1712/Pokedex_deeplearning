import os
import cv2
import numpy as np


# --- Noise Functions ---

def gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_pepper_noise(img, prob=0.02):
    noisy = img.copy()
    h, w, c = img.shape

    # salt
    num_salt = int(prob * h * w / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    noisy[coords[0], coords[1]] = 255

    # pepper
    num_pepper = int(prob * h * w / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1]] = 0

    return noisy


def speckle_noise(img):
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise * 0.2
    return np.clip(noisy, 0, 255).astype(np.uint8)


# --- Main ---

img_path = "test/squirtle 2.png"   # ảnh của bạn
img = cv2.imread(img_path)

if img is None:
    print("Không đọc được ảnh!")
    exit()

img = cv2.resize(img, (128, 128))

# tạo nhiễu
g = gaussian_noise(img)
sp = salt_pepper_noise(img)
speckle = speckle_noise(img)

# lưu
output_dir = "Test"
os.makedirs(output_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(img_path))[0]

cv2.imwrite(os.path.join(output_dir, f"{base_name}_gaussian.jpg"), g)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_saltpepper.jpg"), sp)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_speckle.jpg"), speckle)

print(f"Done! Đã lưu ảnh nhiễu của {base_name} vào thư mục Test.")