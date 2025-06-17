import os
import cv2
import random
from tqdm import tqdm
from albumentations import (
    Compose, HueSaturationValue, RGBShift, RandomBrightnessContrast,
    ChannelShuffle, ElasticTransform, ShiftScaleRotate, HorizontalFlip
)

# Folder dataset awal
dataset_dir = 'dataset'
# Folder hasil augmentasi
output_dir = 'aug_dataset'
# Target gambar per kelas
target_per_class = 100  # Sesuai permintaan kamu, jadi 100 per kelas

# Augmentasi gabungan
augmentation = Compose([
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ChannelShuffle(p=0.5),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=0, p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    HorizontalFlip(p=0.5)
])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop setiap class
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class {class_name}...")
    image_paths = [os.path.join(class_path, fname) for fname in os.listdir(class_path) if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(image_paths) == 0:
        print(f"Warning: No images found in {class_name}")
        continue

    save_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    image_count = 0
    # Salin gambar asli ke folder baru
    for img_path in tqdm(image_paths, desc=f"Copying originals for {class_name}"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to copy {img_path} (file corrupted or unreadable)")
            continue
        save_path = os.path.join(save_class_dir, f"{image_count + 1}.jpg")
        cv2.imwrite(save_path, img)
        image_count += 1

        if image_count >= target_per_class:
            break  # Kalau sudah cukup, berhenti salin gambar

    print(f"Copied {image_count} original images for class {class_name}.")

    # Augmentasi sampai mencapai target_per_class jika gambar asli kurang dari 100
    while image_count < target_per_class:
        img_path = random.choice(image_paths)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read {img_path} (file corrupted or unreadable)")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = augmentation(image=img)
        aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

        image_count += 1
        save_path = os.path.join(save_class_dir, f"{image_count}.jpg")
        cv2.imwrite(save_path, aug_img)

    print(f"Completed class {class_name} -> Total images: {image_count}")
