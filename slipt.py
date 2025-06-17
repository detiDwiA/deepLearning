import os
import shutil
import random

# Folder dataset hasil augmentasi
source_dir = 'aug_dataset'
# Folder output
train_dir = 'splittt_dataset/train'
val_dir = 'splittt_dataset/val'

# Buat folder tujuan
for folder in [train_dir, val_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Split tiap class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Buat folder per class
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training images
    for img_name in train_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(train_class_dir, img_name)
        shutil.copyfile(src, dst)

    # Copy validation images
    for img_name in val_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(val_class_dir, img_name)
        shutil.copyfile(src, dst)

    print(f"Class {class_name} -> Training: {len(train_images)} | Validation: {len(val_images)}")

print("\nDataset splitting completed successfully!")
