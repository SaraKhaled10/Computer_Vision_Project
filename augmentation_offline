# augmentation_b0.py
#
# Offline augmentation ONLY for EfficientNet-B0.
# - Input : Data/Fish_Dataset/<class>/*.jpg
# - Output: Data/Fish_Dataset_B0_Augmented/<class>/*.jpg
#
# For each class, we keep the SAME number of images as the original:
#   ~50% images are augmented versions
#   ~50% images are original copies
#
# This way, total dataset size stays similar (~9000 images),
# but with more variation for B0 training.

import os
import shutil
from PIL import Image
from torchvision import transforms

# ----- PATHS -----
RAW_DIR = r"Data/Fish_Dataset"               # original dataset
AUG_DIR = r"Data/Fish_Dataset_B0_Augmented"  # augmented dataset for B0


# ----- AUGMENTATION PIPELINE (OFFLINE) -----
# This is applied only when we decide to augment an image.
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02,
    ),
])


def is_image_file(fname: str) -> bool:
    """Check if a file is an image by extension."""
    fname = fname.lower()
    return fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")


def main():
    print(f"Raw dataset root: {RAW_DIR}")
    print(f"Augmented B0 dataset root: {AUG_DIR}")

    # Create root output dir (if not exists)
    os.makedirs(AUG_DIR, exist_ok=True)

    # Loop over each class folder
    for class_name in sorted(os.listdir(RAW_DIR)):
        in_class_dir = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(in_class_dir):
            continue  # skip non-folder files

        out_class_dir = os.path.join(AUG_DIR, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        print(f"\nProcessing class: {class_name}")

        # List all image files in this class
        image_files = [f for f in sorted(os.listdir(in_class_dir)) if is_image_file(f)]
        num_images = len(image_files)
        if num_images == 0:
            print("  No images found, skipping.")
            continue

        # We'll augment the first half, copy the second half
        half = num_images // 2
        print(f"  Found {num_images} images. "
              f"Augmenting ~{half}, copying ~{num_images - half}.")

        count_written = 0

        for idx, fname in enumerate(image_files):
            src_path = os.path.join(in_class_dir, fname)
            dst_path = os.path.join(out_class_dir, fname)

            # Load image as PIL
            img = Image.open(src_path).convert("RGB")

            if idx < half:
                # ----- AUGMENTED VERSION -----
                aug_img = augment_transform(img)
                aug_img.save(dst_path)
            else:
                # ----- ORIGINAL COPY -----
                shutil.copy2(src_path, dst_path)

            count_written += 1

        print(f"  Done class: {class_name} | Saved {count_written} images.")

    print("\nOffline augmentation for B0 finished.")
    print(f"Final augmented dataset is in: {AUG_DIR}")


if __name__ == "__main__":
    main()
