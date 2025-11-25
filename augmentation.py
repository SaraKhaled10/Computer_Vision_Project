# augmentation.py
#
# Create a clean copy of the dataset (NO EXTRA AUGMENTATION)
# This keeps the sample size the same (~9000 images)

import os
import shutil

RAW_DIR = r"Data/Fish_Dataset"
AUG_DIR = r"Data/Fish_Dataset_Augmented"

def is_image_file(fname: str) -> bool:
    fname = fname.lower()
    return fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")

def main():
    print(f"Raw dataset root: {RAW_DIR}")
    print(f"New dataset root (no augmentation): {AUG_DIR}")

    # create output dir
    os.makedirs(AUG_DIR, exist_ok=True)

    # loop through class folders
    for class_name in sorted(os.listdir(RAW_DIR)):
        in_class_dir = os.path.join(RAW_DIR, class_name)

        if not os.path.isdir(in_class_dir):
            continue

        out_class_dir = os.path.join(AUG_DIR, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        print(f"\nCopying class: {class_name}")

        for fname in sorted(os.listdir(in_class_dir)):
            if not is_image_file(fname):
                continue

            src = os.path.join(in_class_dir, fname)
            dst = os.path.join(out_class_dir, fname)

            # copy the original image ONLY (no augmented images)
            shutil.copy2(src, dst)

        print(f"  Done copying class: {class_name}")

    print("\n Dataset copy finished.")
    print(f"Dataset (≈9000 images) created at: {AUG_DIR}")

if __name__ == "__main__":
    main()
