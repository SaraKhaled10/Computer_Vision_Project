# prepare_split.py
#
# Create a clean train/val/test split from the ORIGINAL dataset.
# Input:  Data/Fish_Dataset/<class>/*.jpg
# Output: Data/Fish_Split/train|val|test/<class>/*.jpg

import os
import shutil
import random

RAW_DIR = r"Data/Fish_Dataset"
SPLIT_ROOT = r"Data/Fish_Split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # remaining

def is_image_file(fname: str) -> bool:
    fname = fname.lower()
    return fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")

def main():
    random.seed(42)
    print(f"Reading original dataset from: {RAW_DIR}")
    print(f"Writing split dataset to: {SPLIT_ROOT}")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SPLIT_ROOT, split), exist_ok=True)

    for class_name in sorted(os.listdir(RAW_DIR)):
        class_dir = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\nProcessing class: {class_name}")
        images = [f for f in os.listdir(class_dir) if is_image_file(f)]
        images.sort()
        random.shuffle(images)

        n = len(images)
        n_train = int(TRAIN_RATIO * n)
        n_val   = int(VAL_RATIO * n)
        n_test  = n - n_train - n_val

        train_files = images[:n_train]
        val_files   = images[n_train:n_train + n_val]
        test_files  = images[n_train + n_val:]

        print(f"  total: {n}, train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

        # create class subfolders and copy files
        for split, file_list in [("train", train_files),
                                 ("val",   val_files),
                                 ("test",  test_files)]:
            out_class_dir = os.path.join(SPLIT_ROOT, split, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            for fname in file_list:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(out_class_dir, fname)
                shutil.copy2(src, dst)

    print("\nDone. Clean split created under:", SPLIT_ROOT)

if __name__ == "__main__":
    main()
