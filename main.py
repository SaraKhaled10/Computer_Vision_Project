# main.py
#
# Training EfficientNet on the Fish_Dataset
# - uses torchvision ImageFolder + random_split
# - supports multiple EfficientNet variants
# - integrates preprocessing.py for transforms
# - logs test accuracy per model and picks best

import os
import time
import csv

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

# import model builder and scaling params
from ModelBuilder import efficientnet, _MODEL_PARAMS
# import preprocessing pipeline
from preprocessing import preprocess_image
# import utils for optimizer and lr schedule (if needed)
from utils import build_optimizer, build_learning_rate, load_model_weights


# ============================================================
# Device configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


# ============================================================
# Dataset setup
# ============================================================

# root folder that contains class subfolders
# make sure this path matches your project structure
data_dir = r"Data/Fish_Dataset"

# load full dataset with PIL loader; transforms are applied later
full_dataset = datasets.ImageFolder(
    data_dir,
    loader=lambda path: datasets.folder.default_loader(path)
)

# compute split sizes for 70-15-15
num_samples = len(full_dataset)
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

# split into train/val/test with reproducibility
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

# build dataloaders; transforms are set inside train_and_eval for each model
# num_workers=0 to avoid Windows multiprocessing issues
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False,
    num_workers=0, pin_memory=False
)

# get class names and count
class_names = full_dataset.classes
num_classes = len(class_names)
print("classes:", class_names)


# ============================================================
# Results tracking
# ============================================================

results = []  # list of dicts per model variant
best_model = {"name": None, "test_acc": 0.0, "params": None}


# ============================================================
# Train / Eval function for one EfficientNet variant
# ============================================================

def train_and_eval(model_name, num_epochs=5, lr=1e-4,
                   optimizer_name="adam", weight_path=None):
    print(f"\n=== training {model_name} ===")

    # read scaling parameters from ModelBuilder
    width, depth, resolution, dropout = _MODEL_PARAMS[model_name]
    print(f"config -> width: {width}, depth: {depth}, "
          f"resolution: {resolution}, dropout: {dropout}")

    # assign preprocessing functions that respect the model's input size
    # note: train_dataset, val_dataset, test_dataset are Subset objects
    train_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=True
    )
    val_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=False
    )
    test_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=False
    )

    # build EfficientNet model for this variant
    model = efficientnet(model_name=model_name)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # optional: load pretrained weights for fine-tuning
    if weight_path is not None:
        model = load_model_weights(model, weight_path, device=device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=1e-4,
    )

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # ------------------------ Training loop ------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # small progress print so it doesn't look stuck
            if batch_idx % 5 == 0:
                print(f"  processing batch {batch_idx + 1}/{len(train_loader)}")

            images, labels = images.to(device), labels.to(device)

            # optional learning rate schedule
            global_step = epoch * steps_per_epoch + batch_idx
            current_lr = build_learning_rate(
                initial_lr=lr,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                lr_decay_type="cosine",
                total_steps=total_steps,
                warmup_epochs=0,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        elapsed = int(time.time() - start_time)
        print(
            f"epoch {epoch + 1}/{num_epochs} | "
            f"loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time: {elapsed}s"
        )

        # ------------------------ Validation loop ------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"validation accuracy: {val_acc:.4f}")

    # ------------------------ Final test evaluation ------------------------
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"final test accuracy ({model_name}): {test_acc:.4f}")

    # save checkpoint
    checkpoint_path = os.path.join("checkpoints", f"{model_name}_final.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"model saved to {checkpoint_path}")

    # log results
    results.append({
        "model": model_name,
        "width": width,
        "depth": depth,
        "resolution": resolution,
        "dropout": dropout,
        "optimizer": optimizer_name,
        "initial_lr": lr,
        "test_acc": test_acc,
    })

    # track best model
    global best_model
    if test_acc > best_model["test_acc"]:
        best_model = {
            "name": model_name,
            "test_acc": test_acc,
            "params": (width, depth, resolution, dropout),
            "optimizer": optimizer_name,
            "initial_lr": lr,
        }


# ============================================================
# Main: run several EfficientNet variants & save summary
# ============================================================

if __name__ == "__main__":
    # CHOOSE WHICH MODELS TO TRAIN
    # For the project, using b0–b4 is enough. You can reduce this list if training is too slow.
    model_list = [
        "efficientnet-b3",
        "efficientnet-b4",
    ]

    # Set how many epochs you want (increase for better accuracy)
    num_epochs = 5

    for model_name in model_list:
        train_and_eval(
            model_name=model_name,
            num_epochs=num_epochs,
            lr=1e-4,
            optimizer_name="adam",
            weight_path=None,
        )

    # Save results to CSV
    csv_path = "results_summary.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "width",
                "depth",
                "resolution",
                "dropout",
                "optimizer",
                "initial_lr",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nresults saved to {csv_path}")

    # Print best configuration
    print("\n=== best model configuration ===")
    print(f"model: {best_model['name']}")
    print(f"test accuracy: {best_model['test_acc']:.4f}")
    w, d, r, dr = best_model["params"]
    print(f"width: {w}, depth: {d}, resolution: {r}, dropout: {dr}")
    print(f"optimizer: {best_model['optimizer']} | initial lr: {best_model['initial_lr']}")
