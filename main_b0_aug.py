# main_b0_aug.py
#
# Train EfficientNet-b0 on the OFFLINE-AUGMENTED dataset
# using the SAME preprocessing.py as the original project.
#
# Original main.py and preprocessing.py are NOT modified.

import os
import time
import csv
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

# import model builder and scaling params
from ModelBuilder import efficientnet, _MODEL_PARAMS
# import preprocessing pipeline (same file as before)
from preprocessing import preprocess_image
# import utils for optimizer, lr, and optional weight loading
from utils import build_optimizer, build_learning_rate, load_model_weights

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Dataset setup (AUGMENTED)
# -----------------------------
# Use the offline-augmented dataset you created with augmentation.py
data_dir = r"Data/Fish_Dataset_Augmented"   # <--- IMPORTANT CHANGE

# root with class subfolders
full_dataset = datasets.ImageFolder(
    data_dir,
    loader=lambda path: datasets.folder.default_loader(path)  # PIL loader
)

# split 70 / 15 / 15
num_samples = len(full_dataset)
train_size = int(0.7 * num_samples)
val_size   = int(0.15 * num_samples)
test_size  = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# initial dummy transforms (real ones set inside train_and_eval)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

# class info
class_names = full_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Num samples:", num_samples)
print("Train / Val / Test:", train_size, val_size, test_size)

# -----------------------------
# Results tracking
# -----------------------------
results = []
best_model = {"name": None, "test_acc": 0.0, "params": None}


# -----------------------------
# Train & Eval for ONE model (b0)
# -----------------------------
def train_and_eval_b0(num_epochs=5, lr=1e-4, optimizer_name="adam", weight_path=None):
    model_name = "efficientnet-b0"
    print(f"\n=== Training {model_name} on AUGMENTED dataset ===")

    # EfficientNet scaling params
    width, depth, resolution, dropout = _MODEL_PARAMS[model_name]
    print(f"Config -> width: {width}, depth: {depth}, resolution: {resolution}, dropout: {dropout}")

    # set preprocessing for this model
    train_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=True
    )
    val_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=False
    )
    test_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=False
    )

    # rebuild loaders (now they will use the new transforms)
    global train_loader, val_loader, test_loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

    # build model
    model = efficientnet(model_name=model_name)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # optional: load pretrained weights
    if weight_path is not None:
        model = load_model_weights(model, weight_path, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), optimizer_name=optimizer_name, lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            global_step = epoch * steps_per_epoch + batch_idx
            current_lr = build_learning_rate(
                initial_lr=lr,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                lr_decay_type="cosine",
                total_steps=total_steps,
                warmup_epochs=0
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
            total  += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc  = correct / total if total > 0 else 0.0
        elapsed    = int(time.time() - start_time)
        print(f"Epoch {epoch+1}/{num_epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time: {elapsed}s")

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total  += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"  Validation accuracy: {val_acc:.4f}")

    # final test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total  += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"\nFinal TEST accuracy ({model_name} on augmented dataset): {test_acc:.4f}")

    # save checkpoint (this overwrites the old b0 file)
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # log results
    results.append({
        "model": model_name,
        "width": width,
        "depth": depth,
        "resolution": resolution,
        "dropout": dropout,
        "optimizer": optimizer_name,
        "initial_lr": lr,
        "test_acc": test_acc
    })

    global best_model
    best_model = {
        "name": model_name,
        "test_acc": test_acc,
        "params": (width, depth, resolution, dropout),
        "optimizer": optimizer_name,
        "initial_lr": lr
    }


if __name__ == "__main__":
    # train only b0 on augmented dataset
    train_and_eval_b0(num_epochs=5, lr=1e-4, optimizer_name="adam", weight_path=None)

    # save results to CSV
    csv_path = "results_summary_b0_augmented.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "width", "depth", "resolution", "dropout",
                        "optimizer", "initial_lr", "test_acc"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # print best config (only b0 here, but kept for consistency)
    print("\n=== Best model configuration (augmented) ===")
    print(f"Model: {best_model['name']}")
    print(f"Test accuracy: {best_model['test_acc']:.4f}")
    if best_model["params"] is not None:
        w, d, r, dr = best_model["params"]
        print(f"Width: {w}, depth: {d}, resolution: {r}, dropout: {dr}")
    print(f"Optimizer: {best_model['optimizer']} | initial lr: {best_model['initial_lr']}")
