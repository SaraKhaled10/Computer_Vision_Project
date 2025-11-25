# Resnet18.py
#
# ResNet-18 baseline on a CLEAN split:
# Data/Fish_Split/train, val, test
# Augmentation ONLY on train (online), no leakage.

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Paths
# =========================
SPLIT_ROOT = r"Data/Fish_Split"
train_dir = os.path.join(SPLIT_ROOT, "train")
val_dir   = os.path.join(SPLIT_ROOT, "val")
test_dir  = os.path.join(SPLIT_ROOT, "test")

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Datasets & Loaders
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

# =========================
# Model
# =========================
model = models.resnet18(pretrained=True)  # warnings are ok
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# =========================
# Training
# =========================
num_epochs = 2   # small number just for baseline
best_val_acc = 0.0
best_model_path = "checkpoints/resnet18_split_best.pth"
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    elapsed = int(time.time() - start_time)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {elapsed}s")

    # validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    print(f"Validation accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Saved best model to", best_model_path)

# =========================
# Test evaluation
# =========================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n=== ResNet-18 on CLEAN SPLIT (no leakage) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# save CSV
csv_path = "resnet18_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["accuracy", "precision", "recall", "f1_score"])
    writer.writeheader()
    writer.writerow({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

print("Results saved to", csv_path)
