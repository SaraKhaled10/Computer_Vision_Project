# eval_ckpt_main.py
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import classification_report, accuracy_score

from ModelBuilder import efficientnet
from preprocessing import preprocess_image
from utils import load_model_weights

# path must match your project
data_dir = r"Data/Fish_Dataset_Augmented"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def evaluate_model(model_name):
    print(f"\n=== Evaluating {model_name} ===")

    # load dataset
    full_dataset = datasets.ImageFolder(
        data_dir,
        loader=lambda path: datasets.folder.default_loader(path)
    )
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # same split as training
    num_samples = len(full_dataset)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    _, _, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # preprocess
    test_dataset.dataset.transform = lambda img: preprocess_image(
        img, model_name=model_name, is_training=False
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # build model
    model = efficientnet(model_name=model_name)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.to(device)

    # load weights
    weight_path = f"checkpoints/{model_name}_final.pth"
    model = load_model_weights(model, weight_path, device=device)

    # evaluate
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    # accuracy
    acc = accuracy_score(all_true, all_pred)

    # precision / recall / f1
    report = classification_report(
        all_true, all_pred,
        target_names=class_names,
        digits=4
    )

    print(f"\nAccuracy for {model_name}: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    return acc, report


if __name__ == "__main__":
    results = {}

    for m in ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2"]:
        acc, rep = evaluate_model(m)
        results[m] = {"accuracy": acc, "report": rep}

    print("\n=== FINAL SUMMARY ===")
    for m in results:
        print(f"\nModel: {m}")
        print(f"Accuracy: {results[m]['accuracy']:.4f}")
        print(results[m]['report'])
