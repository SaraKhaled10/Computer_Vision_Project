# Computer_Vision_Project

This project implements a full deep-learning pipeline for fish species classification using EfficientNet (B0, B1, B2) and ResNet-18. The repository includes custom preprocessing, offline data augmentation, and a clean train/validation/test splitting procedure to ensure proper evaluation without data leakage. The workflow begins with downloading the Kaggle fish dataset into Data/Fish_Dataset, optionally generating an expanded dataset using augmentation.py, and training EfficientNet models on both the original and augmented sets using main.py and main_b0_aug.py. Model checkpoints and metrics are saved automatically, and evaluation is performed through eval_ckpt_main.py, which outputs accuracy, precision, recall, and F1-scores. For a clean baseline, prepare_split.py creates a structured, leakage-free dataset under Data/Fish_Split, which is used by Resnet18.py to train and evaluate a strong ResNet-18 model. The project structure includes architecture utilities, preprocessing scripts, model builders, training pipelines, and evaluation scripts, providing a complete framework for comparing EfficientNet variants against a traditional CNN baseline. The results consistently show that EfficientNet models achieve strong performance (especially with augmentation), while ResNet-18 delivers the highest accuracy under the clean split setup. This repository demonstrates dataset handling, augmentation, model training, and evaluation best practices in PyTorch for image classification research.

# Fish Classification Project

This repository contains PyTorch implementations of **EfficientNet** and **ResNet-18** models applied to a fish image dataset. It includes data preprocessing, dataset splitting, training, evaluation, and utility scripts for model building and inference.

---

## Project Structure

### Dataset Preparation
- **prepare_split.py**  
  Creates a reproducible train/validation/test split of the original fish dataset in a 70/15/15 ratio. Only image files are included, and class metadata is preserved. Output directories:  
  `Data/Fish_Split/train|val|test/<class>/`.

- **augment.py**  
  Copies the original dataset into a clean folder without offline augmentations, ensuring unbiased training. All augmentations are applied later during model training.

---

### Preprocessing
- **preprocessing.py**  
  Defines image transformations for EfficientNet models:
  - Resizing to model input resolution
  - Normalization using ImageNet mean and standard deviation
  - Optional data augmentation (flip, rotation, color jitter) during training
  - Includes `preprocess_image` to convert a PIL image to a tensor ready for model input

---

### Model Implementations
- **EfficientnetModel.py**  
  PyTorch implementation of EfficientNet (B0–B8):
  - MBConv blocks with optional squeeze-and-excitation
  - Swish activation
  - Drop connect for stochastic depth
  - Dynamically scales width and depth according to global coefficients
  - Constructs the stem, repeated MBConv blocks, and head

- **resnet18.py**  
  Baseline ResNet-18 classifier on the clean split:
  - Online augmentation only on training set
  - Trains for a few epochs
  - Evaluates accuracy, precision, recall, and F1-score
  - Saves results to CSV

- **efficientnet_builder.py**  
  Converts TensorFlow EfficientNet builder to PyTorch:
  - Defines model scaling parameters for each variant
  - `BlockDecoder` converts string-encoded MBConv blocks to structured arguments
  - `get_model_params` returns block arguments and global params
  - `efficientnet()` constructs a PyTorch EfficientNet instance

- **utils.py**  
  PyTorch-adapted utility functions for EfficientNet:
  - Wrappers for Conv2d, DepthwiseConv2d, BatchNorm2d
  - Learning rate schedulers and optimizer builders
  - Weight-loading utilities for `.pth` files
  - Helpers for padding and optional activation functions

---

### Training & Evaluation
- **main.py**  
  Trains and evaluates multiple EfficientNet variants on the original dataset:
  - Applies preprocessing and splits dataset
  - Training loops with optional learning rate scheduling
  - Saves checkpoints and CSV summaries

- **Main_b0_aug.py**  
  Trains EfficientNet-B0 on the offline-augmented dataset:
  - Full training loop with validation
  - Evaluates on test set
  - Saves best model and logs results

- **eval_ckpt_example.ipynb**  
  Demonstrates single-image inference:
  - Preprocessing, model loading, and prediction
  - Prints predicted class and probability

- **eval_ckpt_main.py**  
  Full evaluation of EfficientNet models on the test set:
  - Loads clean dataset and applies preprocessing
  - Loads pretrained weights
  - Computes accuracy, precision, recall, F1-score
  - Prints detailed classification report

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- PIL / Pillow

---

## Usage

1. **Prepare the dataset:**
```bash
python prepare_split.py

