
# preprocessing.py
# Central place for all image transforms used with EfficientNet

from torchvision import transforms
from ModelBuilder import _MODEL_PARAMS

def get_preprocessing_transforms(model_name="efficientnet-b0", is_training=True):
    """
    Build a torchvision transform pipeline that:
    - resizes to the correct EfficientNet resolution
    - applies data augmentation if training
    - converts to tensor
    - normalizes with ImageNet mean/std
    """
    # _MODEL_PARAMS: (width, depth, resolution, dropout)
    _, _, resolution, _ = _MODEL_PARAMS[model_name]

    base_transforms = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    if is_training:
        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
        return transforms.Compose(aug_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)


def preprocess_image(pil_image, model_name="efficientnet-b0", is_training=True):
    """
    Convenience wrapper used in main.py:
    takes a PIL image and returns a transformed tensor.
    """
    transform = get_preprocessing_transforms(model_name=model_name, is_training=is_training)
    return transform(pil_image)


