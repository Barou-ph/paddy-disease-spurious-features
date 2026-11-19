from __future__ import annotations

from torchvision import transforms


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    horizontal_flip: bool = True,
    rotation_degrees: float = 10.0,
):
    """ImageNet-compatible preprocessing for pretrained torchvision models."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        ops = [transforms.Resize((image_size, image_size))]
        if horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())
        if rotation_degrees > 0:
            ops.append(transforms.RandomRotation(rotation_degrees))
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
