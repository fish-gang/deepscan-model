import random

import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet statistics (used for torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UnderwaterColorShift:
    """Simulates red-channel attenuation and blue-green cast from water depth.

    At ~5m depth, red light is mostly absorbed by water. This transform
    randomly attenuates R and boosts B to close the gap between iNaturalist
    training photos and actual underwater inference conditions.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] *= random.uniform(0.4, 1.0)  # attenuate red
        arr[:, :, 2] = np.clip(
            arr[:, :, 2] * random.uniform(1.0, 1.2), 0, 255
        )  # boost blue
        return Image.fromarray(arr.astype(np.uint8))


def train_transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.5, 1.0)
            ),  # YOLO crops are already fish-centered, avoid tiny crops
            transforms.RandomHorizontalFlip(),  # fish appear from both sides
            transforms.RandomRotation(degrees=30),  # fish swim at various angles
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05
            ),  # underwater lighting and turbidity
            transforms.RandomApply(
                [UnderwaterColorShift()], p=0.5
            ),  # simulate red attenuation at depth
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.3,  # motion blur from fish and phone movement
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.3
            ),  # snorkeler rarely shoots straight on
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)
            ),  # partial occlusion by coral or other fish
        ]
    )


def val_transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),  # e.g. 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
