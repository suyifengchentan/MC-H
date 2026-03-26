from __future__ import annotations

from typing import Callable, Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TwoViewTransform:
    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_train_transform(image_size: int, profile: str) -> Callable:
    if profile != "fine_grained_two_view":
        raise ValueError(f"Unsupported transform_profile: {profile}")

    single_view = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return TwoViewTransform(single_view)


def build_eval_transform(image_size: int) -> Callable:
    resize_size = int(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

