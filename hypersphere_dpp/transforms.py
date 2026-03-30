from __future__ import annotations

from typing import Callable

import kornia.augmentation as K
import torch
from PIL import ImageOps
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TwoViewBatchTransform(nn.Module):
    def __init__(self, transform: nn.Module):
        super().__init__()
        self.transform = transform

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(images), self.transform(images)


class Uint8ToFloat(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.float().div(255.0)


class PILToTensorCopy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.PILToTensor()

    def forward(self, image) -> torch.Tensor:
        # torchvision may return tensors backed by non-resizable storages here,
        # which breaks DataLoader's default collate path. Clone into a plain tensor.
        return self.transform(image).clone()


class ResizeKeepAspectPadToSquare(nn.Module):
    def __init__(
        self,
        size: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        fill: tuple[int, int, int] = tuple(round(value * 255) for value in CLIP_MEAN),
    ):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = transforms.functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
        )

        pad_width = self.size - resized_width
        pad_height = self.size - resized_height
        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - (pad_width // 2),
            pad_height - (pad_height // 2),
        )
        return ImageOps.expand(resized, border=padding, fill=self.fill)


class NormalizeBatch(nn.Module):
    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


def _gaussian_blur_kernel_size(image_size: int) -> int:
    kernel_size = max(3, image_size // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _disable_kornia_extra_features(module: nn.Module) -> None:
    # Kornia ImageModule/ImageSequential keeps a detached CPU copy of outputs by
    # default for visualization helpers, which can cause large host-memory spikes
    # during training. Disable those extras for the training path.
    if hasattr(module, "disable_features"):
        module.disable_features = True
    for child in module.children():
        _disable_kornia_extra_features(child)


def _build_gpu_two_view_transform(
    image_size: int,
    crop_scale: tuple[float, float],
    crop_ratio: tuple[float, float],
    color_jitter: tuple[float, float, float, float],
    color_jitter_p: float,
    grayscale_p: float,
    blur_p: float,
) -> nn.Module:
    ops: list[nn.Module] = [
        Uint8ToFloat(),
        K.RandomResizedCrop(
            size=(image_size, image_size),
            scale=crop_scale,
            ratio=crop_ratio,
            resample="BICUBIC",
            p=1.0,
        ),
        K.RandomHorizontalFlip(p=0.5),
    ]
    if color_jitter_p > 0:
        ops.append(
            K.ColorJiggle(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
                p=color_jitter_p,
            )
        )
    if grayscale_p > 0:
        ops.append(K.RandomGrayscale(p=grayscale_p))
    if blur_p > 0:
        ops.append(
            K.RandomGaussianBlur(
                kernel_size=(
                    _gaussian_blur_kernel_size(image_size),
                    _gaussian_blur_kernel_size(image_size),
                ),
                sigma=(0.1, 2.0),
                p=blur_p,
            )
        )
    ops.append(NormalizeBatch(mean=CLIP_MEAN, std=CLIP_STD))
    transform = nn.Sequential(*ops)
    _disable_kornia_extra_features(transform)
    return TwoViewBatchTransform(transform)


def build_train_decode_transform(image_size: int, profile: str) -> Callable:
    if profile in {"tiny_imagenet200_two_view", "small_image_two_view"}:
        # Tiny-ImageNet style samples are already small; keep CPU decode cheap and
        # let the GPU-side crop/resize path do the upscale work.
        return PILToTensorCopy()

    resize_size = int(image_size / 0.875)
    if profile == "stanford_dogs_two_view":
        return transforms.Compose(
            [
                # Stanford Dogs contains many elongated full-body images; preserve
                # aspect ratio here instead of warping them into a square before
                # the GPU-side random crop creates the two training views.
                ResizeKeepAspectPadToSquare(
                    resize_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                PILToTensorCopy(),
            ]
        )

    return transforms.Compose(
        [
            # Keep a fixed square decode size so DataLoader can stack samples,
            # while still leaving the actual random crop to the GPU transform.
            transforms.Resize(
                (resize_size, resize_size), interpolation=InterpolationMode.BICUBIC
            ),
            PILToTensorCopy(),
        ]
    )


def build_gpu_train_transform(image_size: int, profile: str) -> nn.Module:
    if profile == "cub_fine_grained_two_view":
        # CUB is a small fine-grained dataset where color/texture cues are
        # crucial. Use milder crops and weaker color destruction to preserve
        # discriminative local details.
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.65, 1.0),
            crop_ratio=(0.85, 1.2),
            color_jitter=(0.15, 0.15, 0.12, 0.03),
            color_jitter_p=0.7,
            grayscale_p=0.0,
            blur_p=0.0,
        )

    if profile == "fine_grained_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.5, 1.0),
            crop_ratio=(0.75, 1.3333333333),
            color_jitter=(0.2, 0.2, 0.2, 0.05),
            color_jitter_p=0.8,
            grayscale_p=0.1,
            blur_p=0.0,
        )

    if profile == "stanford_dogs_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.65, 1.0),
            crop_ratio=(0.85, 1.2),
            color_jitter=(0.12, 0.12, 0.12, 0.03),
            color_jitter_p=0.6,
            grayscale_p=0.02,
            blur_p=0.0,
        )

    if profile == "imagenet10_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.35, 1.0),
            crop_ratio=(0.7, 1.4),
            color_jitter=(0.4, 0.4, 0.25, 0.08),
            color_jitter_p=0.8,
            grayscale_p=0.05,
            blur_p=0.15,
        )

    if profile == "imagenet50_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.45, 1.0),
            crop_ratio=(0.75, 1.3333333333),
            color_jitter=(0.3, 0.3, 0.2, 0.05),
            color_jitter_p=0.75,
            grayscale_p=0.05,
            blur_p=0.1,
        )

    if profile == "imagenet100_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.45, 1.0),
            crop_ratio=(0.75, 1.3333333333),
            color_jitter=(0.3, 0.3, 0.2, 0.05),
            color_jitter_p=0.75,
            grayscale_p=0.05,
            blur_p=0.1,
        )

    if profile == "tiny_imagenet200_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.6, 1.0),
            crop_ratio=(0.9, 1.1),
            color_jitter=(0.2, 0.2, 0.15, 0.03),
            color_jitter_p=0.6,
            grayscale_p=0.02,
            blur_p=0.0,
        )

    if profile == "small_image_two_view":
        return _build_gpu_two_view_transform(
            image_size=image_size,
            crop_scale=(0.7, 1.0),
            crop_ratio=(0.9, 1.1),
            color_jitter=(0.25, 0.25, 0.2, 0.03),
            color_jitter_p=0.6,
            grayscale_p=0.02,
            blur_p=0.0,
        )

    raise ValueError(f"Unsupported transform_profile: {profile}")


def build_eval_transform(image_size: int, profile: str | None = None) -> Callable:
    if profile in {"tiny_imagenet200_two_view", "small_image_two_view"}:
        return transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    resize_size = int(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
