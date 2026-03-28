from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class AircraftDataset(Dataset):
    """FGVC-Aircraft variant split loader."""

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.data_root = self.root / "data"
        self.images_base_path = self.data_root / "images"

        split_file = self.data_root / f"images_variant_{split}.txt"
        classes_file = self.data_root / "variants.txt"

        if not self.images_base_path.is_dir():
            raise FileNotFoundError(
                f"FGVC_Aircraft images directory not found: {self.images_base_path}"
            )
        if not split_file.is_file():
            raise FileNotFoundError(f"FGVC_Aircraft split file not found: {split_file}")
        if not classes_file.is_file():
            raise FileNotFoundError(
                f"FGVC_Aircraft class file not found: {classes_file}"
            )

        with classes_file.open("r", encoding="utf-8") as handle:
            self.classes = [line.strip() for line in handle if line.strip()]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                image_id, class_name = line.split(" ", 1)
                self.samples.append((f"{image_id}.jpg", self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]
        image_path = self.images_base_path / image_name
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, idx
