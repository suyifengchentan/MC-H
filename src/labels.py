from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelEntry:
    wnid: str
    name: str


_DATASET_TO_LABEL_FILE = {
    "Image10": "imagenet_10.rtf",
    "Image50": "imagenet_50.rtf",
    "Image100": "imagenet_100.rtf",
    "Image200": "imagenet_200.rtf",
}


def default_label_config_path(dataset_name: str) -> Path | None:
    file_name = _DATASET_TO_LABEL_FILE.get(dataset_name)
    if file_name is None:
        return None
    return Path(__file__).resolve().parents[1] / "config" / "labels" / file_name


def load_label_entries(dataset_name: str) -> list[LabelEntry]:
    config_path = default_label_config_path(dataset_name)
    if config_path is None or not config_path.is_file():
        return []

    if config_path.suffix.lower() == ".json":
        return _load_json_entries(config_path)
    return _load_text_entries(config_path)


def load_label_wnids(dataset_name: str) -> list[str]:
    return [entry.wnid for entry in load_label_entries(dataset_name)]


def load_label_name_map(dataset_name: str) -> dict[str, str]:
    return {entry.wnid: entry.name for entry in load_label_entries(dataset_name)}


def _load_json_entries(path: Path) -> list[LabelEntry]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        entries: list[LabelEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            wnid = str(item.get("wnid", "")).strip()
            name = str(item.get("name", "")).strip()
            if wnid:
                entries.append(LabelEntry(wnid=wnid, name=name or wnid))
        return entries

    if isinstance(payload, dict):
        entries: list[LabelEntry] = []
        for wnid, name in payload.items():
            wnid_text = str(wnid).strip()
            if wnid_text:
                entries.append(LabelEntry(wnid=wnid_text, name=str(name).strip() or wnid_text))
        return entries

    raise ValueError(f"Unsupported label JSON format: {path}")


def _load_text_entries(path: Path) -> list[LabelEntry]:
    entries: list[LabelEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            wnid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else wnid
            entries.append(LabelEntry(wnid=wnid, name=name))
    return entries
