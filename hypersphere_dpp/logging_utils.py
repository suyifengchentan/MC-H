from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping


def setup_training_logger(output_dir: str | Path) -> logging.Logger:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hypersphere_dpp.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        directory / "train.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def append_metrics_record(output_dir: str | Path, record: Mapping[str, object]) -> None:
    metrics_path = Path(output_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
