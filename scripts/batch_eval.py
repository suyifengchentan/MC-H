#!/usr/bin/env python3
"""
Batch evaluation script for hypersphere_dpp checkpoints.

Scans config/train/*/ for configs, extracts model_path, runs evaluation
using the corresponding test config, and outputs results to a CSV table.

Usage:
    python scripts/batch_eval.py [--output RESULTS_CSV]
"""

from __future__ import annotations

import ast
import csv
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import yaml


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = ArgumentParser(description="Batch evaluate hypersphere_dpp checkpoints.")
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running.",
    )
    return parser.parse_args()


def setup_log_file():
    """Create a log file with timestamp."""
    log_name = f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = PROJECT_ROOT / log_name
    return log_path


def get_train_info(train_cfg_path: Path) -> tuple[str, str]:
    """Extract dataset name and model_path from a train config YAML."""
    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg.get("dataset", "")
    model_path = cfg.get("model_path", "")
    return dataset_name, model_path


def find_checkpoints():
    """Scan save/hypersphere_dpp/ for directories containing .pt files.
    Yields (checkpoint_dir, test_config_rel) by matching checkpoint dir name
    against config/train/*/*.yaml filenames.
    """
    save_dir = PROJECT_ROOT / "save" / "hypersphere_dpp"
    train_dir = PROJECT_ROOT / "config" / "train"

    for ckpt_dir in sorted(save_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        # Find all .pt files in this directory; prefer best.pt, then last.pt
        pt_files = list(ckpt_dir.glob("*.pt"))
        if not pt_files:
            continue

        # Only use best.pt; skip if not found
        pt_names = {p.name for p in pt_files}
        if "best.pt" not in pt_names:
            print(f"[WARN] No best.pt in '{ckpt_dir.name}', skipping", file=sys.stderr)
            continue
        pt_file = "best.pt"
        ckpt_path = str(ckpt_dir / pt_file)

        # Try to find a matching train config by prefix.
        # e.g. "Image200_CLIP_ViTB16_HypersphereDPP_20260328_212034"
        # matches model_path "save/hypersphere_dpp/Image200_CLIP_ViTB16_HypersphereDPP_20260328_212034"
        ckpt_name = ckpt_dir.name

        # Find the corresponding train config
        matched = False
        for category_dir in sorted(train_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for train_cfg in sorted(category_dir.iterdir()):
                if train_cfg.suffix != ".yaml":
                    continue
                dataset_name, model_path = get_train_info(train_cfg)
                if model_path and ckpt_name.startswith(Path(model_path).name):
                    # Found match (checkpoint dir name starts with model_path name, supports timestamp suffix)
                    test_cfg = PROJECT_ROOT / "config" / "test" / category_dir.name / train_cfg.name
                    if test_cfg.exists():
                        test_config_rel = str(test_cfg.relative_to(PROJECT_ROOT / "config" / "test"))
                        yield ckpt_path, test_config_rel, dataset_name
                        matched = True
                        break
            if matched:
                break

        if not matched:
            print(f"[WARN] No matching config found for checkpoint dir '{ckpt_name}', skipping", file=sys.stderr)


def run_eval(test_config_rel: str, checkpoint_path: str, log_path: Path) -> dict | None:
    """
    Run eval.py on the given test_config + checkpoint.
    Returns {"acc": float, "nmi": float, "ari": float} or None on failure.
    """
    eval_script = PROJECT_ROOT / "eval.py"
    cmd = [
        sys.executable, str(eval_script),
        "--config", str(PROJECT_ROOT / "config" / "test" / test_config_rel),
        "--checkpoint", checkpoint_path,
        "--split", "test",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        combined = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            with open(log_path, "a") as f:
                f.write(f"[ERROR] eval failed for {checkpoint_path} (exit {result.returncode})\n")
                f.write(combined)
                f.write("\n" + "=" * 60 + "\n")
            print(f"[ERROR] eval failed for {checkpoint_path}: see {log_path}", file=sys.stderr)
            return None

        # Parse output: looks like "test={'acc': 0.54, 'nmi': 0.67, 'ari': 0.41}"
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("test=") or line.startswith("val="):
                try:
                    dict_str = line.split("=", 1)[1].strip()
                    metrics = ast.literal_eval(dict_str)
                    return metrics
                except Exception as e:
                    with open(log_path, "a") as f:
                        f.write(f"[WARN] Could not parse eval output '{line}': {e}\n")
                    print(f"[WARN] Could not parse eval output: see {log_path}", file=sys.stderr)
        with open(log_path, "a") as f:
            f.write(f"[WARN] No parseable output from eval for {checkpoint_path}\n")
            f.write(f"stdout:\n{result.stdout}\n")
            f.write(f"stderr:\n{result.stderr}\n")
            f.write("\n" + "=" * 60 + "\n")
        print(f"[WARN] No parseable output from eval: see {log_path}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as f:
            f.write(f"[ERROR] eval timed out for {checkpoint_path}\n")
        print(f"[ERROR] eval timed out for {checkpoint_path}", file=sys.stderr)
        return None
    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"[ERROR] eval exception for {checkpoint_path}: {e}\n")
        print(f"[ERROR] eval exception for {checkpoint_path}: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()

    rows = []

    log_path = setup_log_file()

    for ckpt_path, test_config_rel, dataset_name in find_checkpoints():
        if args.dry_run:
            print(f"[DRY-RUN] Would evaluate: dataset={dataset_name}, checkpoint={ckpt_path}")
            continue

        print(f"[INFO] Evaluating: dataset={dataset_name}, checkpoint={ckpt_path}", file=sys.stderr)
        metrics = run_eval(test_config_rel, ckpt_path, log_path)

        checkpoint_dir = str(Path(ckpt_path).parent)
        row = {
            "dataset": dataset_name,
            "checkpoint_dir": checkpoint_dir,
            "ACC": "",
            "NMI": "",
            "ARI": "",
        }

        if metrics:
            row["ACC"] = f"{metrics.get('acc', ''):.4f}" if metrics.get("acc") is not None else ""
            row["NMI"] = f"{metrics.get('nmi', ''):.4f}" if metrics.get("nmi") is not None else ""
            row["ARI"] = f"{metrics.get('ari', ''):.4f}" if metrics.get("ari") is not None else ""

        rows.append(row)
        print(f"  -> ACC={row['ACC']}, NMI={row['NMI']}, ARI={row['ARI']}", file=sys.stderr)

    if args.dry_run:
        return

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "checkpoint_dir", "ACC", "NMI", "ARI"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Results written to {output_path}")
    if log_path.exists():
        print(f"[INFO] Error log written to {log_path}")


if __name__ == "__main__":
    main()
