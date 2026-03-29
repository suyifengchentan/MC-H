#!/usr/bin/env python3
"""
Batch evaluation script for hypersphere_dpp checkpoints.

Scans save/hypersphere_dpp/*/ for checkpoints (best.pt), runs evaluation
using the corresponding test config, and outputs results to a CSV table.

Usage:
    python scripts/batch_eval.py [--remote USER@HOST] [--remote-base REMOTE_BASE]
                                 [--output RESULTS_CSV]
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


# Project root (directory containing this script's parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SSH_KEY = os.path.expanduser("~/.openclaw/workspace/privatekey/readonly_key")


def parse_args():
    parser = ArgumentParser(description="Batch evaluate hypersphere_dpp checkpoints.")
    parser.add_argument(
        "--remote",
        default=None,
        help="Remote SSH connection string, e.g. geyingjie@192.168.1.7:8811. "
             "If omitted, runs locally.",
    )
    parser.add_argument(
        "--remote-base",
        default="/home/geyingjie/hypersphere",
        help="Base path on remote machine where the project lives.",
    )
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


def get_train_info(train_cfg_path: Path) -> tuple[str, str]:
    """Extract dataset name and model_path from a train config YAML."""
    import yaml
    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg.get("dataset", "")
    model_path = cfg.get("model_path", "")
    return dataset_name, model_path


def find_configs():
    """Find all train and test config pairs, yield (dataset_name, model_path, test_config_rel)."""
    train_dir = PROJECT_ROOT / "config" / "train"
    test_dir = PROJECT_ROOT / "config" / "test"

    for category_dir in sorted(train_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for train_cfg in sorted(category_dir.iterdir()):
            if train_cfg.suffix != ".yaml":
                continue

            # Corresponding test config
            test_cfg = test_dir / category_dir.name / train_cfg.name
            if not test_cfg.exists():
                print(f"[WARN] No test config for {train_cfg.name}, skipping", file=sys.stderr)
                continue

            dataset_name, model_path = get_train_info(train_cfg)
            if not dataset_name or not model_path:
                print(f"[WARN] Missing dataset or model_path in {train_cfg}, skipping", file=sys.stderr)
                continue

            # Relative path from config/test to test config: "coarse_grained/cifar10_..."
            test_config_rel = str(test_cfg.relative_to(PROJECT_ROOT / "config" / "test"))

            yield dataset_name, model_path, test_config_rel


def build_remote_cmd(remote: str, remote_base: str, script_path: str, *script_args):
    """Build an SSH command that runs script_path on remote with script_args."""
    return [
        "ssh", "-i", REMOTE_SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        remote,
        f"cd {remote_base} && python {script_path} {' '.join(script_args)}"
    ]


def run_eval(test_config: str, checkpoint: str, remote: str | None, remote_base: str) -> dict | None:
    """
    Run eval.py on the given test_config + checkpoint.
    Returns {"ACC": float, "NMI": float, "ARI": float} or None on failure.
    """
    eval_script = PROJECT_ROOT / "eval.py"

    if remote:
        ssh_key = os.environ.get("SSH_KEY", REMOTE_SSH_KEY)
        cmd = [
            "ssh", "-i", ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            remote,
            f"cd {remote_base} && python eval.py --config config/test/{test_config} --checkpoint {checkpoint} --split test"
        ]
    else:
        cmd = [
            sys.executable, str(eval_script),
            "--config", str(PROJECT_ROOT / "config" / "test" / test_config),
            "--checkpoint", checkpoint,
            "--split", "test",
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[ERROR] eval failed for {checkpoint}: {result.stderr[:300]}", file=sys.stderr)
            return None

        # Parse output: looks like "test={'acc': 0.54, 'nmi': 0.67, 'ari': 0.41}"
        output = result.stdout.strip()
        # Find the dict-like string
        import ast
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("test=") or line.startswith("val="):
                try:
                    # Extract the dict part
                    dict_str = line.split("=", 1)[1].strip()
                    metrics = ast.literal_eval(dict_str)
                    return metrics
                except Exception as e:
                    print(f"[WARN] Could not parse eval output '{line}': {e}", file=sys.stderr)
        print(f"[WARN] No parseable output from eval for {checkpoint}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"[ERROR] eval timed out for {checkpoint}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERROR] eval exception for {checkpoint}: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()

    rows = []

    for dataset_name, model_path, test_config_rel in find_configs():
        # Checkpoint path relative to project root
        checkpoint_path = str(Path(model_path) / "best.pt")
        # For remote, the full remote path
        remote_checkpoint = f"{args.remote_base}/{model_path}/best.pt" if args.remote else checkpoint_path

        if args.dry_run:
            print(f"[DRY-RUN] Would evaluate: dataset={dataset_name}, checkpoint={remote_checkpoint}, test_config={test_config_rel}")
            continue

        print(f"[INFO] Evaluating: dataset={dataset_name}, checkpoint={remote_checkpoint}", file=sys.stderr)
        metrics = run_eval(test_config_rel, remote_checkpoint, args.remote, args.remote_base)

        row = {
            "dataset": dataset_name,
            "checkpoint_dir": model_path,
            "ACC": "",
            "NMI": "",
            "ARI": "",
        }

        if metrics:
            row["ACC"] = f"{metrics.get('acc', ''):.4f}" if metrics.get("acc") is not None else ""
            row["NMI"] = f"{metrics.get('nmi', ''):.4f}" if metrics.get("nmi") is not None else ""
            row["ARI"] = f"{metrics.get('ari', ''):.4f}" if metrics.get("ari") is not None else ""

        rows.append(row)

        # Print intermediate result
        print(f"  -> ACC={row['ACC']}, NMI={row['NMI']}, ARI={row['ARI']}", file=sys.stderr)

    if args.dry_run:
        return

    # Write CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "checkpoint_dir", "ACC", "NMI", "ARI"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Results written to {output_path}")


if __name__ == "__main__":
    main()
