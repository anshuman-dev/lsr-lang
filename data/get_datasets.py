#!/usr/bin/env python3
"""
Download the LSR-v2 datasets from HuggingFace Hub.

Source: https://huggingface.co/datasets/LSR-FG/lsr-datasets

Usage:
    python data/get_datasets.py --dest data/raw --datasets normal hard rope
"""

import argparse
from pathlib import Path


REPO_ID = "LSR-FG/lsr-datasets"

# (train_file, holdout_file)
DATASET_FILES = {
    "normal": (
        "box_stacking_normal_task_2500.pkl",
        "box_stacking_normal_task_holdout.pkl",
    ),
    "hard": (
        "box_stacking_hard_task_2500.pkl",
        "box_stacking_hard_task_holdout.pkl",
    ),
    "rope": (
        "rope_box_task_2500.pkl",
        "rope_box_task_holdout.pkl",
    ),
}


def download(dest: Path, keys: list):
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for key in keys:
        train_f, holdout_f = DATASET_FILES[key]
        for fname in (train_f, holdout_f):
            out = dest / fname
            if out.exists():
                print(f"[{key}] {fname} already exists, skipping.")
                continue
            print(f"[{key}] downloading {fname} ...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            print(f"[{key}] {fname} done.")
    print("All done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["normal", "hard", "rope"],
        default=["normal", "hard", "rope"],
    )
    args = parser.parse_args()
    download(Path(args.dest), args.datasets)


if __name__ == "__main__":
    main()
