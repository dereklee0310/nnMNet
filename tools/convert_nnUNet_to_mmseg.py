import argparse
import gc
import itertools
import json
import shutil
from pathlib import Path

import nnunetv2
import rich
import rich_argparse
import torch
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class Formatter(
    rich_argparse.RawTextRichHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    # argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an nnUNet dataset to a mmsegmentation dataset.",
        formatter_class=Formatter,
    )
    parser.add_argument(
        "src_dir", help="Source directory, e.g., nnUNet_raw/Dataset777_SynMars-TW/"
    )
    parser.add_argument(
        "dst_dir", help="Destination directory, e.g. mmseg_datasets/SynMars-TW"
    )
    parser.add_argument(
        "split_file",
        help="Split file, e.g., nnUNet_preprocessed/Dataset777_SynMars-TW/splits_final.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "images/training").mkdir(parents=True, exist_ok=True)
    (dst / "images/validation").mkdir(parents=True, exist_ok=True)
    (dst / "images/testing").mkdir(parents=True, exist_ok=True)
    (dst / "annotations/training").mkdir(parents=True, exist_ok=True)
    (dst / "annotations/validation").mkdir(parents=True, exist_ok=True)
    (dst / "annotations/testing").mkdir(parents=True, exist_ok=True)

    with open(args.split_file) as f:
        data = json.load(f)[0]

    for file_id in data["train"]:
        src_file = src / "imagesTr" / f"{file_id}_0000.png"
        dst_file = dst / "images/training" / f"{file_id}.png"
        shutil.copy(src_file, dst_file)
        src_file = src / "labelsTr" / f"{file_id}.png"
        dst_file = dst / "annotations/training" / f"{file_id}.png"
        shutil.copy(src_file, dst_file)

    for file_id in data["val"]:
        src_file = src / "imagesTr" / f"{file_id}_0000.png"
        dst_file = dst / "images/validation" / f"{file_id}.png"
        shutil.copy(src_file, dst_file)
        src_file = src / "labelsTr" / f"{file_id}.png"
        dst_file = dst / "annotations/validation" / f"{file_id}.png"
        shutil.copy(src_file, dst_file)

    # Need to remove _0000 for images
    for filename in (src / "imagesTs").glob("*"):
        shutil.copy(filename, dst / "images/testing" / f"{filename.stem[:-5]}.png")
    shutil.copytree(src / "labelsTs", dst / "annotations/testing", dirs_exist_ok=True)
