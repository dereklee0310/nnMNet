import argparse
import json
import os
import subprocess
from pathlib import Path
from timeit import default_timer as timer

from rich import print
from rich.panel import Panel
from utils.utils import DATASETS_INFO, Formatter, setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Default Parser.",
        formatter_class=Formatter,
    )
    parser.add_argument("trainer", help="Trainer name, e.g., nnUNetTrainer_MetaWNet")
    parser.add_argument(
        "-D",
        "--dataset",
        default="777",
        choices=["777", "778", "779", "780", "781"],
        help="Id of dataset to use, (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--val_best",
        action="store_true",
        help="Use the best checkpoint",
    )
    parser.add_argument(
        "-n",
        "--num_gpus",
        default="2",
        type=str,
        help="Number of GPUs to use, (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        nargs="?",
        default=[0, 1],
        help="CUDA_VISIBLE_DEVICES, (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size, (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--strides",
        default=[1, 2, 2, 2, 2, 2, 2, 2],
        nargs="+",
        type=int,
        help="Decoder head strides, (default: %(default)s)",
    )
    parser.add_argument(
        "-V",
        "--validation",
        action="store_true",
        help="Only run the validation",
    )
    parser.add_argument(
        "-c",
        "--cont",
        action="store_true",
        help="Continue training",
    )
    return parser.parse_args()


def change_batch_size(args, batch_size):
    filename = (
        f"nnUNet_preprocessed/{DATASETS_INFO[args.dataset]['name']}/nnUNetPlans.json"
    )
    with open(filename, "r+") as f:
        data = json.load(f)
        old_batch_size = data["configurations"]["2d"]["batch_size"]
        data["configurations"]["2d"]["batch_size"] = batch_size
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    return old_batch_size


def change_strides(args, strides):
    filename = (
        f"nnUNet_preprocessed/{DATASETS_INFO[args.dataset]['name']}/nnUNetPlans.json"
    )

    with open(filename, "r+") as f:
        data = json.load(f)
        old_strides = data["configurations"]["2d"]["architecture"]["arch_kwargs"][
            "strides"
        ]
        data["configurations"]["2d"]["architecture"]["arch_kwargs"]["strides"] = strides
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    return old_strides


def main():
    args = parse_args()
    # Hacky way to use a temporary batch size
    if args.batch_size:
        old_batch_size = change_batch_size(args, args.batch_size)
    # old_strides = change_strides(args, [[x, x] for x in args.strides])
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in args.device_ids)})
    train_command = [
        "nnUNetv2_train",
        args.dataset,
        "2d",
        "0",
        "-tr",
        args.trainer,
        "-num_gpus",
        args.num_gpus,
        "-device",
        "cuda",
    ]
    # Only use it for quick test!
    if args.val_best:
        train_command.append("--val_best")
    if args.validation:
        train_command.append("--val")
    if args.cont:
        train_command.append("--c")
    print(Panel.fit(str(env), title="Envs"))
    print(Panel.fit(" ".join(train_command), title="Command"))

    try:
        with open(f"training_{str(args.device_ids)}.log", "w") as log_file:
            subprocess.run(
                train_command, stdout=log_file, stderr=subprocess.STDOUT, env=env
            )
    finally:
        if args.batch_size:
            _ = change_batch_size(args, old_batch_size)  # Change it back
        # _ = change_strides(args, old_strides)  # Change it back


if __name__ == "__main__":
    main()
