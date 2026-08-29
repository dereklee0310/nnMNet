import argparse
import os
import subprocess
from pathlib import Path
from timeit import default_timer as timer

from rich import print
from rich.panel import Panel
from utils.utils import DATASETS_INFO, Formatter, setup_logger

logger = setup_logger()

LOG_FILE = "testing.log"


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
        default=["0", "1"],
        help="CUDA_VISIBLE_DEVICES, (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size, just a place holder here, (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = DATASETS_INFO[args.dataset]["name"]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in args.device_ids)})
    test_command = [
        "nnUNetv2_predict",
        "-i",
        f"nnUNet_raw/{dataset}/imagesTs/",
        "-o",
        f"nnUNet_results/{dataset}/{args.trainer}__nnUNetPlans__2d/fold_0/prediction",
        "-chk",
        "checkpoint_best.pth" if args.val_best else "checkpoint_final.pth",
        "-d",
        args.dataset,
        "-c",
        "2d",
        "-tr",
        args.trainer,
        "-f",
        "0",
        "-device",
        "cuda",
    ]

    evaluate_command = [
        "nnUNetv2_evaluate_folder",
        "-djfile",
        f"nnUNet_results/{dataset}/{args.trainer}__nnUNetPlans__2d/dataset.json",
        "-pfile",
        f"nnUNet_results/{dataset}/{args.trainer}__nnUNetPlans__2d/plans.json",
        f"nnUNet_raw/{dataset}/labelsTs",
        f"nnUNet_results/{dataset}/{args.trainer}__nnUNetPlans__2d/fold_0/prediction",
    ]

    print(Panel.fit(str(env), title="Envs"))
    print(Panel.fit(" ".join(test_command), title="Test Command"))
    print(Panel.fit(" ".join(evaluate_command), title="Evaluate Command"))

    with open(LOG_FILE, "w") as log_file:
        subprocess.run(test_command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    with open(LOG_FILE, "a+") as log_file:
        subprocess.run(
            evaluate_command, stdout=log_file, stderr=subprocess.STDOUT, env=env
        )

if __name__ == "__main__":
    main()
