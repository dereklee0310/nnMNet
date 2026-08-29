import argparse
import logging
import logging.config
from collections import Counter
from pathlib import Path

import numpy as np
import rich.logging  # noqa: F401
import rich_argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from PIL import Image
from torchinfo import summary as _summary

from functools import partial

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WNet2D import WNet2D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_MNet import MNet, MNetS


class Formatter(
    rich_argparse.RawTextRichHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    # argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter
    pass


DATASETS_INFO = {
    "777": {
        "name": "Dataset777_SynMars-TW",
        "prefix": "nnUNet_results/Dataset777_SynMars-TW/",
        "num_classes": 9,
        "classes": {
            # 0: "background",
            1: "Big rock",
            2: "Small rock",
            3: "Gravel",
            4: "Bedrock",
            5: "Ridge",
            6: "Sand",
            7: "Soil",
            8: "Sky",
        },
        "input_shape": (1, 3, 512, 512),
        "palette": (
            (0, 0, 0),
            (255, 255, 0),
            (255, 150, 0),
            (255, 0, 0),
            (0, 255, 0),
            (255, 190, 255),
            (0, 255, 255),
            (0, 0, 255),
            (102, 102, 102),
        ),
    },
    "778": {
        "name": "Dataset778_MarsScapes",
        "prefix": "nnUNet_results/Dataset778_MarsScapes/",
        "num_classes": 10,
        "classes": {
            # 0: "background",
            1: "Soil",
            2: "Bedrock",
            3: "Gravel",
            4: "Sand",
            5: "Big rock",
            6: "Ridge",
            7: "Sky",
            8: "Rover",
            9: "Unlabeled",
        },
        "input_shape": (1, 3, 256, 512),
        "palette": (
            (0, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 0),
            (128, 128, 128),
            (34, 56, 19),
            (0, 85, 0),
            (170, 85, 0),
        ),
    },
    "779": {
        "name": "Dataset779_SynMars-Air",
        "prefix": "nnUNet_results/Dataset779_SynMars-Air/",
        "num_classes": 9,
        "classes": {
            # 0: "background",
            1: "Big rock",
            2: "Small rock",
            3: "Gravel",
            4: "Bedrock",
            5: "Ridge",
            6: "Sand",
            7: "Soil",
            8: "Sky",
        },
        "input_shape": (1, 3, 512, 512),
        "palette": (
            (0, 0, 0),
            (255, 255, 0),
            (255, 150, 0),
            (255, 0, 0),
            (0, 255, 0),
            (255, 190, 255),
            (0, 255, 255),
            (0, 0, 255),
            (102, 102, 102),
        ),
    },
    "780": {
        "name": "Dataset780_MSL-Seg",
        "prefix": "nnUNet_results/Dataset780_MSL-Seg/",
        "num_classes": 10,
        "classes": {
            # 0: "background",
            1: "Martian Soil",
            2: "Sands",
            3: "Gravel",
            4: "Bedrock",
            5: "Rocks",
            6: "Tracks",
            7: "Shadows",
            8: "Background",
            9: "Unknown",
        },
        "input_shape": (1, 3, 500, 560),
        "palette": (
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (192, 0, 0),
            (64, 0, 0),
        ),
    },
    "781": {
        "name": "Dataset781_S5Mars",
        "prefix": "nnUNet_results/Dataset781_S5Mars/",
        "num_classes": 10,
        "classes": {
            # 0: "background",
            1: "Sky",
            2: "Ridge",
            3: "Soil",
            4: "Sand",
            5: "Bedrock",
            6: "Rock",
            7: "Rover",
            8: "Trace",
            9: "Hole",
        },
        "input_shape": (1, 3, 512, 512),
        "palette": (
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 64, 0),
            (128, 128, 128),
            (255, 128, 0),
        ),
    },
}

# Turn off the deep supervision!
MODEL_REGISTRY = {
    "nnUNetTrainer_1000epochs": partial(
        nnUNetTrainer.build_network_architecture,
        "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        {
            "n_stages": 8,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
            ],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_conv_per_stage": [2, 2, 2, 2, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2, 2, 2, 2, 2],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        ["conv_op", "norm_op", "dropout_op", "nonlin"],
        3,
        9,
        False,
    ),
    # "nnUNetTrainer_WNet": partial(WNet2D, 3, 9, False),
    "nnUNetTrainer_MNet": partial(MNet, 3, 9, False),
    "nnUNetTrainer_MNetS": partial(MNetS, 3, 9, False),
}


def summary(model, input, device):
    _summary(model=model, input_size=input, device=device)


def get_params(model, input, device):
    return f"{_summary(model, input, verbose=0, device=device).total_params / 1e6:.2f}"


def _get_flops(model, input, device):
    model.eval()
    model.to(device)
    input = torch.rand(input).to(device)
    return FlopCountAnalysis(model, input)


def get_flops(model, input, device):
    flops = _get_flops(model, input, device)
    return f"{flops.total() / 1e9:.2f}"


def flop(model, input, device):
    flops = _get_flops(model, input, device)
    print(f"{flop_count_table(flops)}")


def setup_logger() -> logging.Logger:
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        # "filters": {}
        "formatters": {
            # RichHandler do the job for us, so we don't need to incldue time & level
            "iso-8601-simple": {
                "format": "%(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "iso-8601-detailed": {
                "format": "%(asctime)s [%(levelname)s] %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "stdout": {
                "level": "INFO",
                "formatter": "iso-8601-simple",
                "()": "rich.logging.RichHandler",
                "rich_tracebacks": True,
            },
            # "file": {
            #     "class": "logging.handlers.RotatingFileHandler",
            #     "level": "INFO",
            #     "formatter": "iso-8601-detailed",
            #     "filename": "logs/.log",
            #     "maxBytes": 10000,
            #     "backupCount": 0,
            # },
        },
        "loggers": {"root": {"level": "INFO", "handlers": ["stdout"]}},
        # "loggers": {"root": {"level": "INFO", "handlers": ["stdout", "file"]}},
    }
    logging.config.dictConfig(config=logging_config)
    return logging.getLogger(__name__)


def test_model(model, input_size=(1, 3, 512, 512)):
    print(model(torch.rand(*input_size))[0].shape)


def get_plans(dataset_id):
    # return f"nnUNet_results/{dataset}/{trainer}__nnUNetPlans__2d/plans.json"
    return f"nnUNet_preprocessed/{get_dataset_name(dataset_id)}/nnUNetPlans.json"


def get_checkpoint_file(dataset_id, trainer):
    return f"nnUNet_results/{get_dataset_name(dataset_id)}/{trainer}__nnUNetPlans__2d/fold_0/checkpoint_final.pth"


def get_prediction(dataset_id, trainer, filename):
    return f"nnUNet_results/{get_dataset_name(dataset_id)}/{trainer}__nnUNetPlans__2d/fold_0/prediction/{filename}"


def get_groundtruth(dataset_id, filename):
    return f"nnUNet_raw/{get_dataset_name(dataset_id)}/labelsTs/{filename}"


def get_image_dir(dataset_id):
    return f"nnUNet_raw/{get_dataset_name(dataset_id)}/imagesTs"


def get_dataset_name(dataset_id):
    return DATASETS_INFO[dataset_id]["name"]


if __name__ == "__main__":
    pass
