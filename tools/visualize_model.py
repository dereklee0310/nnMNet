import argparse
import gc
import itertools
import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nnunetv2
import numpy as np
import rich
import rich_argparse
import torch
import torch.nn.functional as F
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from PIL import Image
from torchvision import transforms
from utils.utils import *

# This create the whole trainer, we just need the class for build_network_architecture()
# from nnunetv2.run.run_training import get_trainer_from_args

# Depth = 1: print([n for n, _ in model.named_children()])
# Recursive: print([n for n, _ in model.named_modules()])
# This has been reordered for visualization ;)
ALL_MODULES = [
    "encoder1_l1_local",
    "encoder1_l2_local",
    "encoder1_l3_local",
    "encoder1_l4_local",
    "encoder1_l1_global",
    "encoder1_l2_global",
    "encoder1_l3_global",
    "encoder1_l4_global",
    "decoder1_l1_local",
    "decoder1_l2_local",
    "decoder1_l3_local",
    "decoder1_l4_local",
    "decoder1_l1_global",
    "decoder1_l2_global",
    "decoder1_l3_global",
    "decoder1_l4_global",
    "encoder2_l1_local",
    "encoder2_l2_local",
    "encoder2_l3_local",
    "encoder2_l4_local",
    "encoder2_l1_global",
    "encoder2_l2_global",
    "encoder2_l3_global",
    "encoder2_l4_global",
    "decoder2_l1_local",
    "decoder2_l2_local",
    "decoder2_l3_local",
    "decoder2_l4_local",
    "decoder2_l1_local_output",
    "decoder2_l2_local_output",
    "decoder2_l3_local_output",
    "decoder2_l4_local_output",
    "x_e2_l0_fuse",
    "x_d1_l1_fuse",
    "x_d1_l2_fuse",
    "x_d1_l3_fuse",
    "x_e2_l1_fuse",
    "x_e2_l2_fuse",
    "x_e2_l3_fuse",
    "x_e2_l4_fuse",
    "x_d2_l0_fuse",
    "x_d2_l1_fuse",
    "x_d2_l2_fuse",
    "x_d2_l3_fuse",
    "input_l0",
    "output_l0",
]

ENCODER2_MODULES = [
    "encoder2_l1_local",
    "encoder2_l2_local",
    "encoder2_l3_local",
    "encoder2_l4_local",
    "encoder2_l1_global",
    "encoder2_l2_global",
    "encoder2_l3_global",
    "encoder2_l4_global",
]

L0_MODULES = ("x_e2_l0_fuse", "x_d2_l0_fuse")

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print model's heatmaps.",
        formatter_class=Formatter,
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Visualize the heatmaps of all modules.",
    )

    parser.add_argument(
        "-s",
        "--subnet",
        action="store_true",
        help="Visualize the heatmaps of subnetworks.",
    )

    parser.add_argument(
        "-H",
        "--heatmap",
        action="store_true",
        help="Visualize some of the heatmaps.",
    )

    parser.add_argument(
        "-i",
        "--image",
        action="store_true",
        help="Show original image.",
    )

    parser.add_argument(
        "-g",
        "--groundtruth",
        action="store_true",
        help="Show groundtruth.",
    )

    parser.add_argument(
        "-p",
        "--prediction",
        action="store_true",
        help="Show prediction.",
    )

    parser.add_argument(
        "-I",
        "--input-images",
        type=str,
        nargs="+",
        help="Input image file for heatmap visualization.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory.",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="The device PyTorch use, (default: %(default)s)",
    )

    parser.add_argument(
        "-D",
        "--dataset",
        default="777",
        choices=["777", "778", "779", "780", "781"],
        help="Id of dataset to use, (default: %(default)s)",
    )

    parser.add_argument("trainers", nargs="+", help="Trainer names")
    return parser.parse_args()


class Visualizer:
    def __init__(self, args):
        self.args = args
        self.dataset = DATASETS_INFO[args.dataset]["name"]
        self.num_classes = DATASETS_INFO[args.dataset]["num_classes"]
        self.palette = DATASETS_INFO[args.dataset]["palette"]
        self.mean, self.std = self.get_rgb_mean_std()
        self.output_dir = Path(self.args.output_dir) / self.dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_rgb_mean_std(self):
        with open(get_plans(self.args.dataset)) as f:
            data_dict = json.load(f)["foreground_intensity_properties_per_channel"]
        mean, std = [], []
        for v in data_dict.values():
            mean.append(v["mean"] / 255.0)
            std.append(v["std"] / 255.0)
        return mean, std

    def show(self):
        if self.args.all:
            self.visualize_all()
        if self.args.subnet:
            self.visualize_subnet()
        if self.args.heatmap:
            self.visualize_heatmap()
        if self.args.image:
            self.show_original_image()
        if self.args.groundtruth:
            self.show_groundtruth()
        if self.args.prediction:
            self.show_prediction()

    def transform_image(self, input_image):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        tensor = transform(input_image)
        tensor = tensor.unsqueeze(0)
        return tensor

    def grayscale_to_rgb(self, gray_image):
        palette = np.array(self.palette, dtype=np.uint8)
        return palette[gray_image]

    def get_heatmap(self, act, origin_size=(512, 512)):
        heatmap = torch.mean(act, dim=1).squeeze().cpu().numpy()
        # heatmap = np.maximum(heatmap, 0) # ReLU

        heatmap_resized = cv2.resize(
            heatmap,
            origin_size,
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 1, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET
        )
        return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    def plot_modules(self, axes, activations):
        idx = 0
        for module in ALL_MODULES:
            act = activations[module]
            logger.debug("Processing module: %s, shape: %s", module, act.shape)
            i, j = idx // 4, idx % 4
            idx += 1
            axes[i][j].imshow(self.get_heatmap(act))
            axes[i][j].axis("off")
            axes[i][j].set_title(module)

    def plot_input_pred(self, axes, trainer, input, prediction):
        axes[-1][-2].imshow(input)
        axes[-1][-2].axis("off")
        axes[-1][-2].set_title("input")

        if prediction is not None:
            axes[-1][-1].imshow(self.grayscale_to_rgb(np.array(prediction)))
        axes[-1][-1].axis("off")
        axes[-1][-1].set_title("pred")

    @staticmethod
    def get_activation(name, activations):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    def get_activations(self, trainer, model_class, input_tensor, modules):
        model = model_class()
        # See initialize_from_trained_model_folder in
        # nnunetv2/inference/predict_from_raw_data.py
        checkpoint = torch.load(
            get_checkpoint_file(self.args.dataset, trainer),
            map_location=self.args.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["network_weights"])

        activations = {}
        for module in modules:
            getattr(model, module).register_forward_hook(
                self.get_activation(module, activations)
            )

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return activations

    def visualize_all(self):
        # Create it here as we need a fresh iterator
        trainer_model_pairs = name_model_pairs(self.args.trainers, self.num_classes)
        for trainer, model_class in trainer_model_pairs:
            if trainer != "nnUNetTrainer_MNet":
                logger.error("Eh? Use -a on nnUNetTrainer_MNet, not %s", trainer)
                continue

            for filepath in self.args.input_images:
                input = Image.open(filepath).convert("RGB")
                # For actual inference, use nnunet api or see nnunetv2/inference/predict_from_raw_data.py
                prediction = None
                try:
                    prediction = Image.open(
                        get_prediction(self.args.dataset, trainer, Path(filepath).name)
                    )
                except FileNotFoundError as e:
                    logger.error("Prediction not found, make sure it's in the test set")
                    logger.error(e, exc_info=True)
                input_tensor = self.transform_image(input)
                activations = self.get_activations(
                    trainer, model_class, input_tensor, ALL_MODULES
                )

                fig, axes = plt.subplots(12, 4, figsize=(10, 40))
                self.plot_modules(axes, activations)
                self.plot_input_pred(axes, trainer, input, prediction)
                fig.tight_layout()
                (self.output_dir / trainer).mkdir(parents=True, exist_ok=True)
                output_path = (
                    self.output_dir / trainer / f"{Path(filepath).stem}_heatmap.png"
                )
                fig.savefig(output_path)

    def visualize_subnet(self):
        trainer_model_pairs = name_model_pairs(self.args.trainers, self.num_classes)
        for trainer, model_class in trainer_model_pairs:
            if trainer != "nnUNetTrainer_MNet":
                logger.error("Eh? Use -s on nnUNetTrainer_MNet, not %s", trainer)
                continue
            for filepath in self.args.input_images:
                input_image = Image.open(filepath).convert("RGB")
                input_tensor = self.transform_image(input_image)
                activations = self.get_activations(
                    trainer, model_class, input_tensor, L0_MODULES
                )

                (self.output_dir / trainer).mkdir(parents=True, exist_ok=True)
                for module, name in zip(L0_MODULES, ("subnetwork1", "subnetwork2")):
                    heatmap = Image.fromarray(self.get_heatmap(activations[module]))
                    output_path = (
                        self.output_dir / trainer / f"{Path(filepath).stem}_{name}.png"
                    )
                    heatmap.save(output_path)

    def visualize_heatmap(self):
        trainer_model_pairs = name_model_pairs(self.args.trainers, self.num_classes)
        for trainer, model_class in trainer_model_pairs:
            if trainer not in ("nnUNetTrainer_MNet", "nnUNetTrainer_WNet"):
                logger.error("Eh? Use -s on nnUNetTrainer_MNet or nnUNetTrainer_WNet, not %s", trainer)
                continue
            for filepath in self.args.input_images:
                input_image = Image.open(filepath).convert("RGB")
                input_tensor = self.transform_image(input_image)
                activations = self.get_activations(
                    trainer, model_class, input_tensor, ENCODER2_MODULES
                )

                (self.output_dir / trainer).mkdir(parents=True, exist_ok=True)
                for module in ENCODER2_MODULES:
                    heatmap = Image.fromarray(self.get_heatmap(activations[module]))
                    output_path = (
                        self.output_dir / trainer / f"{Path(filepath).stem}_{module}.png"
                    )
                    heatmap.save(output_path)

    def show_original_image(self):
        for filepath in self.args.input_images:
            filestem = Path(filepath).stem
            shutil.copy(filepath, self.output_dir / f"{filestem}_image.png")

    def show_groundtruth(self):
        for filepath in self.args.input_images:
            filestem = Path(filepath).stem
            filename = Path(filepath).name
            groundtruth = Image.open(get_groundtruth(self.args.dataset, filename))
            groundtruth_rgb = self.grayscale_to_rgb(np.array(groundtruth))
            groundtruth_image = Image.fromarray(groundtruth_rgb.astype(np.uint8))
            groundtruth_image.save(self.output_dir / f"{filestem}_groundtruth.png")

    def show_prediction(self):
        trainer_model_pairs = name_model_pairs(self.args.trainers, self.num_classes)
        for trainer, _ in trainer_model_pairs:
            for filepath in self.args.input_images:
                filestem = Path(filepath).stem
                filename = Path(filepath).name
                try:
                    prediction = Image.open(
                        get_prediction(self.args.dataset, trainer, filename)
                    )
                except FileNotFoundError:
                    logger.critical(
                        "Prediction not found, make sure it's in the test set"
                    )
                    sys.exit()

                (self.output_dir / trainer).mkdir(parents=True, exist_ok=True)
                prediction_rgb = self.grayscale_to_rgb(np.array(prediction))
                prediction_img = Image.fromarray(prediction_rgb.astype(np.uint8))
                prediction_img.save(
                    self.output_dir / trainer / f"{filestem}_prediction.png"
                )


def name_model_pairs(class_names, num_classes):
    # Imitate the way what nnUNet use to find the trainer by name ;)
    # https://github.com/MIC-DKFZ/nnUNet/blob/8c4184d46b60059ff7dc8f74cd535e13554bdeca/nnunetv2/run/run_training.py#L32
    for class_name in class_names:
        folder = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
        module = "nnunetv2.training.nnUNetTrainer"
        trainer = recursive_find_python_class(folder, class_name, module)
        if trainer is None:
            logger.critical(f"Invalid model: {class_name}")
            return
        yield (
            trainer.__name__,
            lambda: trainer.build_network_architecture(
                None,
                None,
                None,
                num_input_channels=3,
                num_output_channels=num_classes,
                enable_deep_supervision=True,
            ),
        )


def main():
    args = parse_args()
    visualizer = Visualizer(args)
    visualizer.show()


if __name__ == "__main__":
    main()
