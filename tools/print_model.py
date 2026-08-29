import argparse
import gc
import itertools
import json
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
from tabulate import tabulate
from utils.utils import DATASETS_INFO, Formatter, flop, get_flops, get_params, summary

# Shit for quick debug, PlainConvUNet
# from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
# INPUT = (1, 3, 512, 512) # 256
# with open("nnUNet_preprocessed/Dataset777_SynMars-TW/nnUNetPlans.json") as f:
# with open("nnUNet_preprocessed/Dataset778_MarsScapes/nnUNetPlans.json") as f:
#     nnUNetPlans = json.load(f)["configurations"]["2d"]["architecture"]
# nnUNetPlans["arch_class_name"] = nnUNetPlans["network_class_name"]
# del nnUNetPlans["network_class_name"]
# nnUNetPlans["input_channels"] = 3
# nnUNetPlans["output_channels"] = 9 # 10
# nnUNetPlans["deep_supervision"] = True
# nnUNetPlans["arch_kwargs_req_import"] = nnUNetPlans["_kw_requires_import"]
# del nnUNetPlans["_kw_requires_import"]
# nnUNet = get_network_from_plans(**nnUNetPlans)
# print(f"Params: {get_params(nnUNet, INPUT, 'cuda')}")
# print(f"Flops: {get_flops(nnUNet, INPUT, 'cuda')}")
# exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print model's info.",
        formatter_class=Formatter,
    )
    parser.add_argument(
        "-t",
        "--type",
        default="b",
        choices=["p", "f", "b"],
        help="Print model parameters, flops, or both, (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the full model summary or/and flop count table",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="The device PyTorch use, (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Use summary.json under prediction/ (Test set) instead of validation/",
    )
    parser.add_argument(
        "-i",
        "--iou",
        action="store_true",
        help="Print model mIoU",
    )
    parser.add_argument(
        "-I",
        "--class-iou",
        action="store_true",
        help="Print model IoU for each class",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="Use models from other frameworks, e.g. Light4Mars-B*",
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


class ModelAnalyzer:
    def __init__(self, args, name_model_pairs):
        self.args = args
        self.name_model_pairs, self.backup_pairs = itertools.tee(name_model_pairs)
        self.input_shape = DATASETS_INFO[self.args.dataset]["input_shape"]

    def get_info(self, name, *args):
        print(name)
        info = [name.replace("nnUNetTrainer_", "")]

        set = "prediction" if self.args.predict else "validation"
        prefix = DATASETS_INFO[self.args.dataset]["prefix"]
        summary_file = f"{prefix}/{name}__nnUNetPlans__2d/fold_0/{set}/summary.json"
        num_classes = len(DATASETS_INFO[self.args.dataset]["classes"])
        try:
            with open(summary_file) as f:
                data = json.load(f)
            if self.args.class_iou:
                for class_iou in data["mean"].values():
                    info.append(f"{class_iou['IoU'] * 100:.2f}")
            if self.args.iou:
                info.append(f"{data['foreground_mean']['IoU'] * 100:.2f}")
        except FileNotFoundError as e:
            print(e)
            print(f"File not found! Skipping mIoU of {name}")
            if self.args.class_iou:
                info.extend(["N/A"] * num_classes)
            if self.args.iou:
                info.append("N/A")

        if not args[0]:
            info.extend([None, None])
        else:
            if self.args.type == "b":
                if self.args.verbose:
                    print(args[0])
                    summary(*args)
                    flop(*args)
                info.extend([get_params(*args), get_flops(*args)])
            elif self.args.type == "p":
                if self.args.verbose:
                    summary(*args)
                info.append(get_params(*args))
            else:  # "f"
                if self.args.verbose:
                    flop(*args)
                info.append(get_flops(*args))
        return info

    def print(self):
        headers = ["Model"]

        if self.args.class_iou:
            headers.extend(DATASETS_INFO[self.args.dataset]["classes"].values())
        if self.args.iou:
            headers.append("mIoU (%)")

        if self.args.type == "b":
            headers.extend(["Params (M)", "FLOPs (G)"])
        elif self.args.type == "p":
            headers.append("Params (M)")
        else:
            headers.append("FLOPs (G)")

        table = Table(*headers, box=box.HEAVY)
        data = []
        # if self.args.models:
        #     with open("plots/results.json") as f:
        #         models_data = json.load(f)

        #     for model_name in self.args.models:
        #         model_info = models_data[model_name][self.args.dataset]
        #         info = [
        #             model_name,
        #             str(model_info["params"]),
        #             str(model_info["flops"]),
        #         ]
        #         if self.args.iou:
        #             info.append(f"{model_info["foreground_mean"]["IoU"] * 100:.2f}")
        #             table.add_row(*info)
        #             data.append(info)
        #         else:
        #             table.add_row(*info)
        #             data.append(info)

        for name, model_class in self.name_model_pairs:
            try:
                model = model_class()
            except TypeError:
                model = None
            if self.args.verbose:
                rich.print(Panel.fit(name))
            model_info = self.get_info(name, model, self.input_shape, self.args.device)
            table.add_row(*model_info)
            data.append(model_info)
            if model:
                model.cpu()
                del model
                gc.collect()
            torch.cuda.empty_cache()
        # Console().print(table)
        print(tabulate(data, headers=headers, tablefmt="github", floatfmt=".2f"))
        self.name_model_pairs, self.backup_pairs = itertools.tee(self.backup_pairs)

    def get_dict(self):
        """Just for convenience so we can merge other metrics like mIoU to it."""
        # we better load it one by one...
        data = []
        for name, model_class in self.name_model_pairs:
            model = model_class()
            model_info = self.get_info(name, model, self.input_shape, self.args.device)
            data.append(model_info)
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        self.name_model_pairs, self.backup_pairs = itertools.tee(self.backup_pairs)
        return data


# def name_model_pairs(class_names):
#     # Imitate the way what nnUNet use to find the trainer by name ;)
#     # https://github.com/MIC-DKFZ/nnUNet/blob/8c4184d46b60059ff7dc8f74cd535e13554bdeca/nnunetv2/run/run_training.py#L32
#     for class_name in class_names:
#         folder = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
#         module = "nnunetv2.training.nnUNetTrainer"
#         trainer = recursive_find_python_class(folder, class_name, module)
#         if trainer is None:
#             print(f"Invalid model: {class_name}")
#             return
#         yield (
#             trainer.__name__,
#             lambda: trainer.build_network_architecture(
#                 None,
#                 None,
#                 None,
#                 num_input_channels=3,
#                 num_output_channels=8,
#                 enable_deep_supervision=True,
#             ),
#         )

def name_model_pairs(class_names, num_classes):
    # Imitate the way what nnUNet use to find the trainer by name ;)
    # https://github.com/MIC-DKFZ/nnUNet/blob/8c4184d46b60059ff7dc8f74cd535e13554bdeca/nnunetv2/run/run_training.py#L32
    for class_name in class_names:
        folder = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
        module = "nnunetv2.training.nnUNetTrainer"
        trainer = recursive_find_python_class(folder, class_name, module)
        if trainer is None:
            print(f"Invalid model: {class_name}")
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
    # device = torch.device('cuda:0')
    # free, total = torch.cuda.mem_get_info(device)
    # used_mem = (total - free) / 1024 ** 2
    # print(used_mem)
    args = parse_args()
    num_classes = DATASETS_INFO[args.dataset]["num_classes"]
    analyzer = ModelAnalyzer(args, name_model_pairs(args.trainers, num_classes))
    analyzer.print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------- #
#                                  Deprecated                                  #
# ---------------------------------------------------------------------------- #
# This has been replaced by a simpler summary function that only return params
# Call it like this! simple_summary(summary(model, INPUT, verbose=0))

# DIVIDER_LEN = 80
# DIVIDER = "=" * DIVIDER_LEN
# def simple_summary(summary: ModelStatistics):
#     """Print results of the summary."""
#     total_params = ModelStatistics.format_output_num(
#         summary.total_params, summary.formatting.params_units
#     )
#     trainable_params = ModelStatistics.format_output_num(
#         summary.trainable_params, summary.formatting.params_units
#     )
#     non_trainable_params = ModelStatistics.format_output_num(
#         summary.total_params - summary.trainable_params,
#         summary.formatting.params_units,
#     )
#     summary_str = (
#         f"Total params{total_params}\n"
#         f"Trainable params{trainable_params}\n"
#         f"Non-trainable params{non_trainable_params}\n"
#     )
#     if summary.input_size:
#         macs = ModelStatistics.format_output_num(
#             summary.total_mult_adds, summary.formatting.macs_units
#         )
#         input_size = summary.to_megabytes(summary.total_input)
#         output_bytes = summary.to_megabytes(summary.total_output_bytes)
#         param_bytes = summary.to_megabytes(summary.total_param_bytes)
#         total_bytes = summary.to_megabytes(
#             summary.total_input
#             + summary.total_output_bytes
#             + summary.total_param_bytes
#         )
#         summary_str += (
#             f"Total mult-adds{macs}\n{DIVIDER}\n"
#             f"Input size (MB): {input_size:0.2f}\n"
#             f"Forward/backward pass size (MB): {output_bytes:0.2f}\n"
#             f"Params size (MB): {param_bytes:0.2f}\n"
#             f"Estimated Total Size (MB): {total_bytes:0.2f}\n"
#         )
#     summary_str += DIVIDER
#     print(summary_str)


# from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
# with open(
#     "nnUNet_preprocessed/Dataset778_MarsScapes/nnUNetResEncUNetMPlans.json"
# ) as f:
#     ResEncUNetMPlans = json.load(f)["configurations"]["2d"]["architecture"]
# ResEncUNetMPlans["arch_class_name"] = ResEncUNetMPlans["network_class_name"]
# del ResEncUNetMPlans["network_class_name"]
# ResEncUNetMPlans["input_channels"] = 3
# ResEncUNetMPlans["output_channels"] = 1
# ResEncUNetMPlans["arch_kwargs_req_import"] = ResEncUNetMPlans["_kw_requires_import"]
# del ResEncUNetMPlans["_kw_requires_import"]
# ResEncUNetM = get_network_from_plans(**ResEncUNetMPlans)
# print(f"Params: {get_params(ResEncUNetM, INPUT, "cuda")}")
# print(f"Flops: {get_flops(ResEncUNetM, INPUT, "cuda")}")

# ---------------------------------------------------------------------------- #
#               Visualize PyTorch execution graph, too verbose :(              #
# ---------------------------------------------------------------------------- #
# from torchviz import make_dot
# x = torch.randn(1, 3, 512, 512).to("cuda")
# y = wnet(x)
# dot = make_dot(y[0].mean(), params=dict(wnet.named_parameters()))
# dot.format = "png"
# dot.render("./plots/model")
