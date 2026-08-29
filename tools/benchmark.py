# benchmark.py
import argparse
import time
import os

import numpy as np
import torch

from utils.utils import get_checkpoint_file, MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("trainer", help="Trainer name")

    parser.add_argument(
        "--input-shape", nargs=2, type=int, default=[512, 512], metavar=("H", "W")
    )

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--warmup", type=int, default=50)

    parser.add_argument("--iters", type=int, default=500)

    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


def build_model(trainer):
    if trainer not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown trainer '{trainer}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    # from nnunetv2.training.nnUNetTrainer.ablation import nnUNetTrainer_SecondNet2X
    # model = nnUNetTrainer_SecondNet2X.build_network_architecture(
    #     None, None, None, 3, 9, True
    # )
    # checkpoint = torch.load(
    #     get_checkpoint_file("777", "nnUNetTrainer_SecondNet2X"),
    #     map_location="cpu",
    #     weights_only=False,
    # )

    model = MODEL_REGISTRY[trainer]()
    checkpoint = torch.load(
        get_checkpoint_file("777", trainer),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["network_weights"])
    return model


def load_weights(trainer, model):
    checkpoint = torch.load(
        get_checkpoint_file("777", trainer),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["network_weights"])


def benchmark(model, device, batch_size, h, w, warmup=5, iters=2000):
    model.eval()

    x = torch.randn(batch_size, 3, h, w, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = 0.0

    with torch.no_grad():
        for _ in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start

    latency_ms = total_time / iters * 1000
    fps = batch_size * iters / total_time

    return latency_ms, fps


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = build_model(args.trainer)
    model.to(device)

    latency, fps = benchmark(
        model=model,
        device=device,
        batch_size=args.batch_size,
        h=args.input_shape[0],
        w=args.input_shape[1],
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"Traienr    : {args.trainer}")
    print(f"Latency    : {latency:.3f} ms")
    print(f"FPS        : {fps:.2f}")


if __name__ == "__main__":
    main()
