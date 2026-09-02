#!/bin/bash

nnUNetv2_plan_and_preprocess -d 777 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 778 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 779 --verify_dataset_integrity
uv run tools/convert_to_one_split.py