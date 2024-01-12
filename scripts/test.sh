#!/bin/bash
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

EXPERIMENT_NAME="2024-01-01_000000"

python main_panoptic.py \
general.mode="test" \
general.ckpt_path="saved/$EXPERIMENT_NAME/last-epoch.ckpt" \
general.dbscan_eps=1.0