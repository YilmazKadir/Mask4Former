#!/bin/bash
EXPERIMENT_NAME="2024-01-01_000000"

python main_panoptic_4d.py \
general.mode="validate" \
general.ckpt_path="saved/$EXPERIMENT_NAME/last-epoch.ckpt" \
general.dbscan_eps=1.0