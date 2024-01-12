import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import sys
import pytorch_lightning as pl
from pathlib import Path
import os

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


def associate_instances(previous_ins_label, current_ins_label):
    previous_instance_ids, c_p = np.unique(previous_ins_label[previous_ins_label != 0], return_counts=True)
    current_instance_ids, c_c = np.unique(current_ins_label[current_ins_label != 0], return_counts=True)

    associations = {0: 0}

    large_previous_instance_ids = []
    large_current_instance_ids = []
    for id, count in zip(previous_instance_ids, c_p):
        if count > 25:
            large_previous_instance_ids.append(id)
    for id, count in zip(current_instance_ids, c_c):
        if count > 50:
            large_current_instance_ids.append(id)

    p_n = len(large_previous_instance_ids)
    c_n = len(large_current_instance_ids)

    association_costs = torch.zeros(p_n, c_n)
    for i, p_id in enumerate(large_previous_instance_ids):
        for j, c_id in enumerate(large_current_instance_ids):
            intersection = np.sum((previous_ins_label == p_id) & (current_ins_label == c_id))
            union = np.sum(previous_ins_label == p_id) + np.sum(current_ins_label == c_id) - intersection
            iou = intersection / union
            cost = 1 - iou
            association_costs[i, j] = cost

    idxes_1, idxes_2 = linear_sum_assignment(association_costs)

    for i1, i2 in zip(idxes_1, idxes_2):
        if association_costs[i1][i2] < 1.0:
            associations[large_current_instance_ids[i2]] = large_previous_instance_ids[i1]
    return associations


def save_predictions(sem_preds, ins_preds, seq_name, sweep_name):
    filename = Path("/globalwork/yilmaz/submission/sequences") / seq_name / "predictions"
    # assert not filename.exists(), "Path exists"
    filename.mkdir(parents=True, exist_ok=True)
    learning_map_inv = {
        1: 10,  # "car"
        2: 11,  # "bicycle"
        3: 15,  # "motorcycle"
        4: 18,  # "truck"
        5: 20,  # "other-vehicle"
        6: 30,  # "person"
        7: 31,  # "bicyclist"
        8: 32,  # "motorcyclist"
        9: 40,  # "road"
        10: 44,  # "parking"
        11: 48,  # "sidewalk"
        12: 49,  # "other-ground"
        13: 50,  # "building"
        14: 51,  # "fence"
        15: 70,  # "vegetation"
        16: 71,  # "trunk"
        17: 72,  # "terrain"
        18: 80,  # "pole"
        19: 81,  # "traffic-sign"
    }
    sem_preds = np.vectorize(learning_map_inv.__getitem__)(sem_preds)
    panoptic_preds = (ins_preds << 16) + sem_preds
    file_path = str(filename / sweep_name) + ".label"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(panoptic_preds.astype(np.uint32).tobytes())
