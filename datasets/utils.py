import numpy as np
import yaml
from pathlib import Path


def save_database(database, mode, save_dir):
    for element in database:
        dict_to_yaml(element)
    save_yaml(save_dir / (mode + "_database.yaml"), database)


def save_yaml(path, file):
    with open(path, "w") as f:
        yaml.safe_dump(file, f, default_style=None, default_flow_style=False)


def dict_to_yaml(dictionary):
    if not isinstance(dictionary, dict):
        return
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dict_to_yaml(v)
        if isinstance(v, np.ndarray):
            dictionary[k] = v.tolist()
        if isinstance(v, Path):
            dictionary[k] = str(v)


def load_yaml(filepath):
    with open(filepath) as f:
        file = yaml.safe_load(f)
    return file


def parse_calibration(filename):
    calib = {}

    with open(filename) as calib_file:
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose
    return calib


def parse_poses(filename, calibration):
    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    with open(filename) as file:
        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def merge_trainval(save_dir, generate_instances):
    joint_db = [
        load_yaml(save_dir / (mode + "_database.yaml"))
        for mode in ["train", "validation"]
    ]
    save_yaml(save_dir / "trainval_database.yaml", joint_db)

    if generate_instances:
        joint_db = [
            load_yaml(save_dir / (mode + "_instances_database.yaml"))
            for mode in ["train", "validation"]
        ]
        save_yaml(save_dir / "trainval_instances_database.yaml", joint_db)
