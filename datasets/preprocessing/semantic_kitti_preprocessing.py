import re
import numpy as np
import yaml
from pathlib import Path
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from fire import Fire


class SemanticKittiPreprocessing:
    def __init__(
        self,
        data_dir: str = "/globalwork/data/SemanticKITTI/dataset",
        save_dir: str = "/globalwork/yilmaz/data/processed/semantic_kitti",
        modes: tuple = ("train", "validation", "test"),
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        self.config = self._load_yaml("conf/semantic-kitti.yaml")
        self.create_label_database("conf/semantic-kitti.yaml")
        self.pose = dict()

        for mode in self.modes:
            scene_mode = "valid" if mode == "validation" else mode
            self.pose[mode] = dict()
            for scene in sorted(self.config["split"][scene_mode]):
                filepaths = list(self.data_dir.glob(f"*/{scene:02}/velodyne/*bin"))
                filepaths = [str(file) for file in filepaths]
                self.files[mode].extend(natsorted(filepaths))
                calibration = parse_calibration(Path(filepaths[0]).parent.parent / "calib.txt")
                self.pose[mode].update(
                    {
                        scene: parse_poses(
                            Path(filepaths[0]).parent.parent / "poses.txt",
                            calibration,
                        ),
                    }
                )

    def preprocess(self):
        for mode in self.modes:
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                database.append(filebase)
            self.save_database(database, mode)
        self.joint_database()

    def make_instance_database(self):
        train_database = self._load_yaml(self.save_dir / "train_database.yaml")
        instance_database = {}
        for sample in tqdm(train_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "train_instances")

        validation_database = self._load_yaml(self.save_dir / "validation_database.yaml")
        for sample in tqdm(validation_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "trainval_instances")

    def extract_instance_from_file(self, sample):
        points = np.fromfile(sample["filepath"], dtype=np.float32).reshape(-1, 4)
        pose = np.array(sample["pose"]).T
        points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
        label = np.fromfile(sample["label_filepath"], dtype=np.uint32)
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", sample["filepath"]).group(1, 2)
        file_instances = []
        for panoptic_label in np.unique(label):
            semantic_label = panoptic_label & 0xFFFF
            semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(semantic_label)
            if np.isin(semantic_label, range(1, 9)):
                instance_mask = label == panoptic_label
                instance_points = points[instance_mask, :]
                filename = f"{scene}_{panoptic_label:010d}_{sub_scene}.npy"
                instance_filepath = self.save_dir / "instances" / filename
                instance = {
                    "scene": scene,
                    "sub_scene": sub_scene,
                    "panoptic_label": f"{panoptic_label:010d}",
                    "instance_filepath": str(instance_filepath),
                    "semantic_label": semantic_label.item(),
                }
                if not instance_filepath.parent.exists():
                    instance_filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(instance_filepath, instance_points.astype(np.float32))
                file_instances.append(instance)
        return file_instances

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def joint_database(self, train_modes=["train", "validation"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "trainval_database.yaml", joint_db)

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(file, f, default_style=None, default_flow_style=False)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def create_label_database(self, config_file):
        if (self.save_dir / "label_database.yaml").exists():
            return self._load_yaml(self.save_dir / "label_database.yaml")
        config = self._load_yaml(config_file)
        label_database = {}
        for key, old_key in config["learning_map_inv"].items():
            label_database.update(
                {
                    key: {
                        "name": config["labels"][old_key],
                        "color": config["color_map"][old_key][::-1],
                        "validation": not config["learning_ignore"][key],
                    }
                }
            )

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", filepath).group(1, 2)
        sample = {
            "filepath": filepath,
            "scene": int(scene),
            "pose": self.pose[mode][int(scene)][int(sub_scene)].tolist(),
        }

        if mode in ["train", "validation"]:
            # getting label info
            label_filepath = filepath.replace("velodyne", "labels").replace("bin", "label")
            sample["label_filepath"] = label_filepath
        return sample


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


if __name__ == "__main__":
    Fire(SemanticKittiPreprocessing)
