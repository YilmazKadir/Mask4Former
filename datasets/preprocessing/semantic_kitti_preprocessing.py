import re
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from fire import Fire
from datasets.utils import (
    parse_calibration,
    parse_poses,
    load_yaml,
    save_database,
    merge_trainval,
)


class SemanticKittiPreprocessing:
    def __init__(
        self,
        data_dir: str = "/globalwork/data/SemanticKITTI/dataset",
        save_dir: str = "/globalwork/yilmaz/data/processed/semantic_kitti",
        generate_instances: bool = True,
        modes: tuple = ("train", "validation", "test"),
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.generate_instances = generate_instances
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if generate_instances:
            self.instances_dir = self.save_dir / "instances"
            if not self.instances_dir.exists():
                self.instances_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_yaml("conf/semantic-kitti.yaml")
        self.files = {}
        self.pose = {}

        for mode in self.modes:
            self.files[mode] = []
            self.pose[mode] = {}
            for sequence in sorted(self.config["split"][mode]):
                filepaths = list(self.data_dir.glob(f"*/{sequence:02}/velodyne/*bin"))
                filepaths = [str(file) for file in filepaths]
                calibration = parse_calibration(
                    Path(filepaths[0]).parent.parent / "calib.txt"
                )
                self.files[mode].extend(sorted(filepaths))
                self.pose[mode][sequence] = parse_poses(
                    Path(filepaths[0]).parent.parent / "poses.txt", calibration
                )

    def preprocess(self):
        for mode in self.modes:
            database = []
            instance_database = {}
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                database.append(filebase)
                if self.generate_instances and mode in ["train", "validation"]:
                    instances = self.extract_instance_from_file(filebase)
                    for instance in instances:
                        unique_identifier = (
                            f"{instance['sequence']}_{instance['panoptic_label']}"
                        )
                        if unique_identifier in instance_database:
                            instance_database[unique_identifier]["filepaths"].append(
                                instance["instance_filepath"]
                            )
                        else:
                            instance_database[unique_identifier] = {
                                "semantic_label": instance["semantic_label"],
                                "filepaths": [instance["instance_filepath"]],
                            }
            save_database(database, mode, self.save_dir)
            if self.generate_instances and mode in ["train", "validation"]:
                save_database(
                    list(instance_database.values()), f"{mode}_instances", self.save_dir
                )
        merge_trainval(self.save_dir, self.generate_instances)

    def process_file(self, filepath, mode):
        sequence, scan = re.search(
            r"/sequences/(\d{2})/velodyne/(\d{6})\.bin$", filepath
        ).group(1, 2)
        filebase = {
            "filepath": filepath,
            "sequence": int(sequence),
            "pose": self.pose[mode][int(sequence)][int(scan)].tolist(),
        }
        if mode in ["train", "validation"]:
            label_filepath = filepath.replace("velodyne", "labels").replace(
                "bin", "label"
            )
            filebase["label_filepath"] = label_filepath
        return filebase

    def extract_instance_from_file(self, filebase):
        points = np.fromfile(filebase["filepath"], dtype=np.float32).reshape(-1, 4)
        pose = np.array(filebase["pose"]).T
        points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
        label = np.fromfile(filebase["label_filepath"], dtype=np.uint32)
        sequence, scan = re.search(
            r"/sequences/(\d{2})/velodyne/(\d{6})\.bin$", filebase["filepath"]
        ).group(1, 2)
        file_instances = []
        for panoptic_label in np.unique(label):
            semantic_label = panoptic_label & 0xFFFF
            semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(
                semantic_label
            )
            if np.isin(semantic_label, range(1, 9)):
                instance_mask = label == panoptic_label
                instance_points = points[instance_mask, :]
                filename = f"{sequence}_{panoptic_label:010d}_{scan}.npy"
                instance_filepath = self.instances_dir / filename
                instance = {
                    "sequence": sequence,
                    "panoptic_label": f"{panoptic_label:010d}",
                    "instance_filepath": str(instance_filepath),
                    "semantic_label": semantic_label.item(),
                }
                np.save(instance_filepath, instance_points.astype(np.float32))
                file_instances.append(instance)
        return file_instances


if __name__ == "__main__":
    Fire(SemanticKittiPreprocessing)
