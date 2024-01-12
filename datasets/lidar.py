import numpy as np
import volumentations as V
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Union
from random import random, choice, uniform


class LidarDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = "data/processed/semantic_kitti",
        mode: Optional[str] = "train",
        add_distance: Optional[bool] = False,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        instance_population: Optional[int] = 0,
        sweep: Optional[int] = 1,
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.ignore_label = ignore_label
        self.add_distance = add_distance
        self.instance_population = instance_population
        self.sweep = sweep
        self.config = self._load_yaml("conf/semantic-kitti.yaml")

        # loading database file
        database_path = Path(self.data_dir)
        if not (database_path / f"{mode}_database.yaml").exists():
            print(f"generate {database_path}/{mode}_database.yaml first")
            exit()
        self.data = self._load_yaml(database_path / f"{mode}_database.yaml")

        self.label_info = self._select_correct_labels(self.config["learning_ignore"])
        # augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")
        # reformulating in sweeps
        data = [[]]
        last_scene = self.data[0]["scene"]
        for x in self.data:
            if x["scene"] == last_scene:
                data[-1].append(x)
            else:
                last_scene = x["scene"]
                data.append([x])
        for i in range(len(data)):
            data[i] = list(self.chunks(data[i], sweep))
        self.data = [val for sublist in data for val in sublist]

        if instance_population > 0:
            self.instance_data = self._load_yaml(database_path / f"{mode}_instances_database.yaml")

    def chunks(self, lst, n):
        if "train" in self.mode or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i : i + n]
        else:
            for i in range(0, len(lst) - n + 1, n - 1):
                yield lst[i : i + n]
            if i != len(lst) - n:
                yield lst[i + n - 1 :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        coordinates_list = []
        features_list = []
        labels_list = []
        acc_num_points = [0]
        for time, scan in enumerate(self.data[idx]):
            points = np.fromfile(scan["filepath"], dtype=np.float32).reshape(-1, 4)
            coordinates = points[:, :3]
            # rotate and translate
            pose = np.array(scan["pose"]).T
            coordinates = coordinates @ pose[:3, :3] + pose[3, :3]
            coordinates_list.append(coordinates)
            acc_num_points.append(acc_num_points[-1] + len(coordinates))
            features = points[:, 3:4]
            time_array = np.ones((features.shape[0], 1)) * time
            features = np.hstack((time_array, features))
            features_list.append(features)
            if "test" in self.mode:
                labels = np.zeros_like(features).astype(np.int64)
                labels_list.append(labels)
            else:
                panoptic_label = np.fromfile(scan["label_filepath"], dtype=np.uint32)
                semantic_label = panoptic_label & 0xFFFF
                semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(semantic_label)
                labels = np.hstack((semantic_label[:, None], panoptic_label[:, None]))
                labels_list.append(labels)

        coordinates = np.vstack(coordinates_list)
        features = np.vstack(features_list)
        labels = np.vstack(labels_list)

        if "train" in self.mode and self.instance_population > 0:
            max_instance_id = np.amax(labels[:, 1])
            pc_center = coordinates.mean(axis=0)
            instance_c, instance_f, instance_l = self.populate_instances(
                max_instance_id, pc_center, self.instance_population
            )
            coordinates = np.vstack((coordinates, instance_c))
            features = np.vstack((features, instance_f))
            labels = np.vstack((labels, instance_l))

        if self.add_distance:
            center_coordinate = coordinates.mean(0)
            features = np.hstack(
                (
                    features,
                    np.linalg.norm(coordinates - center_coordinate, axis=1)[:, np.newaxis],
                )
            )

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        features = np.hstack((coordinates, features))

        labels[:, 0] = np.vectorize(self.label_info.__getitem__)(labels[:, 0])

        return {
            "num_points": acc_num_points,
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "sequence": scan["scene"],
        }

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def _select_correct_labels(self, learning_ignore):
        count = 0
        label_info = dict()
        for k, v in learning_ignore.items():
            if v:
                label_info[k] = self.ignore_label
            else:
                label_info[k] = count
                count += 1
        return label_info

    def _remap_model_output(self, output):
        inv_map = {v: k for k, v in self.label_info.items()}
        output = np.vectorize(inv_map.__getitem__)(output)
        return output

    def populate_instances(self, max_instance_id, pc_center, instance_population):
        coordinates_list = []
        features_list = []
        labels_list = []
        for _ in range(instance_population):
            instance_dict = choice(self.instance_data)
            idx = np.random.randint(len(instance_dict["filepaths"]))
            instance_list = []
            for time in range(self.sweep):
                if idx < len(instance_dict["filepaths"]):
                    filepath = instance_dict["filepaths"][idx]
                    instance = np.load(filepath)
                    time_array = np.ones((instance.shape[0], 1)) * time
                    instance = np.hstack((instance[:, :3], time_array, instance[:, 3:4]))
                    instance_list.append(instance)
                    idx = idx + 1
            instances = np.vstack(instance_list)
            coordinates = instances[:, :3] - instances[:, :3].mean(0)
            coordinates += pc_center + np.array([uniform(-10, 10), uniform(-10, 10), uniform(-1, 1)])
            features = instances[:, 3:]
            semantic_label = instance_dict["semantic_label"]
            labels = np.zeros_like(features, dtype=np.int64)
            labels[:, 0] = semantic_label
            max_instance_id = max_instance_id + 1
            labels[:, 1] = max_instance_id
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]
            coordinates_list.append(coordinates)
            features_list.append(features)
            labels_list.append(labels)
        return np.vstack(coordinates_list), np.vstack(features_list), np.vstack(labels_list)
