import torch
import numpy as np


class VoxelizeCollate:
    def __init__(self, ignore_label=255, voxel_size=1):
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label

    def __call__(self, batch):
        (
            coordinates,
            features,
            labels,
            original_labels,
            inverse_maps,
            num_points,
            sequences,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i, sample in enumerate(batch):
            original_labels.append(sample["labels"])
            num_points.append(sample["num_points"])
            sequences.append(sample["sequence"])
            sample_c, sample_f, sample_l, inverse_map = voxelize(
                sample["coordinates"],
                sample["features"],
                sample["labels"],
                self.voxel_size,
            )
            inverse_maps.append(inverse_map)
            coordinates.append(
                torch.hstack((torch.ones((sample_c.shape[0], 1)) * i, sample_c))
            )
            features.append(sample_f)
            labels.append(sample_l)

        # Concatenate all lists
        target = generate_target(features, labels, self.ignore_label)
        coordinates = torch.vstack(coordinates).int()
        features = torch.vstack(features)
        raw_coordinates = features[:, :4]
        features = features[:, 4:]

        return (
            NoGpu(
                coordinates,
                features,
                raw_coordinates,
                original_labels,
                inverse_maps,
                num_points,
                sequences,
            ),
            target,
        )


def voxelize(coordinates, features, labels, voxel_size):
    coordinates = np.floor(coordinates / voxel_size).astype(np.int32)
    coordinates = coordinates - np.min(coordinates, axis=0)

    _, unique_map, inverse_map = np.unique(
        coordinates, return_index=True, return_inverse=True, axis=0
    )

    sample_c = torch.from_numpy(coordinates[unique_map])
    sample_f = torch.from_numpy(features[unique_map])
    sample_l = torch.from_numpy(labels[unique_map])
    return sample_c, sample_f, sample_l, inverse_map


def generate_target(features, labels, ignore_label):
    target = []

    for feat, lb in zip(features, labels):
        raw_coords = feat[:, :3]
        raw_coords = (raw_coords - raw_coords.min(0)[0]) / (
            raw_coords.max(0)[0] - raw_coords.min(0)[0]
        )
        mask_labels = []
        binary_masks = []
        bboxs = []

        panoptic_labels = lb[:, 1].unique()
        for panoptic_label in panoptic_labels:
            mask = lb[:, 1] == panoptic_label

            sem_labels = lb[mask, 0]
            if sem_labels[0] != ignore_label:
                mask_labels.append(sem_labels[0])
                binary_masks.append(mask)
                mask_coords = raw_coords[mask, :]
                bboxs.append(
                    torch.hstack(
                        (
                            mask_coords.mean(0),
                            mask_coords.max(0)[0] - mask_coords.min(0)[0],
                        )
                    )
                )

        if len(mask_labels) != 0:
            mask_labels = torch.stack(mask_labels)
            binary_masks = torch.stack(binary_masks)
            bboxs = torch.stack(bboxs)
            target.append(
                {"labels": mask_labels, "masks": binary_masks, "bboxs": bboxs}
            )

    return target


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        raw_coordinates,
        original_labels=None,
        inverse_maps=None,
        num_points=None,
        sequences=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.raw_coordinates = raw_coordinates
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.num_points = num_points
        self.sequences = sequences
