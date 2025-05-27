import statistics
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import DBSCAN
from collections import defaultdict
from utils.utils import associate_instances, save_predictions, generate_logs


class Panoptic4D(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)

        self.criterion = hydra.utils.instantiate(config.loss)
        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        data, target = batch
        raw_coordinates = data.raw_coordinates

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device
        )
        losses = self.criterion(output, target)

        logs = generate_logs(losses, "train")
        self.log_dict(logs)

        loss = sum(losses.values())
        self.training_step_outputs.append(loss.cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        num_points = data.num_points
        sequences = data.sequences

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device, is_eval=True
        )

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"]
        offset_coords_idx = 0

        for logit, mask, map, label, n_point, seq in zip(
            pred_logits,
            pred_masks,
            inverse_maps,
            original_labels,
            num_points,
            sequences,
        ):
            if seq != self.last_seq:
                self.last_seq = seq
                self.previous_instances = None
                self.max_instance_id = self.config.model.num_queries
                self.scene = 0

            class_confidence, classes = torch.max(logit.detach().cpu(), dim=1)
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence = class_confidence[None, ...] * foreground_confidence
            confidence = confidence[map].numpy()

            ins_preds = np.argmax(confidence, axis=1)
            sem_preds = classes[ins_preds].numpy()
            ins_preds += 1
            ins_preds[np.isin(sem_preds, self.config.data.stuff_cls_ids)] = 0
            sem_labels = label[:, 0]
            _, ins_labels = self.validation_dataset.label_parser(label[:, 1])

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                curr_coords_idx = mask.shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[map].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx

                ins_ids = np.unique(ins_preds)
                for ins_id in ins_ids:
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[instance_mask])
                            .labels_
                        )
                        new_mask = np.zeros(ins_preds.shape, dtype=np.int64)
                        new_mask[instance_mask] = clusters + 1
                        for cluster_id in np.unique(new_mask):
                            if cluster_id != 0:
                                db_max_instance_id += 1
                                ins_preds[new_mask == cluster_id] = db_max_instance_id

            self.max_instance_id = max(db_max_instance_id, self.max_instance_id)
            for i in range(len(n_point) - 1):
                indices = range(n_point[i], n_point[i + 1])
                if i == 0 and self.previous_instances is not None:
                    current_instances = ins_preds[indices]
                    associations = associate_instances(
                        self.previous_instances, current_instances
                    )
                    for id in np.unique(ins_preds):
                        if associations.get(id) is None:
                            self.max_instance_id += 1
                            associations[id] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    self.class_evaluator.addBatch(
                        sem_preds, ins_preds, sem_labels, ins_labels, indices, seq
                    )
            if i > 0:
                self.previous_instances = ins_preds[indices]

        losses = self.criterion(output, target)
        logs = generate_logs(losses, "val")
        logs["val_loss_mean"] = sum(losses.values()).cpu().item()
        self.validation_step_outputs.append(logs)

    def test_step(self, batch, batch_idx):
        data, _ = batch
        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        num_points = data.num_points
        sequences = data.sequences

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device, is_eval=True
        )

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"]

        offset_coords_idx = 0

        for logit, mask, map, n_point, seq in zip(
            pred_logits, pred_masks, inverse_maps, num_points, sequences
        ):
            if seq != self.last_seq:
                self.last_seq = seq
                self.previous_instances = None
                self.max_instance_id = self.config.model.num_queries
                self.scene = 0
            class_confidence, classes = torch.max(logit.detach().cpu(), dim=1)
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence = class_confidence[None, ...] * foreground_confidence
            confidence = confidence[map].numpy()

            ins_preds = np.argmax(confidence, axis=1)
            sem_preds = classes[ins_preds].numpy() + 1
            ins_preds += 1
            ins_preds[np.isin(sem_preds, self.config.data.stuff_cls_ids)] = 0

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                curr_coords_idx = mask.shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[map].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx

                ins_ids = np.unique(ins_preds)
                for ins_id in ins_ids:
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[instance_mask])
                            .labels_
                        )
                        new_mask = np.zeros(ins_preds.shape, dtype=np.int64)
                        new_mask[instance_mask] = clusters + 1
                        for cluster_id in np.unique(new_mask):
                            if cluster_id != 0:
                                db_max_instance_id += 1
                                ins_preds[new_mask == cluster_id] = db_max_instance_id

            self.max_instance_id = max(db_max_instance_id, self.max_instance_id)
            for i in range(len(n_point) - 1):
                indices = range(n_point[i], n_point[i + 1])
                if i == 0 and self.previous_instances is not None:
                    current_instances = ins_preds[indices]
                    associations = associate_instances(
                        self.previous_instances, current_instances
                    )
                    for id in np.unique(ins_preds):
                        if associations.get(id) is None:
                            self.max_instance_id += 1
                            associations[id] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    save_predictions(
                        sem_preds[indices],
                        ins_preds[indices],
                        f"{seq:02}",
                        f"{self.scene:06}",
                    )
                    self.scene += 1
            if i > 0:
                self.previous_instances = ins_preds[indices]

        return {}

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        train_loss_mean = sum(self.training_step_outputs) / len(
            self.training_step_outputs
        )
        self.log_dict({"train_loss_mean": train_loss_mean}, sync_dist=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.last_seq = None
        class_names = self.config.data.class_names
        lstq, aq, all_aq, iou, all_iou = self.class_evaluator.getPQ4D()
        self.class_evaluator.reset()
        results = {}
        results["val_mean_aq"] = aq
        results["val_mean_iou"] = iou
        results["val_mean_lstq"] = lstq
        for i, (aq, iou) in enumerate(zip(all_aq, all_iou)):
            results[f"val_{class_names[i]}_aq"] = aq.item()
            results[f"val_{class_names[i]}_iou"] = iou.item()
        self.log_dict(results)

        losses = defaultdict(list)
        for output in self.validation_step_outputs:
            for key, val in output.items():
                losses[key].append(val)

        logs = {k: statistics.mean(v) for k, v in losses.items()}

        self.log_dict(logs)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
