import numpy as np
import math


class LSTQ:
    def __init__(
        self,
        n_classes,
        ignore_label=255,
        stuff_cls_ids=[],
        offset=2**32,
        min_points=50,
    ):
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.things_cls_ids = [
            id for id in range(self.n_classes) if id not in stuff_cls_ids
        ]
        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = (
            min_points  # smallest number of points to consider instances in gt
        )
        self.eps = 1e-15

    def reset(self):
        # iou stuff
        self.px_iou_conf_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )
        self.sequences = []
        self.preds = {}
        self.gts = {}
        self.intersects = {}

    def addBatchSemIoU(self, x_sem, y_sem):
        # idxs are labels and predictions
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def getSemIoUStats(self):
        conf = self.px_iou_conf_matrix.copy().astype(np.float64)

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.float64) / union.astype(np.float64)
        iou_mean = iou.mean()

        return iou_mean, iou

    def update_dict_stat(self, stat_dict, unique_ids, unique_cnts):
        for uniqueid, counts in zip(unique_ids, unique_cnts):
            if uniqueid == 1:
                continue  # 1 -- no instance
            if uniqueid in stat_dict:
                stat_dict[uniqueid] += counts
            else:
                stat_dict[uniqueid] = counts

    def addBatchPanoptic4D(self, seq, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        if seq not in self.sequences:
            self.sequences.append(seq)
            self.preds[seq] = {}
            self.gts[seq] = [{} for i in range(self.n_classes)]
            self.intersects[seq] = [{} for i in range(self.n_classes)]

        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        preds = self.preds[seq]
        # generate the areas for each unique instance prediction (i.e., set1)
        unique_pred, counts_pred = np.unique(x_inst_row, return_counts=True)
        self.update_dict_stat(preds, unique_pred, counts_pred)

        for cl in range(self.n_classes):
            # Per-class accumulated stats
            cl_gts = self.gts[seq][cl]
            cl_intersects = self.intersects[seq][cl]

            # get a binary class mask (filter acc. to semantic class!)
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (mask-out everything but _this_ class)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance gt_np (i.e., set2)
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            self.update_dict_stat(
                cl_gts,
                unique_gt[counts_gt > self.min_points],
                counts_gt[counts_gt > self.min_points],
            )
            y_inst_in_cl[
                np.isin(y_inst_in_cl, unique_gt[counts_gt <= self.min_points])
            ] = 0

            # generate intersection using offset
            offset_combo = (
                x_inst_row[y_inst_in_cl > 0]
                + self.offset * y_inst_in_cl[y_inst_in_cl > 0]
            )
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            self.update_dict_stat(cl_intersects, unique_combo, counts_combo)

    def getPQ4D(self):
        pan_aq = np.zeros(self.n_classes, dtype=np.float64)
        pan_aq_ovr = 0.0
        num_tubes = np.zeros(self.n_classes, dtype=np.float64)

        for seq in self.sequences:
            preds = self.preds[seq]
            for cl in range(self.n_classes):
                cl_gts = self.gts[seq][cl]
                cl_intersects = self.intersects[seq][cl]
                outer_sum_iou = 0.0
                for gt_id, gt_size in cl_gts.items():
                    num_tubes[cl] += 1
                    inner_sum_iou = 0.0
                    for pr_id, pr_size in preds.items():
                        TPA_key = pr_id + self.offset * gt_id
                        if TPA_key in cl_intersects:
                            TPA_ovr = cl_intersects[TPA_key]
                            inner_sum_iou += TPA_ovr * (
                                TPA_ovr / (gt_size + pr_size - TPA_ovr)
                            )
                    outer_sum_iou += float(inner_sum_iou) / float(gt_size)
                pan_aq[cl] += outer_sum_iou
                pan_aq_ovr += outer_sum_iou

        AQ_overall = np.sum(pan_aq_ovr) / np.sum(num_tubes[self.things_cls_ids])
        AQ = pan_aq / np.maximum(num_tubes, self.eps)

        iou_mean, iou = self.getSemIoU()

        LSTQ = math.sqrt(AQ_overall * iou_mean)
        return LSTQ, AQ_overall, AQ, iou_mean, iou

    def addBatch(
        self, x_sem, x_inst, y_sem, y_inst, indices, seq
    ):  # x=preds, y=targets
        x_sem = x_sem[indices]
        x_inst = x_inst[indices]
        y_sem = y_sem[indices]
        y_inst = y_inst[indices]

        # only interested in points that are outside the void area (not in excluded classes)
        gt_not_in_excl_mask = y_sem != self.ignore_label
        # remove all other points
        x_sem = x_sem[gt_not_in_excl_mask]
        y_sem = y_sem[gt_not_in_excl_mask]
        x_inst = x_inst[gt_not_in_excl_mask]
        y_inst = y_inst[gt_not_in_excl_mask]

        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic4D(seq, x_sem, x_inst, y_sem, y_inst)
