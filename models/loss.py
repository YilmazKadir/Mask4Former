import torch
import torch.nn.functional as F
from torch import nn
from models.matcher import HungarianMatcher


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def box_loss(inputs: torch.Tensor, targets: torch.Tensor, num_bboxs: float):
    loss = F.l1_loss(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_bboxs


box_loss_jit = torch.jit.script(box_loss)  # type: torch.jit.ScriptModule


class LossOverall(nn.Module):
    def __init__(self, num_classes, w_class, w_mask, w_dice, w_box, no_object_coef):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = {
            "w_class": w_class,
            "w_mask": w_mask,
            "w_dice": w_dice,
            "w_box": w_box,
        }
        self.matcher = HungarianMatcher(self.loss_weights)

        ce_class_weights = torch.ones(num_classes + 1)
        ce_class_weights[-1] = no_object_coef
        self.register_buffer("ce_class_weights", ce_class_weights)

    def loss_classes(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.ce_class_weights,
            ignore_index=255,
        )
        return loss_ce

    def loss_masks(self, outputs, targets, indices):
        loss_masks = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id]["masks"][target_id].float()
            num_masks = target_mask.shape[0]

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
        return torch.sum(torch.stack(loss_masks))

    def loss_dices(self, outputs, targets, indices):
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id]["masks"][target_id].float()
            num_masks = target_mask.shape[0]

            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        return torch.sum(torch.stack(loss_dices))

    def loss_bboxs(self, outputs, targets, indices):
        loss_box = torch.zeros(1, device=outputs["pred_bboxs"].device)

        if self.loss_weights["w_box"] == 0:
            return loss_box

        for batch_id, (map_id, target_id) in enumerate(indices):
            pred_bboxs = outputs["pred_bboxs"][batch_id, map_id, :]
            target_bboxs = targets[batch_id]["bboxs"][target_id]
            target_classes = targets[batch_id]["labels"][target_id]
            keep_things = target_classes < 8
            if torch.any(keep_things):
                target_bboxs = target_bboxs[keep_things]
                pred_bboxs = pred_bboxs[keep_things]
                num_bboxs = target_bboxs.shape[0]
                loss_box += box_loss_jit(pred_bboxs, target_bboxs, num_bboxs)
        return loss_box

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {
            "loss_class": self.loss_classes(outputs, targets, indices)
            * self.loss_weights["w_class"],
            "loss_mask": self.loss_masks(outputs, targets, indices)
            * self.loss_weights["w_mask"],
            "loss_dice": self.loss_dices(outputs, targets, indices)
            * self.loss_weights["w_dice"],
            "loss_box": self.loss_bboxs(outputs, targets, indices)
            * self.loss_weights["w_box"],
        }

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                losses.update(
                    {
                        f"loss_class_{i}": self.loss_classes(
                            aux_outputs, targets, indices
                        )
                        * self.loss_weights["w_class"],
                        f"loss_mask_{i}": self.loss_masks(aux_outputs, targets, indices)
                        * self.loss_weights["w_mask"],
                        f"loss_dice_{i}": self.loss_dices(aux_outputs, targets, indices)
                        * self.loss_weights["w_dice"],
                        f"loss_box_{i}": self.loss_bboxs(aux_outputs, targets, indices)
                        * self.loss_weights["w_box"],
                    }
                )
        return losses
