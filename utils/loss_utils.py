import torch
import torch.nn.functional as F

from utils.box_utils import xywh2xyxy, generalized_box_iou


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / batch_size


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / batch_size


def trans_vg_loss(batch_pred, batch_target, pred_masks=None, gt_masks=None, alpha=1):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    losses['loss_bbox'] = (loss_bbox.sum() / num_boxes) * alpha
    losses['loss_giou'] = (loss_giou.sum() / num_boxes) * alpha
    if pred_masks is not None and gt_masks is not None:
        focal_loss = sigmoid_focal_loss(pred_masks, gt_masks)
        dice_loss_ = dice_loss(pred_masks, gt_masks)
        losses['focal_loss'] = focal_loss
        losses['dice_loss'] = dice_loss_

    return losses
