import torch
from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num

def trans_vg_eval_miou(pred_masks, gt_masks, threshold=0.5):
    t_pred_masks = torch.ones(pred_masks.shape)
    t_pred_masks[pred_masks < threshold] = 0
    t_gt_masks = torch.ones(gt_masks.shape)
    t_gt_masks[gt_masks <= 0.0] = 0
    t_pred_masks = t_pred_masks.view(t_pred_masks.shape[0], -1)
    t_gt_masks = t_gt_masks.view(t_gt_masks.shape[0], -1)
    intersection = torch.sum(t_pred_masks * t_gt_masks, dim=1)
    union = torch.sum(t_pred_masks + t_gt_masks, dim=1) - intersection
    iou = intersection / union
    miou = torch.mean(iou)
    return miou

