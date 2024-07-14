# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import gc
import math
import os
import sys
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils


def train_one_epoch(args, model: torch.nn.Module, scaler, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0, alpha: float = 1.0, is_collect=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        gt_mask, pred_mask = None, None
        if args.is_segment:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        if gt_mask is not None:
            gt_mask = gt_mask.to(device)
        with autocast():
            # model forward
            stage_loss = None
            if args.is_segment:
                output, pred_mask = model(img_data, text_data)
            else:
                output = model(img_data, text_data)

            loss_dict = loss_utils.trans_vg_loss(output, target, pred_mask, gt_mask, alpha)
            if stage_loss is not None:
                loss_dict['stage_loss'] = stage_loss
            losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        # losses.backward()
        scaler.scale(losses).backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        gt_mask = None
        if args.is_segment:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        if gt_mask is not None:
            gt_mask = gt_mask.to(device)
        if args.is_segment:
            pred_boxes, pred_mask = model(img_data, text_data)
        else:
            pred_boxes = model(img_data, text_data)
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)
        if gt_mask is not None:
            mask_miou = eval_utils.trans_vg_eval_miou(pred_mask, gt_mask)
            metric_logger.update_v2('mask_miou', mask_miou, batch_size)
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    tot = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    tot_time = []

    pred_box_list = []
    gt_box_list = []
    tot_num = 0
    tot_iou = 0
    for _, batch in enumerate(tqdm(data_loader)):
        gt_mask = None
        if args.is_segment:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        if gt_mask is not None:
            output, pred_mask = model(img_data, text_data)
            tot_num += pred_mask.shape[0]
            mask_miou = eval_utils.trans_vg_eval_miou(pred_mask, gt_mask)
            tot_iou += mask_miou * pred_mask.shape[0]
        else:
            output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    # print(sum(tot_time) / len(tot_time))
    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num, tot_iou, tot_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    miou = float(result_tensor[2]) / float(result_tensor[3])

    if args.is_segment is not None:
        return accuracy, miou
    else:
        return accuracy


def clean_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
