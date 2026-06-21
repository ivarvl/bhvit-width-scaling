# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma, accuracy

import utils


def train_mix_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    writer=None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    base = SoftTargetCrossEntropy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "base_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "distillation_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            base_loss = base(outputs.logits, targets)
            with torch.no_grad():
                teacher_out = teacher_model(samples)
                teacher_outputs = (
                    teacher_out.logits
                    if hasattr(teacher_out, "logits")
                    else teacher_out
                )
                distillation_loss = criterion(outputs.logits, teacher_outputs)
        loss = base_loss * (1 - 0.9) + distillation_loss * 0.9
        loss_value = loss.item()
        base_loss_value = base_loss.item()
        distillation_loss_value = distillation_loss.item()
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(base_loss=base_loss_value)
        metric_logger.update(distillation_loss=distillation_loss_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar("train/base_loss_step", base_loss_value, global_step)
            writer.add_scalar(
                "train/distillation_loss_step", distillation_loss_value, global_step
            )
            writer.add_scalar(
                "train/lr_step", optimizer.param_groups[0]["lr"], global_step
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_L1(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    writer=None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            regularization_loss = 0
            n = 0
            for name, parameters in model.named_parameters():
                regular = (
                    "cov1.weight" in name
                    or "cov2.weight" in name
                    or "cov3.weight" in name
                    or "dense.weight" in name
                    or "query.weight" in name
                    or "key.weight" in name
                    or "value.weight" in name
                )
                if regular:
                    n = n + 1
                    regularization_loss += (
                        torch.sum(torch.abs(torch.abs(parameters) - 1.0))
                        / parameters.numel()
                    )
            if n != 0:
                regularization_loss = regularization_loss / n
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_out = teacher_model(samples)
                    teacher_outputs = (
                        teacher_out.logits
                        if hasattr(teacher_out, "logits")
                        else teacher_out
                    )
                loss1 = criterion(outputs.logits, teacher_outputs)
                loss = 0.9 * loss1 + 0.1 * regularization_loss

            else:
                loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar(
                "train/lr_step", optimizer.param_groups[0]["lr"], global_step
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch2(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    writer=None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar(
                "train/lr_step", optimizer.param_groups[0]["lr"], global_step
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    writer=None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if teacher_model is not None and mixup_fn is not None:
                with torch.no_grad():
                    teacher_out = teacher_model(samples)
                    teacher_outputs = (
                        teacher_out.logits
                        if hasattr(teacher_out, "logits")
                        else teacher_out
                    )
                loss1 = criterion(outputs.logits, teacher_outputs)
                loss2 = criterion2(outputs.logits, targets)
                loss = 0.8 * loss1 + 0.2 * loss2

            elif teacher_model is not None and mixup_fn is None:
                with torch.no_grad():
                    teacher_out = teacher_model(samples)
                    teacher_outputs = (
                        teacher_out.logits
                        if hasattr(teacher_out, "logits")
                        else teacher_out
                    )
                loss = criterion(outputs.logits, teacher_outputs)

            else:
                loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar(
                "train/lr_step", optimizer.param_groups[0]["lr"], global_step
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_L1_fixed(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    writer=None,
    total_epochs: int = 300,
    lambda_distill: float = 0.8,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    # Paper eq. 14: beta=0.1 only in final 10% of training, else 0
    beta = 0.1 if epoch >= 0.9 * total_epochs else 0.0
    lam = lambda_distill  # λ=0.8 per paper Table 6
    alpha = 1.0 - lam - beta  # = 0.1 when both active, 0.2 before reg kicks in

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            raw_out = model(samples)
            # Handle both plain tensor and HuggingFace-style output
            outputs = raw_out.logits if hasattr(raw_out, "logits") else raw_out

            # Regularization loss (L_re): mean of ||w| - 1| over binary weight layers
            regularization_loss = torch.tensor(0.0, device=device)
            if beta > 0.0:  # skip computation entirely before final 10%
                n = 0
                for name, parameters in model.named_parameters():
                    is_binary_weight = any(
                        k in name
                        for k in [
                            "cov1.weight",
                            "cov2.weight",
                            "cov3.weight",
                            "dense.weight",
                            "query.weight",
                            "key.weight",
                            "value.weight",
                        ]
                    )
                    if is_binary_weight:
                        n += 1
                        regularization_loss = regularization_loss + (
                            torch.abs(torch.abs(parameters) - 1.0).mean()
                        )
                if n > 0:
                    regularization_loss = regularization_loss / n

            # Build loss per paper eq. 14: (1-λ-β)*L_cls + λ*L_dis + β*L_re
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_raw = teacher_model(samples)
                    teacher_outputs = (
                        teacher_raw.logits
                        if hasattr(teacher_raw, "logits")
                        else teacher_raw
                    )

                loss_dis = criterion(outputs, teacher_outputs)  # DistributionLoss

                if criterion2 is not None and alpha > 0.0:
                    # criterion2 is CrossEntropyLoss; targets may be soft if mixup ran
                    loss_cls = criterion2(outputs, targets)
                    loss = (
                        alpha * loss_cls + lam * loss_dis + beta * regularization_loss
                    )
                else:
                    # No classification term (e.g. pure distillation without mixup)
                    loss = lam * loss_dis + beta * regularization_loss
            else:
                loss_cls = criterion(outputs, targets)
                loss = loss_cls + beta * regularization_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar(
                "train/lr_step", optimizer.param_groups[0]["lr"], global_step
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.logits, target)

        acc1, acc5 = accuracy(output.logits, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
