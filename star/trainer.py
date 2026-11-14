import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from star.loss.classification import cal_loss1, cal_loss2
import torch.nn as nn
from itertools import islice


def train(
    train_config,
    model,
    dataloader,
    loss_functions,
    optimizer,
    scheduler=None,
    scaler=None,
):

    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    max_batches = getattr(train_config, "max_train_batches", 0)
    batches = islice(dataloader, max_batches) if max_batches else dataloader
    if train_config.verbose:
        bar = tqdm(
            batches,
            total=min(len(dataloader), max_batches) if max_batches else len(dataloader),
        )
    else:
        bar = batches

    criterion = nn.CrossEntropyLoss()

    # for loop over one epoch
    for query, reference, ids, labels in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(train_config.device)
                # reference = [img.to(train_config.device) for img in reference]
                reference = torch.stack(reference, dim=1).to(train_config.device)
                labels = labels.to(train_config.device)
                batch_size, seq_len, c, h, w = reference.shape
                reference = reference.view(
                    batch_size * seq_len, c, h, w
                )  # 展平序列维度

                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    output1, output2 = model(query, reference)
                    features1, features2 = (
                        output1[-2],
                        output2[-2],
                    )  # -- for contrastive
                    features2 = features2.view(batch_size, seq_len, -1)  # 恢复序列维度
                    # 聚合序列特征（这里使用平均池化作为示例）
                    features2 = features2.mean(dim=1)
                    features_cls_1, features_cls_2 = (
                        output1[1],
                        output2[1],
                    )  # -- for classifier
                    features_vsam_1, features_vsam_2 = output1[0], output2[0]
                    alignment_shape = features_vsam_2.shape[1:]
                    features_vsam_2 = features_vsam_2.view(
                        batch_size, seq_len, *alignment_shape
                    ).mean(dim=1)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_functions["infoNCE"](
                        features1, features2, model.module.logit_scale.exp()
                    )

                    loss_cls = (
                        0.6
                        * cal_loss1(
                            features_cls_1[0], labels, criterion, batch_size, seq_len
                        )
                        + 0.4
                        * cal_loss1(
                            features_cls_1[1], labels, criterion, batch_size, seq_len
                        )
                        + cal_loss1(
                            features_cls_1[2], labels, criterion, batch_size, seq_len
                        )
                        + 0.6
                        * cal_loss2(
                            features_cls_2[0], labels, criterion, batch_size, seq_len
                        )
                        + 0.4
                        * cal_loss2(
                            features_cls_2[1], labels, criterion, batch_size, seq_len
                        )
                        + cal_loss2(
                            features_cls_2[2], labels, criterion, batch_size, seq_len
                        )
                    )

                    loss_VSAM = loss_functions["VSAM_loss"](
                        features_vsam_1,
                        features_vsam_2,
                        model.module.logit_scale_blocks.exp(),
                    )

                else:
                    loss = loss_functions["infoNCE"](
                        features1, features2, model.logit_scale.exp()
                    )

                    loss_cls = (
                        0.6
                        * cal_loss1(
                            features_cls_1[0], labels, criterion, batch_size, seq_len
                        )
                        + 0.4
                        * cal_loss1(
                            features_cls_1[1], labels, criterion, batch_size, seq_len
                        )
                        + cal_loss1(
                            features_cls_1[2], labels, criterion, batch_size, seq_len
                        )
                        + 0.6
                        * cal_loss2(
                            features_cls_2[0], labels, criterion, batch_size, seq_len
                        )
                        + 0.4
                        * cal_loss2(
                            features_cls_2[1], labels, criterion, batch_size, seq_len
                        )
                        + cal_loss2(
                            features_cls_2[2], labels, criterion, batch_size, seq_len
                        )
                    )

                    loss_VSAM = loss_functions["VSAM_loss"](
                        features_vsam_1, features_vsam_2, model.logit_scale_blocks.exp()
                    )

                lossall = loss + 0.5 * loss_VSAM + 0.05 * loss_cls
                if not torch.isfinite(lossall):
                    raise FloatingPointError(
                        "Non-finite training loss at batch {}".format(step)
                    )

                losses.update(lossall.item())

            scaler.scale(lossall).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), train_config.clip_grad
                )

            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if (
                train_config.scheduler == "polynomial"
                or train_config.scheduler == "cosine"
                or train_config.scheduler == "constant"
            ):
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](
                    features1, features2, model.module.logit_scale.exp()
                )
            else:
                loss = loss_functions["infoNCE"](
                    features1, features2, model.logit_scale.exp()
                )
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), train_config.clip_grad
                )

            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if (
                train_config.scheduler == "polynomial"
                or train_config.scheduler == "cosine"
                or train_config.scheduler == "constant"
            ):
                scheduler.step()

        if train_config.verbose:

            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]["lr"]),
            }

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    from .evaluate.features import extract_features

    return extract_features(train_config, model, dataloader)


def predict_seq_D2S(train_config, model, dataloader):
    from .evaluate.features import extract_features

    return extract_features(train_config, model, dataloader, sequence=True)
