import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from utils import (
    LinearProbDataset,
    count_trainable_parameters,
    count_all_parameters,
    set_random_seed,
)
from tqdm import tqdm
from loguru import logger

"""
Frozen CoCa to liner probe

"""


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
        help="which dataset",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/downstream_task/Beijing_test.csv",
        help="test file path, if None then only train and val",
    )
    parser.add_argument(
        "--linear_probe", type=bool, default=True, help="training if True else testing"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        default="carbon",
        choices=["carbon", "population", "gdp"],
        help="indicator",
    )
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument(
        "--drop_out", type=float, default=0.01, help="dropout in linear probe"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
    parser.add_argument(
        "--log_every_n_steps", type=int, default=100, help="log every n steps"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="./checkpoints/best_model.bin",
        help="pretrained model after running main.py",
    )
    parser.add_argument(
        "--img_embedding_dim", type=int, default=768, help="image encoder output dim"
    )
    parser.add_argument("--seed", type=int, default=132, help="random seed")
    parser.add_argument(
        "--logging_dir", type=str, default="logs/downtask1", help="logging directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/downtask1",
        help="checkpoint path",
    )
    # MLP parameters
    parser.add_argument(
        "--project_dim", type=int, default=256, help="project dimension"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="activation function",
    )

    # data hyper-parameters
    parser.add_argument(
        "--train_dataset_ratio",
        type=float,
        default=0.8,
        help="ratio of training dataset",
    )
    # parser.add_argument("--val_dataset_ratio", type=float, default=0.2,
    #                     help="ratio of validation dataset")
    args = parser.parse_args()

    return args


class CoCaLinearProbe(nn.Module):
    def __init__(self, coca_model, args):
        super().__init__()
        self.coca = coca_model
        self.project = nn.Linear(args.img_embedding_dim, args.project_dim)
        self.activation = nn.ReLU() if args.activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(args.drop_out)
        self.predict = nn.Linear(args.project_dim, 1)

    def forward(self, image_features):
        image_latent = self.coca.encode_image(image_features)
        image_latent = self.project(image_latent)
        image_latent = self.activation(image_latent)
        image_latent = self.dropout(image_latent)
        logits = self.predict(image_latent)
        return logits.squeeze(1)


def create_datasets(args, transform):
    """To create train, val, test datasets."""
    if args.dataset == "Beijing":
        data = pd.read_csv("data/downstream_task/Beijing_train.csv")
    elif args.dataset == "Shanghai":
        data = pd.read_csv("data/downstream_task/Shanghai_train.csv")
    elif args.dataset == "Shenzhen":
        data = pd.read_csv("data/downstream_task/Shenzhen_train.csv")
    elif args.dataset == "Guangzhou":
        data = pd.read_csv("data/downstream_task/Guangzhou_train.csv")
    else:
        raise ValueError("dataset not found")

    # split dataset into train, val, test
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle rows
    train_data = data[: int(len(data) * args.train_dataset_ratio)].reset_index(
        drop=True
    )
    val_data = data[int(len(data) * args.train_dataset_ratio) :].reset_index(drop=True)
    mean = np.mean(train_data[args.indicator])
    std = np.std(train_data[args.indicator])
    # create datasets
    train_dataset = LinearProbDataset(
        args.dataset, train_data, args.indicator, transform, mean, std, False
    )
    val_dataset = LinearProbDataset(
        args.dataset, val_data, args.indicator, transform, mean, std, False
    )

    if args.test_file is not None:
        test_data = pd.read_csv(args.test_file)
        test_dataset = LinearProbDataset(
            args.dataset, test_data, args.indicator, transform, mean, std, True
        )
        return train_dataset, val_dataset, test_dataset, mean, std
    else:
        return train_dataset, val_dataset, None, mean, std


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, data, epoch, optimizer, args, logger):
    """To train one epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    dataloader = data["train_loader"]
    num_batches_per_epoch = len(dataloader)
    sample_digits = math.ceil(math.log(len(dataloader) * args.batch_size + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()  # data loading time
    end = time.time()
    for batch_count, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + batch_count

        (
            images,
            y,
        ) = batch  # images: [batch_size, 3, 224, 224], y: [batch_size]
        images = images.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        # print("y.shape: {}".format(y.shape))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        predicts = model(images)
        # print("predicts.shape: {}".format(predicts.shape))
        # print("y.shape: {}".format(y.shape))

        loss = criterion(predicts, y)

        loss.backward()

        # if args.grad_clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count += 1
        if step % args.log_every_n_steps == 0:
            batch_size = len(images)
            num_samples = step * batch_size
            samples_per_epoch = (
                num_batches_per_epoch * batch_size
            )  # sample size per epoch
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled
            for key in ["mse", "r2", "rmse", "mae", "mape"]:
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
            losses_m["mse"].update(loss.item(), batch_size)
            losses_m["r2"].update(
                r2_score(y.cpu().numpy(), predicts.detach().cpu().numpy()), batch_size
            )
            losses_m["rmse"].update(
                np.sqrt(
                    mean_squared_error(y.cpu().numpy(), predicts.detach().cpu().numpy())
                ),
                batch_size,
            )
            losses_m["mae"].update(
                mean_absolute_error(y.cpu().numpy(), predicts.detach().cpu().numpy()),
                batch_size,
            )
            losses_m["mape"].update(
                mean_absolute_percentage_error(
                    y.cpu().numpy(), predicts.detach().cpu().numpy()
                ),
                batch_size,
            )

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = batch_size / batch_time_m.val
            logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                # f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Metrics: " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                # "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                logger.info({name: val, "step": step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, logger):
    """To evaluate on val dataset."""
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["val_loader"]
    all_y, all_predicts = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, y = batch
            images = images.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            y_hat = model(images)

            all_y.append(y.cpu().numpy())
            all_predicts.append(y_hat.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)

    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
    logger.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    # for name, val in metrics.items():
    #     logger.info({f"val/{name}": val, "epoch": epoch})

    return metrics


def inference(model, data, args, logger):
    """test on test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["test_loader"]
    all_predicts = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, y = batch
            images = images.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            y_hat = model(images)

            # all_y.append(y.cpu().numpy())
            all_predicts.append(y_hat.cpu().numpy())
    # all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)

    # metrics["mse"] = mean_squared_error(all_y, all_predicts)
    # metrics["r2"] = r2_score(all_y, all_predicts)
    # metrics["rmse"] = np.sqrt(metrics["mse"])
    # metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    # metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
    # logger.info(
    #     f"Test: " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    # )

    y_hat = [item * data["std"] + data["mean"] for item in all_predicts]
    test_data = pd.read_csv(args.test_file)
    test_data[args.indicator + "_predict"] = y_hat
    test_data.to_csv(args.test_file[:-4] + "_predicted.csv", index=False)
    # return metrics


def main():
    args = create_args()
    set_random_seed(args.seed)
    # create logger
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logger.remove(handler_id=None)  # remove default logger
    logger.add(os.path.join(args.logging_dir, str(args.seed) + ".log"), level="INFO")
    logger.info(args)

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coca_model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14", pretrained=args.pretrained_model
    )
    model = CoCaLinearProbe(coca_model, args)
    model.to(device)

    for param in model.coca.parameters():
        param.requires_grad = False
    logger.info("model parameters: {}".format(count_all_parameters(model)))
    logger.info(
        "model trainable parameters: {}".format(count_trainable_parameters(model))
    )
    # tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    # create datasets
    train_dataset, val_dataset, test_dataset, mean, std = create_datasets(
        args, transform
    )
    logger.info("train dataset size: {}".format(len(train_dataset)))
    logger.info("val dataset size: {}".format(len(val_dataset)))
    logger.info("test dataset size: {}".format(len(test_dataset)))
    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    data = {}
    data["train_loader"] = train_dataloader
    data["val_loader"] = val_dataloader
    data["test_loader"] = test_dataloader
    data["mean"] = mean
    data["std"] = std

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    best_mse_val_loss = float("inf")
    for epoch in tqdm(range(args.epoch_num), desc="Training"):
        logger.info("Start epoch {}".format(epoch))

        train_one_epoch(model, criterion, data, epoch, optimizer, args, logger)
        completed_epoch = epoch + 1

        cur_metrics = evaluate(model, data, completed_epoch, args, logger)

        # Saving checkpoints.
        # if args.save_logs:
        # TODO maybe we should only save best checkpoints
        checkpoint_dict = {
            "epoch": completed_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cur_metrics["mse"] < best_mse_val_loss:
            torch.save(
                checkpoint_dict,
                # os.path.join(args.checkpoint_dir, f"epoch_{completed_epoch}.pt"),
                os.path.join(args.checkpoint_dir, "best.pt"),
            )
            best_mse_val_loss = cur_metrics["mse"]

    best_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, "best.pt"), map_location=torch.device("cpu")
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    model.to(device)
    if args.test_file is not None:
        inference(model, data, args, logger)


if __name__ == "__main__":
    main()
