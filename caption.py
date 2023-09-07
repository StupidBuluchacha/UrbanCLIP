import math
import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from utils import (
    GenerationDataset,
    count_all_parameters,
    count_trainable_parameters,
    set_random_seed,
)
from tqdm import tqdm
from loguru import logger

"""
Fine-tuning CoCa on Beijing/Shanghai Captions dataset.

Note:
    Pre-training is very demanding in terms of data volume and not affordable for a typical lab
"""


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai"],
        help="which dataset",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/root/"
        + "laion-mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin",
        help="pretrained model, mscoco_finetuned_laion2B-s13B-b90k",
    )

    parser.add_argument(
        "--logging_dir", type=str, default="logs/downtask2", help="logging directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/downtask2",
        help="checkpoint path",
    )
    parser.add_argument(
        "--seed", type=int, default=133, help="random seed for reproducibility"
    )

    args = parser.parse_args()

    return args


def create_datasets(args, transform):
    """To create inference datasets."""
    if args.dataset == "Beijing":
        path = Path("data/images/Beijing")
        jpg_files = list(path.glob("*.jpg"))
    elif args.dataset == "Shanghai":
        path = Path("data/images/Shanghai")
        jpg_files = list(path.glob("*.jpg"))
    else:
        raise ValueError("dataset not found")

    # create datasets
    dataset = GenerationDataset(jpg_files, transform)

    return dataset


def inference(model, image_paths, dataloader, args, logger):
    """test on test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_text_strings = []
    with torch.no_grad():
        for i, images in tqdm(enumerate(dataloader)):
            images = images.to(device=device, non_blocking=True)
            generated = model.generate(images, generation_type="top_p")
            # print(type(generated))  # Tensor
            # print("generates.shape: {}".format(generated.shape))  # torch.Size([batch_size, 12])
            for j in range(generated.shape[0]):
                text = (
                    open_clip.decode(generated[j])
                    .split("<end_of_text>")[0]
                    .replace("<start_of_text>", "")
                )
                all_text_strings.append(text)
    df = pd.DataFrame(
        {
            "image": [str(item).split("/")[-1] for item in image_paths],
            "caption": all_text_strings,
        }
    )
    df.to_csv(os.path.join(args.checkpoint_dir, "captions.csv"), index=False)


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

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14", pretrained=args.pretrained_model
    )
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    logger.info("model parameters: {}".format(count_all_parameters(model)))
    logger.info(
        "model trainable parameters: {}".format(count_trainable_parameters(model))
    )
    # tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    # create datasets
    dataset = create_datasets(args, transform)
    logger.info("inference dataset size: {}".format(len(dataset)))

    # create dataloaders
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    inference(model, dataset.jpg_list, dataloader, args, logger)


if __name__ == "__main__":
    main()
