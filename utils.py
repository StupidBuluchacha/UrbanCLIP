import json
import random
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def count_trainable_parameters(model):
    """To compute the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """To compute the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def set_random_seed(seed):
    """To set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


class CoCaDataset(Dataset):
    def __init__(self, list_data=None, transform=None, tokenizer=None):
        super().__init__()

        self.transform = transform  # image transform for CoCa
        self.tokenizer = tokenizer  # tokenizer for CoCa

        self.img_paths = []
        self.img_tensors = []
        self.captions = []
        self.caption_tokens = []
        for item in list_data:
            _index = np.random.randint(
                0, len(item)
            )  # random select one caption for each image
            self.img_paths.append(os.path.join("./data/images", item[_index]["image"]))
            self.captions.append(item[_index]["caption"])
            im = Image.open(
                os.path.join("./data/images", item[_index]["image"])
            ).convert("RGB")
            # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
            im = transform(im)  # [3, 224, 224]
            self.img_tensors.append(im)
            self.caption_tokens.append(
                self.tokenizer(item[_index]["caption"])
            )  # [1, 77]
        # print(self.img_paths)  # ['./data/images/Beijing/16_12672_4745_s.jpg',]
        # print(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.img_tensors[index], self.caption_tokens[index]


class LinearProbDataset(Dataset):
    """Dataset for linear probe task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        data_name="Beijing",
        df_data=None,
        indicator="CO2",
        transform=None,
        mean=1.0,
        std=1.0,
        is_test=False,
    ):
        super().__init__()

        self.transform = transform  # image transform for CoCa

        # self.img_paths = []
        self.img_tensors = []
        self.y = []
        for idx, row in df_data.iterrows():
            _coordinate = eval(row["Coordinate"])  # tuple
            _image_name = "16_{}_{}_s.jpg".format(_coordinate[0], _coordinate[1])
            if data_name == "Beijing":
                _image_path = os.path.join("./data/images/Beijing", _image_name)
            elif data_name == "Shanghai":
                _image_path = os.path.join("./data/images/Shanghai", _image_name)
            else:
                raise ValueError("data must be Beijing or Shanghai")

            _im = Image.open(_image_path).convert("RGB")
            # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
            _im = transform(_im)  # [3, 224, 224]
            self.img_tensors.append(_im)
            if is_test:  # test set no real indicator value
                self.y.append(0.0)
            else:
                self.y.append((row[indicator] - mean) / std)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.img_tensors[index], np.float32(self.y[index])


class GenerationDataset(Dataset):
    """Dataset for text generation task.

    Args:
        data_name (str): name of dataset, Beijing or Shanghai
        df_data (DataFrame): dataframe of data
        indicator (str): indicator to predict, CO2, O3, SO2
        transform (torchvision.transforms): image transform for CoCa
        mean (float): mean of indicator values
        std (float): std of indicator values
        is_test (bool): whether this is test set
    """

    def __init__(
        self,
        jpg_list=None,
        transform=None,
    ):
        super().__init__()

        self.jpg_list = jpg_list
        self.transform = transform  # image transform for CoCa
        self.img_tensors = []
        for jpg_path in jpg_list:
            _im = Image.open(str(jpg_path)).convert("RGB")
            # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
            _im = transform(_im)  # [3, 224, 224]
            self.img_tensors.append(_im)

    def __len__(self):
        return len(self.img_tensors)

    def __getitem__(self, index):
        return self.img_tensors[index]


if __name__ == "__main__":
    data = json.load(open("data/captions/Beijing_captions.json", "r"))
    dataset = CoCaDataset(data)
    print(len(dataset))
