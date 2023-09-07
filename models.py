import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  # pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    pretrained="/root/autodl-tmp/laion-mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
    # pretrained="/root/autodl-tmp/laion-CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
)
