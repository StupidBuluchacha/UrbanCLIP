import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
from model_init import UrbanCLIP_init

# There could be more options to initialize the parameters! The following checkpoint is one of them.
# Our design is based on CLIP. So CLIP variants are also within our scope. Welcome any commit for UrbanCLIP!
model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  # pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    pretrained="/root/autodl-tmp/laion-mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
    # pretrained="/root/autodl-tmp/laion-CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
)

# more general details of initialized model can be seen as follows:
vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    patch_dropout = 0.5  # https://arxiv.org/abs/2212.00794
)
vit = Extractor(vit, return_embeddings_only = True, detach = False)

urbanclip_init = UrbanCLIP_init(
    dim = 512,                     # model dimension
    img_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
    image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
    num_tokens = 20000,            # number of text tokens
    unimodal_depth = 6,            # depth of the unimodal transformer
    multimodal_depth = 6,          # depth of the multimodal transformer
    dim_head = 64,                 # dimension per attention head
    heads = 8,                     # number of attention heads
    caption_loss_weight = 1.,      # weight on the autoregressive caption loss
    contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
).cuda()

text = torch.randint(0, 20000, (4, 512)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

loss = urbanclip_init(
    text = text,
    images = images,
    return_loss = True  # set this to True to get the full caption + contrastive loss
)
loss.backward()

logits = urbanclip_init(
    text = text,
    images = images
) 

text_embeds, image_embeds = urbanclip_init(
    text = text,
    images = images,
    return_embeddings = True
)
