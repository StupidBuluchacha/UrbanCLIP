# UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web [WWW 2024]
This repo is the implementation of our manuscript entitled [UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web](https://arxiv.org/abs/2310.18340) (Accepted by the Web Conference 2024). 

This repository will be kept under development for better usage. The dataset is under refinement (Part of the current data could be pseudo-data for testing only), but our team has also released a toolkit named [UrbanCLIP Dataset Toolkit](https://github.com/siruzhong/UrbanCLIP-Dataset-Toolkit),  a comprehensive tool chain designed to facilitate the collection, processing, and integration of satellite imagery and associated metadata for urban analysis. 

Stay tuned for more updates! 

**【NEWS！】** Our team extended our work to a more comprehensive scope. More details can be found in the paper entitled [UrbanVLP: A Multi-Granularity Vision-Language Pre-Trained Foundation Model for Urban Indicator Prediction](https://arxiv.org/abs/2403.16831), where we will release the dataset and code base soon.

**【NEWS！】** Our team investigated the [Deep Learning for Cross-Domain Data Fusion in Urban Computing: Taxonomy, Advances, and Outlook](https://arxiv.org/abs/2402.19348). Welcome any feedback!

## Data Directory
```markdown
data/
├── captions/
|   ├── Beijing_captions.json # image-text pairs
|   ├── Shanghai_captions.json
|   ├── Guangzhou_captions.json
|   ├── Shenzhen_captions.json
└── downstream_task/
|   ├── downstream.csv # downstream task data
└── images/ # image data
|   ├── Beijing
|       ├── 16_12672_4745_s.jpg
|       ├── 16_12677_4730_s.jpg
|   ├── Shanghai
|   ├── Guangzhou
|   ├── Shenzhen

```

## Data Example
Garbage in, garbage out! Please spend more time on data double-checking, cleaning, and refinement!
```markdown
{
      "caption": "The image depicts a large, open field with a train track running through the middle of it",
      "image": "Beijing/16_12677_4730_s.jpg"
}
```

## Usage

```markdown
# Pretraining (example command line shown as follows)
CUDA_VISIBLE_DEVICES=7 python main.py --pretrained_model mscoco_finetuned_laion2B-s13B-b90k --dataset Beijing_captions --lr XXX --batch_size XXX --epoch_num XXX
```

```markdown
# Downstream Task1: Indicator prediction (example command line shown as follows)
CUDA_VISIBLE_DEVICES=7 python mlp.py --indicator carbon --dataset Beijing --test_file ./data/downstream_task/Beijing_test.csv --pretrained_model  ./checkpoints/BJ.bin
```

```markdown
# Downstream Task2: Location description generation (example command line shown as follows)
CUDA_VISIBLE_DEVICES=3 python caption.py --pretrained_model ./checkpoints/GZ_16/best_model.bin --dataset XXX
```
