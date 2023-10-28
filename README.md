# UrbanCLIP
Learning Text-Enhanced Urban Region Profiling from the Web

This repository is under development for better usage. The dataset will be released until paper notification, and is under refinement.

Stay tuned for more updates! 

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
```markdown
{
      "caption": "The image depicts a large, open field with a train track running through the middle of it",
      "image": "Beijing/16_12677_4730_s.jpg"
}
```

## Usage

```markdown
# Pretraining (example command line shown as follows)
CUDA_VISIBLE_DEVICES=7 python main.py --pretrained_model mscoco_finetuned_laion2B-s13B-b90k --dataset Beijing_captions --lr 0.00003 --batch_size 128 --epoch_num 100
```

```markdown
# Downstream Task1: Indicator prediction (example command line shown as follows)
CUDA_VISIBLE_DEVICES=7 python mlp.py --indicator carbon --dataset Beijing --test_file ./data/downstream_task/Beijing_test.csv --pretrained_model  ./checkpoints/BJ.bin
```

```markdown
# Downstream Task2: Location description generation (example command line shown as follows)
CUDA_VISIBLE_DEVICES=3 python caption.py --pretrained_model ./checkpoints/GZ_16/best_model.bin --dataset test
```
