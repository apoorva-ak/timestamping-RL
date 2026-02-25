# Audio Flamingo 3 — SFT Cold Start

LoRA supervised fine-tuning of AF3 to cold-start the model on timestamping.

## Usage

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --wandb_project af3-finetune
```

Training JSON is a list of `{"audio_path", "question", "answer"}` objects. Metrics are logged to W&B.
