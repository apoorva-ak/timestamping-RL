import argparse
import math
import os
import json
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb

from utils import MODEL_ID, build_conversation, make_collate_fn, print_trainable_parameters


OUTPUT_DIR = "./models-ft/af3-ft-timestamp"
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LR = 5e-5
WEIGHT_DECAY = 0.01
LORA_R = 128
LORA_DROPOUT = 0.05
LORA_ALPHA = 256
WANDB_PROJECT = "af3-finetune"


class LossHistoryCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "loss_history.json"
        self.train_losses, self.train_steps = [], []
        self.eval_losses, self.eval_steps = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append(float(logs["loss"]))
            self.train_steps.append(step)
        if "eval_loss" in logs:
            self.eval_losses.append(float(logs["eval_loss"]))
            self.eval_steps.append(step)
        with open(self.json_path, "w") as f:
            json.dump({
                "train_steps": self.train_steps, "train_loss": self.train_losses,
                "eval_steps": self.eval_steps, "eval_loss": self.eval_losses,
            }, f, indent=2)
        torch.cuda.empty_cache()
        return control


def load_json_dataset(json_path: str):
    ds = load_dataset("json", data_files=json_path, split="train")
    ds = ds.map(lambda x: {"audio": x["audio_path"]})
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def preprocess_dataset(json_path: str, system_prompt: str | None):
    print(f"Loading: {json_path}")
    ds = load_json_dataset(json_path)
    print(f"  {len(ds)} examples")

    print("  Computing audio durations...")
    def add_duration(example):
        try:
            import librosa
            return {"audio_duration": librosa.get_duration(path=example["audio_path"])}
        except Exception:
            return {"audio_duration": 0.0}

    ds = ds.map(add_duration, desc="Getting audio lengths")
    ds = ds.sort("audio_duration")

    durations = ds["audio_duration"]
    print(f"  Duration stats — min: {min(durations):.1f}s, max: {max(durations):.1f}s, "
          f"mean: {sum(durations)/len(durations):.1f}s")

    def preprocess(example, idx):
        return {
            "conversation": build_conversation(example, system_prompt=system_prompt),
            "idx": idx,
            "audio_path": example["audio_path"],
            "audio_duration": example["audio_duration"],
        }

    return ds.map(preprocess, with_indices=True, remove_columns=ds.column_names, desc="Building conversations")


def compute_save_steps(num_examples: int, batch_size: int, grad_accum: int) -> int:
    steps_per_epoch = max(1, math.ceil(math.ceil(num_examples / batch_size) / grad_accum))
    return max(1, math.ceil(steps_per_epoch * 0.1))


def create_trainer(train_dataset, eval_dataset, args):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, force_download=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    save_steps = compute_save_steps(len(train_dataset), args.per_device_batch_size, args.gradient_accumulation_steps)
    print(f"Saving checkpoints every {save_steps} steps (~0.1 epochs)")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.wandb_run_name,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=save_steps,
        per_device_eval_batch_size=args.per_device_batch_size,
        dataset_kwargs={"skip_prepare_dataset": True},
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=make_collate_fn(processor, debug_first_batch=True),
        processing_class=processor,
        callbacks=[LossHistoryCallback(output_dir=args.output_dir)],
    )
    return trainer, processor


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Audio Flamingo 3 (JSON + W&B)")
    p.add_argument("--train_json", type=str, required=True, help="Path to training JSON file")
    p.add_argument("--val_json", type=str, required=True, help="Path to validation JSON file")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--per_device_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--learning_rate", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--lora_r", type=int, default=LORA_R)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    wandb.init(
        project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity,
        config=vars(args) | {"model_id": MODEL_ID},
    )

    train_dataset = preprocess_dataset(args.train_json, args.system_prompt)
    val_dataset = preprocess_dataset(args.val_json, args.system_prompt)
    print(f"\nDataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    wandb.log({"train_size": len(train_dataset), "val_size": len(val_dataset)})

    trainer, processor = create_trainer(train_dataset, val_dataset, args)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
