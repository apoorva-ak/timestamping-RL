import json
import re
from typing import Any, Dict, List, Optional
import os
import torch
from transformers import AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"

__all__ = [
    "MODEL_ID",
    "build_conversation",
    "build_prompt_only_conversation",
    "make_collate_fn",
    "print_trainable_parameters",
    "safe_parse_prediction",
]


def build_conversation(
    row: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a single-turn conversation from a CSV row.

    Required CSV columns:
        - audio_path: full path to the audio file
        - question: the question/prompt text
        - answer: the response text
    
    The row will also have an 'audio' key added by the dataset loading
    which contains the Audio feature with 'path' attribute.
    
    Args:
        row: dict containing the CSV row data plus the audio feature
        system_prompt: optional system message
    
    Returns:
        List of conversation turns in the expected format
    """
    # Get audio path from the Audio feature (set by datasets library)
    audio_path = row["audio"]["path"]
    
    # Get question and answer from CSV columns
    question_text = row["question"]
    answer_text = row["answer"]
    
    conversation = []

    if system_prompt:
        conversation.append({
            "role": "system",
            "content": system_prompt
        })

    conversation.extend([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text},
                {"type": "audio", "path": audio_path},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer_text}
            ],
        }
    ])

    return conversation


def make_collate_fn(processor: AutoProcessor, debug_first_batch: bool = True):
    """
    Collate function that:
      * Takes examples with a 'conversation' field (audio + text)
      * Calls processor.apply_chat_template(..., output_labels=True)
      * Returns everything the Audio Flamingo model needs.
    """
    first = {"done": False}

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [ex["conversation"] for ex in examples]

        batch_inputs = processor.apply_chat_template(
                            conversations,
                            tokenize=True,
                            add_generation_prompt=False,
                            return_dict=True,
                            output_labels=True,
                        )

        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        labels = batch_inputs["labels"]
        input_features = batch_inputs["input_features"]
        input_features_mask = batch_inputs["input_features_mask"]

        if debug_first_batch and not first["done"]:
            first["done"] = True
            print("=" * 80)
            print("[DEBUG] Example conversation (first item):")
            print(json.dumps(conversations[0], indent=2))
            print("=" * 80)
            print(f"[DEBUG] input_ids shape: {tuple(input_ids.shape)}")
            print(f"[DEBUG] attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"[DEBUG] labels shape: {tuple(labels.shape)}")
            print(f"[DEBUG] input_features shape: {tuple(input_features.shape)}")
            print(f"[DEBUG] input_features_mask shape: {tuple(input_features_mask.shape)}")
            non_pad_tokens = (attention_mask > 0).sum().item()
            print(f"[DEBUG] total non-pad tokens in batch: {non_pad_tokens}")
            print("=" * 80)
            decoded_0 = processor.batch_decode(
                input_ids[:1],
                skip_special_tokens=False,
            )[0]
            print("[DEBUG] Decoded first sequence (truncated to 1000 chars):")
            print(decoded_0[:1000])
            print("=" * 80)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }

    return collate_fn


def print_trainable_parameters(model: torch.nn.Module):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"trainable params: {trainable:,} || all params: {total:,} " f"|| trainable%: {100 * trainable / total:.4f}")


def safe_parse_prediction(text: str, dims: List[str] = None) -> List[float]:
    """
    Parse model predictions, attempting JSON first then regex fallback.
    
    Args:
        text: Raw prediction text from the model
        dims: List of dimension names to extract (optional)
    
    Returns:
        List of float values for each dimension
    """
    if dims is None:
        dims = []
    
    # Try to pull out a JSON object; then fall back to regex
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
    else:
        candidate = text

    obj: Dict[str, Any] = {}
    try:
        obj = json.loads(candidate)
    except Exception:
        obj = {}

    for dim in dims:
        if dim in obj:
            continue
        pattern = rf'"{dim}"\s*:\s*([-+]?\d*\.?\d+)'
        m = re.search(pattern, candidate)
        if m:
            try:
                obj[dim] = float(m.group(1))
            except Exception:
                pass

    preds: List[float] = []
    for dim in dims:
        v = obj.get(dim, 0.0)
        if isinstance(v, str):
            try:
                v = float(v)
            except Exception:
                v = 0.0
        preds.append(float(v))
    return preds
