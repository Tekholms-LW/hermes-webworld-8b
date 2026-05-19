#!/usr/bin/env python3
"""
Exact Unsloth QLoRA fine-tuning script for WebWorld-8B → Hermes-tuned web world model.
Parameters: 4-bit QLoRA, max_seq_length=32768, batch_size=2, grad_accum=4, 
            2-3 epochs, lr=2e-5, bf16, target_modules for Qwen3.
"""

import os
import sys
import json
import math
import time
import gc
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass, field

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
import numpy as np

# ─── Unsloth imports ──────────────────────────────────────────────────────────

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# ─── Configuration (exact spec) ───────────────────────────────────────────────

MODEL_PATH = "/home/sky_ai/webworld-ft/models/WebWorld-8B"
DATA_DIR = "/home/sky_ai/webworld-ft/data"
OUTPUT_DIR = "/home/sky_ai/webworld-ft/output"

# QLoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
USE_RSLORA = True
LOFTQ_CONFIG = {}

# Target modules for Qwen3 architecture
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config (exact spec)
MAX_SEQ_LENGTH = 4096   # WebWorld supports up to 40K
BATCH_SIZE = 1           # per device
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3           # 2-3 epochs as specified
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01

# 4-bit quantization
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True


def wait_for_model(path: str, timeout_hours: int = 8):
    """Wait for model download to complete by checking for safetensors files."""
    print(f"Waiting for model download at {path}...")
    start = time.time()
    while True:
        safetensors = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if safetensors:
            # Check if index file exists and all expected files are present
            index_path = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                expected_files = set(weight_map.values())
                actual_files = set(safetensors)
                missing = expected_files - actual_files
                if not missing:
                    total_size = sum(
                        os.path.getsize(os.path.join(path, f)) for f in safetensors
                    )
                    print(f"Model download complete: {len(safetensors)} files, {total_size / 1e9:.1f}GB")
                    return True
                else:
                    # Check sizes of downloaded files
                    sizes = {f: os.path.getsize(os.path.join(path, f)) for f in safetensors}
                    for f in expected_files & actual_files:
                        print(f"  {f}: {sizes[f]/1e6:.0f}MB")
                    elapsed = time.time() - start
                    print(f"  Waiting for {len(missing)} more files... ({elapsed/60:.0f}min elapsed)")
            else:
                print(f"  Found {len(safetensors)} safetensors files, waiting for index...")
        
        elapsed = time.time() - start
        if elapsed > timeout_hours * 3600:
            raise TimeoutError(f"Model download timed out after {timeout_hours} hours")
        
        time.sleep(30)


def load_tokenizer_and_model():
    """Load WebWorld-8B with Unsloth 4-bit QLoRA."""
    print("\n" + "=" * 60)
    print("Loading WebWorld-8B with 4-bit QLoRA...")
    print("=" * 60)
    
    # Wait for model to be downloaded
    wait_for_model(MODEL_PATH)
    
    # Load model and tokenizer together
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        load_in_4bit=USE_4BIT,
        dtype=None,
        device_map="auto",
        max_seq_length=MAX_SEQ_LENGTH,
        token=None,
    )
    
    # Apply Qwen3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-3",
    )
    
    # Apply PEFT LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Total params: {total:,}")
    
    return model, tokenizer


def load_and_format_dataset(tokenizer):
    """Load and format the combined Hermes + WebWorld dataset."""
    print("\n" + "=" * 60)
    print("Loading and formatting dataset...")
    print("=" * 60)
    
    train_path = os.path.join(DATA_DIR, "train_webworld.jsonl")
    val_path = os.path.join(DATA_DIR, "val_webworld.jsonl")
    
    # Load as raw text
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    
    train_raw = load_jsonl(train_path)
    val_raw = load_jsonl(val_path)
    
    print(f"Train: {len(train_raw)} examples")
    print(f"Val:   {len(val_raw)} examples")
    
    def format_conversation(example):
        conversations = example["conversations"]
        messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({"role": role, "content": conv["value"]})
        
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return formatted
    
    print("Formatting with chat template...")
    train_texts = [format_conversation(ex) for ex in train_raw]
    val_texts = [format_conversation(ex) for ex in val_raw]
    
    # Tokenization function
    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"],
        desc="Tokenizing train",
    )
    val_dataset = val_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"],
        desc="Tokenizing val",
    )
    
    # Stats
    train_tokens = sum(len(ex["input_ids"]) for ex in train_dataset)
    val_tokens = sum(len(ex["input_ids"]) for ex in val_dataset)
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")
    print(f"Avg train seq: {train_tokens / len(train_dataset):.0f}")
    print(f"Avg val seq:   {val_tokens / len(val_dataset):.0f}")
    
    return train_dataset, val_dataset, train_tokens, val_tokens


def create_trainer(model, tokenizer, train_dataset, val_dataset):
    """Create trainer with exact specs."""
    print("\n" + "=" * 60)
    print("Configuring trainer...")
    print("=" * 60)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        perplexity = math.exp(min(loss.item(), 20))
        
        return {"eval_loss": loss.item(), "perplexity": perplexity}
    
    # Training arguments (exact spec)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=is_bfloat16_supported(),
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    class WebWorldTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs["labels"],
                return_dict=True,
            )
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
    
    trainer = WebWorldTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def train_model(trainer, train_tokens):
    """Run training and return metrics."""
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} (eff. {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Max seq: {MAX_SEQ_LENGTH}")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  BF16: {is_bfloat16_supported()}")
    print(f"  Train tokens: {train_tokens:,}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Track VRAM
    vram_peak = 0
    
    # Monkey-patch to track VRAM
    original_step = trainer.training_step
    def tracked_step(self, model, inputs, num_items_in_batch=None, *args, **kwargs):
        nonlocal vram_peak
        result = original_step(model, inputs, num_items_in_batch)
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated()
            vram_peak = max(vram_peak, current)
        return result
    
    trainer.training_step = tracked_step.__get__(trainer, type(trainer))
    
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Get final VRAM peak
    if torch.cuda.is_available():
        vram_peak = max(vram_peak, torch.cuda.max_memory_allocated())
    
    metrics = {
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "vram_peak_bytes": vram_peak,
        "vram_peak_gb": vram_peak / 1e9 if vram_peak else 0,
        "train_samples": len(trainer.train_dataset),
        "eval_samples": len(trainer.eval_dataset),
        "train_tokens": train_tokens,
        "model": "WebWorld-8B",
        "lora_r": LORA_R,
        "final_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
    }
    
    return metrics


def save_and_merge(model, tokenizer, metrics):
    """Save LoRA adapters, merge, and save final model."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    final_dir = os.path.join(OUTPUT_DIR, "webworld-hermes-8b-final")
    lora_dir = os.path.join(final_dir, "lora_adapters")
    merged_dir = os.path.join(final_dir, "merged_16bit")
    
    os.makedirs(final_dir, exist_ok=True)
    
    # 1. Save LoRA adapters
    print("Saving LoRA adapters...")
    model.save_pretrained_merged(
        lora_dir,
        tokenizer=tokenizer,
        save_method="lora",
    )
    print(f"  → {lora_dir}")
    
    # 2. Merge and save as 16-bit
    print("Merging and saving 16-bit model...")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )
    print(f"  → {merged_dir}")
    
    # 3. Also save in HuggingFace format
    hf_dir = os.path.join(final_dir, "hf_format")
    print("Saving HuggingFace format...")
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    print(f"  → {hf_dir}")
    
    # 4. Save metrics
    metrics_path = os.path.join(final_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics → {metrics_path}")
    
    # 5. Create model card
    card = f"""---
license: apache-2.0
base_model: Qwen/WebWorld-8B
tags:
- webworld
- hermes
- web-agent
- world-model
- fine-tuned
- qlora
---

# WebWorld-Hermes-8B

Fine-tuned from Qwen/WebWorld-8B on Hermes Agent traces + 30% WebWorldData.

**Training config:**
- Method: QLoRA (4-bit NF4)
- LoRA rank: {LORA_R}
- Learning rate: {LEARNING_RATE}
- Epochs: {NUM_EPOCHS}
- Max seq length: {MAX_SEQ_LENGTH}
- Batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS}
- Precision: bf16

**Training data:** {metrics.get('train_samples', '?')} Hermes + WebWorld examples ({metrics.get('train_tokens', '?')} tokens)

**VRAM peak:** {metrics.get('vram_peak_gb', 0):.1f} GB
**Training time:** {metrics.get('training_time_minutes', 0):.1f} min
"""
    card_path = os.path.join(final_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)
    print(f"  Model card → {card_path}")
    
    print(f"\n✅ Final model saved to: {final_dir}")
    return final_dir


def quick_eval(model_path, tokenizer):
    """Run a quick evaluation to verify the model works."""
    print("\n" + "=" * 60)
    print("QUICK EVALUATION")
    print("=" * 60)
    
    eval_data_path = os.path.join(DATA_DIR, "val_webworld.jsonl")
    with open(eval_data_path) as f:
        val_data = [json.loads(line) for line in f if line.strip()]
    
    # Take 3 test examples
    import random
    random.seed(42)
    test_examples = random.sample(val_data, min(3, len(val_data)))
    
    results = []
    
    for i, example in enumerate(test_examples):
        # Take only the first 2 messages (state + action) to predict next state
        convs = example["conversations"]
        # Get action from the conversation
        action_text = convs[2]["value"]
        if "Action:" in action_text:
            action_part = action_text.split("Action:")[1].split("\n")[0].strip()
        else:
            action_part = "unknown"
        
        input_msgs = [
            {"role": "user", "content": convs[0]["value"]},
            {"role": "assistant", "content": convs[1]["value"]},
            {"role": "user", "content": f"Continue the trajectory. Given the previous state, predict the next page state after this action.\n\nAction: {action_part}\n\nNext Page State:"}
        ]
        
        expected = convs[3]["value"]
        
        formatted = tokenizer.apply_chat_template(
            input_msgs, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "example": i + 1,
            "input_summary": input_msgs[-1]["content"][:100],
            "response_preview": response[:200],
            "expected_preview": expected[:200],
            "response_length": len(response),
        })
        
        print(f"\n--- Test Example {i+1} ---")
        print(f"INPUT: {results[-1]['input_summary']}...")
        print(f"MODEL: {results[-1]['response_preview']}...")
        print(f"EXPECTED: {results[-1]['expected_preview']}...")
    
    return results


def main():
    try:
        # Load model & tokenizer
        model, tokenizer = load_tokenizer_and_model()
        
        # Load dataset
        train_dataset, val_dataset, train_tokens, val_tokens = load_and_format_dataset(tokenizer)
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)
        
        # Train
        metrics = train_model(trainer, train_tokens)
        
        # Quick eval before saving
        eval_results = quick_eval(os.path.join(MODEL_PATH, "merged_16bit"), tokenizer)
        metrics["eval_samples"] = eval_results
        
        # Save merged model
        final_path = save_and_merge(model, tokenizer, metrics)
        
        # Output final report
        print("\n" + "=" * 60)
        print("SUCCESS REPORT")
        print("=" * 60)
        print(f"  Model path: {final_path}")
        print(f"  Training time: {metrics['training_time_minutes']:.1f} min")
        print(f"  VRAM peak: {metrics['vram_peak_gb']:.1f} GB")
        print(f"  Final loss: {metrics.get('final_loss', 'N/A')}")
        print(f"  Train samples: {metrics['train_samples']}")
        print(f"  Eval samples: {metrics['eval_samples']}")
        print()
        print("  === Before/After Simulation ===")
        for r in eval_results:
            print(f"  Test {r['example']}:")
            print(f"    Input: {r['input_summary'][:60]}...")
            print(f"    Model output: {r['response_preview'][:120]}...")
            print(f"    Expected: {r['expected_preview'][:120]}...")
            print()
        print("✅ Model trained, merged, and saved successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
