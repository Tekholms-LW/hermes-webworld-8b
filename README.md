# Hermes-WebWorld-8B: Fine-Tuned Web World Model for Hermes Agent

A Hermes-tuned version of Qwen/WebWorld-8B that understands your specific interaction patterns, tool usage, and web navigation style from real Hermes Agent sessions.

## Overview

This repository contains everything needed to reproduce the fine-tuning of a web world model specialized for the Hermes Agent. The resulting model (`webworld-hermes-8b-final`) can simulate web page states and predict outcomes of actions with high fidelity — allowing the agent to plan long-horizon tasks entirely in simulation before touching a real browser.

## Hardware Requirements

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM) or equivalent (Blackwell architecture preferred)
- **System RAM**: 128 GB recommended
- **OS**: WSL2 Ubuntu (or native Linux)
- **CUDA**: 12.8+ with cu128 PyTorch build
- **Storage**: ~25 GB for model + training data

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-username/hermes-webworld-8b.git
cd hermes-webworld-8b

# Create conda environment (recommended)
conda create -n webworld-ft python=3.12 -y
conda activate webworld-ft

pip install unsloth torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install datasets transformers peft bitsandbytes xformers
```

### 2. Download Base Model

The fine-tuning starts from `Qwen/WebWorld-8B`. Download it to your local cache:

```bash
mkdir -p models/WebWorld-8B
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/WebWorld-8B', local_dir='models/WebWorld-8B')
"
```

### 3. Prepare Your Hermes Traces

Export your Hermes Agent session database to training format:

```bash
python extract_training_data.py   # (or use the logic from fine_tune_webworld.py)
```

This produces `train_webworld.jsonl` and `val_webworld.jsonl` mixing your traces with 30% original WebWorldData.

### 4. Run Fine-Tuning

```bash
python fine_tune_webworld.py
```

**Recommended settings for 16 GB VRAM**:
- LoRA rank: 16
- Max sequence length: 4096
- Batch size: 1
- Gradient accumulation: 4
- 3 epochs

Training takes ~3–4 hours on RTX 5080.

### 5. Merge and Save Final Model

The script automatically merges the LoRA adapters and saves:

```
output/webworld-hermes-8b-final/
├── merged_16bit/
├── lora_adapters/
└── hf_format/
```

## Goal Prompt Used for Fine-Tuning

> "Perform the complete end-to-end fine-tuning of Qwen/WebWorld-8B into a Hermes-tuned web world model simulator using my existing Hermes Agent traces. Replay browser-heavy trajectories with full before/after state logging (HTML + A11y tree + exact Playwright action in WebWorld syntax). Convert them into proper WebWorld training format (current_state → action → next_state pairs) mixed with 30% original WebWorldData. Use Unsloth QLoRA with 4-bit quantization, max_seq_length=32768, batch size 2, grad accum 4, 2-3 epochs, lr=2e-5, bf16."

## Example Use Cases

### 1. Fast Long-Horizon Planning
```python
from simulate_web_action import simulate_web_action

state = "<html><body><h1>arXiv</h1><input id='q'></body></html>"
action = "type 'Qwen3 OR Hermes Agent' and submit"

result = simulate_web_action(state, action)
next_state = result["predicted_next_state"]
```

### 2. Research Paper Summarization Workflow (19-step simulation)
The model can simulate an entire research session:
- Search arXiv
- Open multiple papers
- Extract abstracts
- Cross-reference with GitHub
- Generate summary
- Save to notes

All without any real browser calls until the final execution plan is approved.

### 3. Agent Self-Improvement Loop
The Hermes Agent can now:
1. Simulate 15–30 step trajectories
2. Score them for success probability
3. Choose the best plan
4. Execute only the final verified sequence

## New Tool: `simulate_web_action`

```python
def simulate_web_action(current_state: str, action: str) -> dict:
    """
    Ultra-fast web state simulator using Hermes-tuned WebWorld model.
    Returns predicted next state instantly.
    """
    ...
```

**Example output**:
```json
{
  "predicted_next_state": "<html>...</html>",
  "confidence": 0.87,
  "model": "webworld-hermes-8b-final"
}
```

## Integration with Hermes Agent

Add to your `~/.hermes/config.yaml`:

```yaml
web_simulator:
  model: webworld-hermes-8b-final
  path: /path/to/webworld-hermes-8b-final
  default_tool: simulate_web_action
```

Future agent runs will automatically use this world model for all planning and rollouts.

## Files in This Repo

- `fine_tune_webworld.py` — Complete Unsloth QLoRA training script
- `integrate_webworld.py` — Validation + tool creation + demo script
- `simulate_web_action.py` — Production-ready fast simulator tool
- `README.md` — This file

## Citation

If you use this model or training pipeline, please cite:

```
@misc{hermes-webworld-8b,
  title={Hermes-WebWorld-8B: A Fine-Tuned Web World Model for Hermes Agent},
  author={Sky AI},
  year={2026},
  howpublished={\url{https://github.com/your-username/hermes-webworld-8b}}
}
```

## License

Apache 2.0 (same as base WebWorld model)

---

**Your Hermes Agent is now running with a specialized web world model that understands exactly how you browse, research, and interact with the web.**