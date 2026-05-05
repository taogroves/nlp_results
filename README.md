# Style-Transfer Fine-Tuning Pipeline

Parameter-efficient fine-tuning (LoRA) applied to **Qwen3.5-9B** to transfer a user's writing style onto a base LLM. Inspired by the [StyleAdaptedLM](https://arxiv.org/abs/2507.18294) framework.

## How It Works

1. **Base inference** — the unmodified model answers a set of writing prompts.
2. **LoRA training** — a lightweight adapter is trained on the user's markdown writing samples using two complementary objectives:
   - *Causal language modelling (CLM)* on chunked raw prose, so the adapter internalises the author's statistical writing patterns.
   - *Style-conditioned instruction pairs*, where prompts are paired with author passages so the adapter also learns to respond to instructions in the author's voice.
3. **Adapted inference** — the model + adapter answers the same prompts.
4. **Report** — base and adapted responses are saved side-by-side in a Markdown document.

The adapter is ~0.5 % of the base model's parameters and can be swapped, shared, or discarded without touching the frozen base weights.

## Hardware Support

| Hardware | Quantisation | Attention | Optimiser |
|---|---|---|---|
| NVIDIA A100 (CUDA) | 4-bit NF4 (QLoRA) | Flash Attention 2 | `paged_adamw_8bit` |
| Apple Silicon (MPS) | None (float16) | Eager | `adamw_torch` |

Hardware is detected automatically at runtime.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your writing samples (*.md files) in data/writing_samples/
cp ~/my_essays/*.md data/writing_samples/

# 3. (Optional) Edit the prompts
#    data/prompts.txt  — one prompt per line

# 4. Run the full pipeline
python run.py

# 5. Find the outputs
#    output/comparison_<timestamp>.md   — side-by-side report
#    output/training_summary.md         — training metrics
#    output/style_adapter/              — saved LoRA weights
```

## CLI Options

```
--model NAME          HuggingFace model ID (default: Qwen/Qwen3.5-9B)
--samples-dir PATH    Directory with .md writing samples
--prompts-file PATH   Text file with one prompt per line
--output-dir PATH     Where to save outputs and adapter weights
--epochs N            Training epochs (default: 3)
--batch-size N        Per-device batch size (default: 1)
--lr FLOAT            Learning rate (default: 2e-4)
--lora-rank N         LoRA rank r (default: 32)
--max-new-tokens N    Max tokens per generated response (default: 512)
--skip-training       Skip training; use an existing adapter
--no-4bit             Disable 4-bit quantisation
```

## Project Structure

```
├── run.py                  # Main entry point
├── config.py               # All tuneable parameters
├── requirements.txt
├── data/
│   ├── writing_samples/    # Place .md files here
│   └── prompts.txt         # Writing prompts
├── src/
│   ├── hardware.py         # Apple Silicon / NVIDIA detection
│   ├── data_pipeline.py    # Markdown loading, dataset construction
│   ├── model_manager.py    # Model download, LoRA attachment
│   ├── trainer.py          # HuggingFace Trainer wrapper
│   ├── inference.py        # Base + adapted generation
│   └── report.py           # Markdown comparison report
└── output/                 # Generated outputs
```

## Approach & Design Rationale

**Why LoRA over full fine-tuning?** Full fine-tuning on a small personal corpus causes catastrophic forgetting — the model overwrites general capabilities in favour of the narrow training distribution. LoRA freezes the base weights and learns a low-rank additive update, preserving the model's reasoning while steering its style. The adapter is <30 MB vs ~18 GB for the full model.

**Why two training objectives?** Raw CLM chunks teach token-level statistical patterns (sentence length distribution, vocabulary preferences, punctuation habits). Style-conditioned instruction pairs teach the adapter to activate these patterns in response to prompts, bridging the gap between passive style knowledge and active style reproduction.

**Why Qwen3.5-9B?** It fits on a single 24 GB GPU at full precision and ~5 GB with 4-bit quantisation. Its architecture explicitly supports LoRA-style PEFT with control tokens pre-trained for this purpose. It also supports 262K context natively.
