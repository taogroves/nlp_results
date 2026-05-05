#!/usr/bin/env python3
"""
Style-Transfer Fine-Tuning Pipeline
====================================

End-to-end script that:
  1. Detects available hardware (Apple Silicon MPS / NVIDIA CUDA).
  2. Downloads and loads the Qwen3.5-9B model.
  3. Generates *base* responses to a set of writing prompts.
  4. Loads the user's markdown writing samples and trains a LoRA adapter
     that captures the user's idiolect.
  5. Generates *style-adapted* responses to the same prompts.
  6. Writes a side-by-side comparison report to the output directory.

Usage
-----
    python run.py                          # run with defaults
    python run.py --samples-dir ./my_docs  # custom sample directory
    python run.py --model Qwen/Qwen3-8B   # different base model
    python run.py --epochs 5              # more training epochs
    python run.py --skip-training          # inference only (adapter must exist)
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

# Disable hf_xet before any huggingface imports — it segfaults on
# Python 3.13 + macOS ARM with large sharded downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

# Ensure the project root is on sys.path so relative imports work when
# invoked as `python run.py` from the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PipelineConfig
from src.hardware import detect_hardware, apply_hardware_profile
from src.model_manager import (
    load_tokenizer,
    load_base_model,
    attach_lora,
    load_adapted_model,
    save_adapter,
)
from src.data_pipeline import prepare_datasets, load_prompts
from src.trainer import StyleTrainer
from src.inference import (
    generate_base_responses,
    generate_adapted_responses,
    PromptResult,
)
from src.report import generate_report, generate_training_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Style-transfer LoRA fine-tuning pipeline")
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID (default: Qwen/Qwen3.5-9B)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="Directory containing .md writing samples")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="Text file with one prompt per line")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for outputs and adapter weights")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device training batch size")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate")
    p.add_argument("--lora-rank", type=int, default=None,
                   help="LoRA rank (r)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Max tokens to generate per response")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training; load an existing adapter instead")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantisation (use full precision)")
    return p.parse_args()


def apply_cli_overrides(cfg: PipelineConfig, args: argparse.Namespace):
    if args.model:
        cfg.model.model_name = args.model
    if args.samples_dir:
        cfg.paths.writing_samples_dir = args.samples_dir
    if args.prompts_file:
        cfg.paths.prompts_file = args.prompts_file
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
        cfg.paths.adapter_dir = args.output_dir / "style_adapter"
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.per_device_batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.lora_rank:
        cfg.lora.r = args.lora_rank
        cfg.lora.lora_alpha = args.lora_rank * 2
    if args.max_new_tokens:
        cfg.generation.max_new_tokens = args.max_new_tokens
    if args.no_4bit:
        cfg.model.load_in_4bit = False


def free_memory(model=None):
    """Release GPU memory between pipeline stages."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = PipelineConfig()
    apply_cli_overrides(cfg, args)

    # ── 1. Hardware ──────────────────────────────────────────────────
    hw = detect_hardware()
    apply_hardware_profile(cfg, hw)
    logger.info("Device: %s  |  dtype: %s", hw.device_name, hw.dtype)

    # ── 2. Tokenizer ─────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.model.model_name, cfg.model.trust_remote_code)

    # ── 3. Load prompts ──────────────────────────────────────────────
    prompts = load_prompts(cfg.paths.prompts_file)

    # ── 4. Base model inference ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Base model inference")
    logger.info("=" * 60)

    base_model = load_base_model(
        cfg.model.model_name, hw,
        load_in_4bit=cfg.model.load_in_4bit,
        use_flash_attention=cfg.model.use_flash_attention,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    base_responses = generate_base_responses(
        base_model, tokenizer, prompts, cfg.generation, hw.device,
    )
    free_memory(base_model)

    # ── 5. Training ──────────────────────────────────────────────────
    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("STAGE 2: LoRA fine-tuning on writing samples")
        logger.info("=" * 60)

        dataset = prepare_datasets(
            cfg.paths.writing_samples_dir,
            cfg.paths.prompts_file,
            tokenizer,
            chunk_size=cfg.training.chunk_size,
            chunk_overlap=cfg.training.chunk_overlap,
            max_seq_length=cfg.model.max_seq_length,
        )

        train_model = load_base_model(
            cfg.model.model_name, hw,
            load_in_4bit=cfg.model.load_in_4bit,
            use_flash_attention=cfg.model.use_flash_attention,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        train_model = attach_lora(train_model, cfg.lora, hw)

        trainer = StyleTrainer(train_model, tokenizer, dataset, cfg, hw)
        train_result = trainer.train()

        generate_training_summary(
            train_result, len(dataset), cfg.model.model_name, cfg.paths.output_dir,
        )
        free_memory(train_model)
    else:
        if not cfg.paths.adapter_dir.exists():
            logger.error("--skip-training was set but no adapter found at %s",
                         cfg.paths.adapter_dir)
            sys.exit(1)
        logger.info("Skipping training; using existing adapter at %s",
                     cfg.paths.adapter_dir)

    # ── 6. Adapted model inference ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3: Style-adapted inference")
    logger.info("=" * 60)

    adapted_model = load_adapted_model(
        cfg.model.model_name,
        cfg.paths.adapter_dir,
        hw,
        load_in_4bit=cfg.model.load_in_4bit,
        use_flash_attention=cfg.model.use_flash_attention,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    adapted_responses = generate_adapted_responses(
        adapted_model, tokenizer, prompts, cfg.generation, hw.device,
    )
    free_memory(adapted_model)

    # ── 7. Report ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4: Generating comparison report")
    logger.info("=" * 60)

    results = [
        PromptResult(prompt=p, base_response=b, adapted_response=a)
        for p, b, a in zip(prompts, base_responses, adapted_responses)
    ]
    report_path = generate_report(results, cfg.paths.output_dir, cfg.model.model_name)

    logger.info("Done!  Report: %s", report_path)
    logger.info("Adapter weights: %s", cfg.paths.adapter_dir)


if __name__ == "__main__":
    main()
