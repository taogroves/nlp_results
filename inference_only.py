#!/usr/bin/env python3
"""
Inference-only script: re-run base and style-adapted generation using
an already-trained LoRA adapter, then produce a side-by-side report.

Skips all training — just loads models, generates, and writes the report.

Usage
-----
    python inference_only.py
    python inference_only.py --adapter-dir output/style_adapter
    python inference_only.py --max-new-tokens 1024
    python inference_only.py --prompts-file data/prompts.txt
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import PipelineConfig
from src.hardware import detect_hardware, apply_hardware_profile
from src.model_manager import load_tokenizer, load_base_model, load_adapted_model
from src.data_pipeline import load_prompts
from src.inference import (
    generate_base_responses,
    generate_adapted_responses,
    PromptResult,
)
from src.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")


def parse_args():
    p = argparse.ArgumentParser(description="Inference-only: base vs style-adapted comparison")
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID (default: Qwen/Qwen3.5-9B)")
    p.add_argument("--adapter-dir", type=Path, default=None,
                   help="Path to trained LoRA adapter (default: output/style_adapter)")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="Text file with one prompt per line")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for the comparison report")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Max tokens to generate per response (default: 512)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantisation")
    p.add_argument("--base-only", action="store_true",
                   help="Only run base model inference (skip adapter)")
    return p.parse_args()


def free_memory(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = PipelineConfig()

    if args.model:
        cfg.model.model_name = args.model
    if args.adapter_dir:
        cfg.paths.adapter_dir = args.adapter_dir
    if args.prompts_file:
        cfg.paths.prompts_file = args.prompts_file
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.max_new_tokens:
        cfg.generation.max_new_tokens = args.max_new_tokens
    if args.temperature:
        cfg.generation.temperature = args.temperature
    if args.no_4bit:
        cfg.model.load_in_4bit = False

    # ── Hardware ─────────────────────────────────────────────────────
    hw = detect_hardware()
    apply_hardware_profile(cfg, hw)
    logger.info("Device: %s  |  dtype: %s", hw.device_name, hw.dtype)

    # ── Tokenizer + prompts ──────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.model.model_name, cfg.model.trust_remote_code)
    prompts = load_prompts(cfg.paths.prompts_file)
    logger.info("%d prompts loaded", len(prompts))

    # ── Base model inference ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Base model inference")
    logger.info("=" * 60)

    base_model = load_base_model(
        cfg.model.model_name, hw,
        load_in_4bit=cfg.model.load_in_4bit,
        use_flash_attention=False,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    base_responses = generate_base_responses(
        base_model, tokenizer, prompts, cfg.generation, hw.device,
    )
    free_memory(base_model)

    # ── Adapted model inference ──────────────────────────────────────
    if args.base_only:
        adapted_responses = ["(skipped — base-only mode)"] * len(prompts)
    else:
        if not cfg.paths.adapter_dir.exists():
            logger.error("Adapter not found at %s — run training first, or pass --adapter-dir",
                         cfg.paths.adapter_dir)
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Style-adapted inference (adapter: %s)", cfg.paths.adapter_dir)
        logger.info("=" * 60)

        adapted_model = load_adapted_model(
            cfg.model.model_name,
            cfg.paths.adapter_dir, hw,
            load_in_4bit=cfg.model.load_in_4bit,
            use_flash_attention=False,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        adapted_responses = generate_adapted_responses(
            adapted_model, tokenizer, prompts, cfg.generation, hw.device,
        )
        free_memory(adapted_model)

    # ── Report ───────────────────────────────────────────────────────
    results = [
        PromptResult(prompt=p, base_response=b, adapted_response=a)
        for p, b, a in zip(prompts, base_responses, adapted_responses)
    ]
    report_path = generate_report(results, cfg.paths.output_dir, cfg.model.model_name)

    logger.info("Done!  Report: %s", report_path)


if __name__ == "__main__":
    main()
