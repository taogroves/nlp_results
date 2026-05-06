"""
Model lifecycle: download, load (with optional quantisation), attach LoRA
adapter, merge, and save.

Supports:
  - NVIDIA A100  → BitsAndBytes 4-bit NF4 quantisation + flash-attn-2
  - Apple Silicon → full-precision float16 on MPS (no bitsandbytes)
"""

import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from src.hardware import HardwareProfile

logger = logging.getLogger(__name__)


def _quantization_config(hw: HardwareProfile) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for 4-bit QLoRA when running on CUDA."""
    if hw.device.type != "cuda":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=hw.dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str,
    hw: HardwareProfile,
    load_in_4bit: bool = True,
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
):
    """Download (if necessary) and load the base model onto the right device."""

    quant_config = _quantization_config(hw) if load_in_4bit else None

    attn_impl = "eager"

    logger.info(
        "Loading %s  quant=%s  attn=%s  device=%s",
        model_name,
        "nf4" if quant_config else "none",
        attn_impl,
        hw.device,
    )

    kwargs = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=hw.dtype,
        device_map="auto" if hw.device.type == "cuda" else None,
        quantization_config=quant_config,
    )

    if attn_impl != "eager":
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    # Move to device explicitly when device_map is not used (MPS / CPU)
    if hw.device.type != "cuda":
        model = model.to(hw.device)

    model.eval()
    return model


def attach_lora(model, lora_cfg, hw: HardwareProfile):
    """Wrap the base model with a fresh LoRA adapter."""

    if hw.device.type == "cuda" and hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=hw.gradient_checkpointing
        )

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA attached — trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model


def load_adapted_model(
    model_name: str,
    adapter_dir: Path,
    hw: HardwareProfile,
    load_in_4bit: bool = True,
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
):
    """Load the base model and layer the trained LoRA adapter on top."""

    base = load_base_model(
        model_name, hw, load_in_4bit, use_flash_attention, trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    logger.info("Loaded style adapter from %s", adapter_dir)
    return model


def save_adapter(model, adapter_dir: Path):
    """Save only the LoRA adapter weights (not the full base model)."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to %s", adapter_dir)
