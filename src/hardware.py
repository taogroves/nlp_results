"""
Hardware detection and configuration.

Automatically selects between Apple Silicon (MPS) and NVIDIA (CUDA),
and adjusts training / inference settings accordingly.
"""

import platform
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    device: torch.device
    device_name: str
    dtype: torch.dtype
    use_flash_attention: bool
    bf16: bool
    fp16: bool
    optim: str
    gradient_checkpointing: bool


def detect_hardware() -> HardwareProfile:
    """Return a HardwareProfile describing the best available accelerator."""

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA device detected: %s", name)
        return HardwareProfile(
            device=torch.device("cuda"),
            device_name=name,
            dtype=torch.bfloat16,
            use_flash_attention=True,
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS backend detected")
        return HardwareProfile(
            device=torch.device("mps"),
            device_name=f"Apple Silicon ({platform.processor() or 'arm'})",
            dtype=torch.float16,
            use_flash_attention=False,  # flash-attn not available on MPS
            bf16=False,
            fp16=True,
            optim="adamw_torch",  # paged_adamw_8bit requires CUDA
            gradient_checkpointing=True,
        )

    logger.warning("No GPU detected — falling back to CPU (training will be very slow)")
    return HardwareProfile(
        device=torch.device("cpu"),
        device_name="CPU",
        dtype=torch.float32,
        use_flash_attention=False,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,
    )


def apply_hardware_profile(cfg, hw: HardwareProfile):
    """Mutate a PipelineConfig in place so it matches the detected hardware."""

    cfg.training.bf16 = hw.bf16
    cfg.training.fp16 = hw.fp16
    cfg.training.optim = hw.optim
    cfg.model.use_flash_attention = hw.use_flash_attention

    if hw.device.type == "mps":
        # bitsandbytes 4-bit quantisation is CUDA-only; disable on MPS
        cfg.model.load_in_4bit = False

    logger.info(
        "Hardware config applied — device=%s  dtype=%s  flash_attn=%s  4bit=%s",
        hw.device, hw.dtype, hw.use_flash_attention, cfg.model.load_in_4bit,
    )
