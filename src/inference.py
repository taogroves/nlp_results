"""
Inference helpers: generate responses from both the base model and the
style-adapted model for a list of prompts.
"""

import logging
from dataclasses import dataclass

import torch
from transformers import GenerationConfig as HFGenerationConfig

from config import GenerationConfig
from src.data_pipeline import STYLE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    prompt: str
    base_response: str
    adapted_response: str


def _build_generation_config(gen_cfg: GenerationConfig) -> HFGenerationConfig:
    return HFGenerationConfig(
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        repetition_penalty=gen_cfg.repetition_penalty,
        do_sample=gen_cfg.do_sample,
    )


def _generate_response(
    model,
    tokenizer,
    prompt_text: str,
    system_prompt: str | None,
    gen_config: HFGenerationConfig,
    device: torch.device,
) -> str:
    """Format a chat-style prompt, run inference, and decode the response."""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_config,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Strip the input tokens to get only the generated portion
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_base_responses(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenerationConfig,
    device: torch.device,
) -> list[str]:
    """Generate vanilla responses from the unmodified base model."""

    gen_config = _build_generation_config(gen_cfg)
    responses = []

    for i, prompt in enumerate(prompts, 1):
        logger.info("Base inference [%d/%d]: %s", i, len(prompts), prompt[:80])
        resp = _generate_response(
            model, tokenizer, prompt,
            system_prompt=None,
            gen_config=gen_config,
            device=device,
        )
        responses.append(resp)

    return responses


def generate_adapted_responses(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenerationConfig,
    device: torch.device,
) -> list[str]:
    """Generate style-conditioned responses using the LoRA-adapted model."""

    gen_config = _build_generation_config(gen_cfg)
    responses = []

    for i, prompt in enumerate(prompts, 1):
        logger.info("Adapted inference [%d/%d]: %s", i, len(prompts), prompt[:80])
        resp = _generate_response(
            model, tokenizer, prompt,
            system_prompt=STYLE_SYSTEM_PROMPT,
            gen_config=gen_config,
            device=device,
        )
        responses.append(resp)

    return responses


def run_comparison(
    base_model,
    adapted_model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenerationConfig,
    device: torch.device,
) -> list[PromptResult]:
    """Generate base and adapted responses for every prompt, returning paired results."""

    base_responses = generate_base_responses(
        base_model, tokenizer, prompts, gen_cfg, device,
    )
    adapted_responses = generate_adapted_responses(
        adapted_model, tokenizer, prompts, gen_cfg, device,
    )

    results = [
        PromptResult(prompt=p, base_response=b, adapted_response=a)
        for p, b, a in zip(prompts, base_responses, adapted_responses)
    ]
    return results
