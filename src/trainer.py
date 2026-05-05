"""
LoRA fine-tuning loop using HuggingFace Trainer.

Trains a lightweight LoRA adapter on the user's writing corpus so the
frozen base model learns to reproduce the author's stylistic patterns.
This follows the StyleAdaptedLM paradigm: the adapter captures style
while the base model retains general instruction-following capability.
"""

import logging
from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import PipelineConfig
from src.hardware import HardwareProfile

logger = logging.getLogger(__name__)


class StyleTrainer:
    """Thin wrapper around HuggingFace Trainer configured for style LoRA."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset: Dataset,
        cfg: PipelineConfig,
        hw: HardwareProfile,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg
        self.hw = hw

        self.output_dir = cfg.paths.adapter_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_training_args(self) -> TrainingArguments:
        tc = self.cfg.training

        kwargs = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.per_device_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            weight_decay=tc.weight_decay,
            warmup_ratio=tc.warmup_ratio,
            lr_scheduler_type=tc.lr_scheduler_type,
            max_grad_norm=tc.max_grad_norm,
            logging_steps=tc.logging_steps,
            save_strategy="steps",
            save_steps=tc.save_steps,
            save_total_limit=2,
            fp16=tc.fp16,
            bf16=tc.bf16,
            optim=tc.optim,
            gradient_checkpointing=self.hw.gradient_checkpointing,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=self.hw.device.type == "cuda",
        )

        # gradient_checkpointing_kwargs only meaningful when enabled
        if self.hw.gradient_checkpointing:
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        return TrainingArguments(**kwargs)

    def train(self):
        logger.info("Starting LoRA fine-tuning (%d samples, %d epoch(s))",
                     len(self.dataset), self.cfg.training.num_epochs)

        training_args = self._build_training_args()

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=collator,
        )

        train_result = trainer.train()
        logger.info("Training complete — loss=%.4f", train_result.training_loss)

        # Save the final adapter
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info("Adapter + tokenizer saved to %s", self.output_dir)

        return train_result
