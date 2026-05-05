"""
Central configuration for the style-transfer fine-tuning pipeline.
All tuneable parameters live here so the rest of the codebase stays clean.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent
    writing_samples_dir: Path = field(default=None)
    prompts_file: Path = field(default=None)
    output_dir: Path = field(default=None)
    adapter_dir: Path = field(default=None)

    def __post_init__(self):
        self.writing_samples_dir = self.writing_samples_dir or self.project_root / "data" / "writing_samples"
        self.prompts_file = self.prompts_file or self.project_root / "data" / "prompts.txt"
        self.output_dir = self.output_dir or self.project_root / "output"
        self.adapter_dir = self.adapter_dir or self.project_root / "output" / "style_adapter"


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    use_flash_attention: bool = True  # auto-disabled on Apple Silicon
    load_in_4bit: bool = True         # QLoRA-style quantisation
    max_seq_length: int = 2048
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_steps: int = 50
    fp16: bool = False   # set dynamically per hardware
    bf16: bool = False   # set dynamically per hardware
    optim: str = "paged_adamw_8bit"
    chunk_size: int = 1024  # token-level chunk size for training samples
    chunk_overlap: int = 128


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
