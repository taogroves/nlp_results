"""
Data pipeline: load markdown writing samples, clean them, and build
a HuggingFace Dataset suitable for causal-language-model LoRA training.

Two training data strategies are implemented:

1. **CLM chunks** — the user's raw prose is tokenised and sliced into
   fixed-length chunks.  The model learns the user's statistical writing
   patterns via next-token prediction on these chunks.

2. **Style-conditioned instruction pairs** — each writing prompt is paired
   with a passage from the user's corpus so the adapter also learns to
   respond to instructions *in the user's voice*.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown loading / cleaning
# ---------------------------------------------------------------------------

def load_writing_samples(samples_dir: Path) -> list[str]:
    """Read every .md file in *samples_dir* and return a list of cleaned texts."""
    md_files = sorted(samples_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {samples_dir}. "
            "Place your writing samples there before running the pipeline."
        )
    logger.info("Found %d markdown file(s) in %s", len(md_files), samples_dir)

    texts: list[str] = []
    for fp in md_files:
        raw = fp.read_text(encoding="utf-8")
        cleaned = _clean_markdown(raw)
        if len(cleaned.split()) >= 20:  # skip very short fragments
            texts.append(cleaned)
        else:
            logger.warning("Skipping %s (fewer than 20 words after cleaning)", fp.name)
    return texts


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting while preserving prose structure."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)                  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)         # links → text
    text = re.sub(r"```[\s\S]*?```", "", text)                   # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)                          # inline code
    text = re.sub(r"(\*{1,3}|_{1,3})(.*?)\1", r"\2", text)      # bold / italic
    text = re.sub(r"^[>\-\*]\s+", "", text, flags=re.MULTILINE)  # blockquotes & lists
    text = re.sub(r"\n{3,}", "\n\n", text)                       # collapse blank lines
    return text.strip()


def load_prompts(prompts_file: Path) -> list[str]:
    """Read one prompt per non-blank line from *prompts_file*."""
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    prompts = [
        line.strip()
        for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_file)
    return prompts


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

STYLE_SYSTEM_PROMPT = (
    "You are a writing assistant. Write your response in the exact personal "
    "style of the author whose examples you have been trained on. Match their "
    "vocabulary, sentence structure, tone, and rhetorical habits precisely."
)


def build_clm_dataset(
    texts: list[str],
    tokenizer,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> Dataset:
    """Tokenise raw texts and split into overlapping chunks for CLM training."""

    all_input_ids: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_input_ids.extend(ids)

    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(all_input_ids) - chunk_size + 1, step):
        chunk = all_input_ids[start : start + chunk_size]
        chunks.append({"input_ids": chunk, "labels": chunk.copy()})

    if not chunks:
        # corpus is shorter than one chunk — use whatever we have
        chunks.append({
            "input_ids": all_input_ids,
            "labels": all_input_ids.copy(),
        })

    logger.info("Built %d CLM training chunks (size=%d, overlap=%d)",
                len(chunks), chunk_size, chunk_overlap)
    return Dataset.from_list(chunks)


def build_style_instruction_dataset(
    texts: list[str],
    prompts: list[str],
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Create instruction-response pairs that teach the adapter to respond
    to arbitrary prompts in the author's style.

    Each sample pairs a randomly selected writing prompt with a passage from
    the user's corpus, formatted as a chat conversation.
    """
    import random

    records = []
    for i, prompt in enumerate(prompts):
        passage = texts[i % len(texts)]
        # Truncate passage to fit within max_length budget
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        if len(passage_tokens) > max_length - 256:
            passage_tokens = passage_tokens[: max_length - 256]
            passage = tokenizer.decode(passage_tokens, skip_special_tokens=True)

        messages = [
            {"role": "system", "content": STYLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": passage},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            max_length=max_length, truncation=True,
        )
        records.append({
            "input_ids": formatted,
            "labels": formatted.copy(),
        })

    # Also add raw CLM passages formatted as assistant turns for diversity
    random.seed(42)
    for text in texts:
        messages = [
            {"role": "system", "content": STYLE_SYSTEM_PROMPT},
            {"role": "user", "content": random.choice(prompts)},
            {"role": "assistant", "content": text},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            max_length=max_length, truncation=True,
        )
        records.append({
            "input_ids": formatted,
            "labels": formatted.copy(),
        })

    logger.info("Built %d style-instruction training samples", len(records))
    return Dataset.from_list(records)


def prepare_datasets(
    samples_dir: Path,
    prompts_file: Path,
    tokenizer,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    max_seq_length: int = 2048,
) -> Dataset:
    """
    Master function: load samples + prompts, build both CLM and instruction
    datasets, and concatenate them into a single shuffled training set.
    """
    from datasets import concatenate_datasets

    texts = load_writing_samples(samples_dir)
    prompts = load_prompts(prompts_file)

    clm_ds = build_clm_dataset(texts, tokenizer, chunk_size, chunk_overlap)
    inst_ds = build_style_instruction_dataset(texts, prompts, tokenizer, max_seq_length)

    combined = concatenate_datasets([clm_ds, inst_ds]).shuffle(seed=42)
    logger.info("Combined training set: %d samples", len(combined))
    return combined
