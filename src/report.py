"""
Generate a side-by-side comparison report (Markdown) showing base model
responses next to style-adapted responses for every prompt.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.inference import PromptResult

logger = logging.getLogger(__name__)


def generate_report(
    results: list[PromptResult],
    output_dir: Path,
    model_name: str,
) -> Path:
    """Write a Markdown report and return the path to the file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_{timestamp}.md"

    lines: list[str] = []
    lines.append(f"# Style Transfer Comparison Report")
    lines.append("")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Prompts evaluated:** {len(results)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"## Prompt {i}")
        lines.append("")
        lines.append(f"> {r.prompt}")
        lines.append("")

        lines.append("### Base Model Response")
        lines.append("")
        lines.append(r.base_response)
        lines.append("")

        lines.append("### Style-Adapted Response")
        lines.append("")
        lines.append(r.adapted_response)
        lines.append("")
        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison report written to %s", report_path)
    return report_path


def generate_training_summary(
    train_result,
    num_samples: int,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Write a short Markdown summary of the training run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.md"

    lines = [
        "# Training Summary",
        "",
        f"**Model:** {model_name}",
        f"**Training samples:** {num_samples}",
        f"**Final training loss:** {train_result.training_loss:.4f}",
        f"**Global steps:** {train_result.global_step}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if hasattr(train_result, "metrics"):
        lines.append("## Metrics")
        lines.append("")
        for k, v in train_result.metrics.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Training summary written to %s", summary_path)
    return summary_path
