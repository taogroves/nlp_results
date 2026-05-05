#!/usr/bin/env python3
"""
Pre-download the base model so run.py starts instantly.

Works around the hf_xet segfault on Python 3.13 / macOS ARM by disabling
the xet backend and downloading shards sequentially via plain HTTPS.

Usage
-----
    python download_model.py                         # download Qwen3.5-9B
    python download_model.py --model Qwen/Qwen3-8B   # different model
"""

import argparse
import os
import sys

# Disable the xet backend BEFORE importing huggingface_hub — it segfaults
# on Python 3.13 + macOS ARM when downloading large sharded models.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Enable hf_transfer for fast parallel HTTPS downloads (if installed).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def ensure_hf_transfer():
    try:
        import hf_transfer  # noqa: F401
        print("hf_transfer enabled (fast Rust HTTPS downloader)")
    except ImportError:
        print("Installing hf_transfer for faster downloads...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "hf_transfer"],
            stdout=subprocess.DEVNULL,
        )
        print("hf_transfer installed and enabled")


def download_with_hub_api(model_id: str):
    """Download every file in the repo one-by-one so a single stalled
    shard doesn't block the whole snapshot_download thread pool."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    siblings = api.model_info(model_id, files_metadata=True).siblings
    files = [s.rfilename for s in siblings]

    print(f"\n{len(files)} files to download:\n")
    for i, fname in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {fname}")
        hf_hub_download(
            repo_id=model_id,
            filename=fname,
            repo_type="model",
        )
        print(f"           done ✓")

    # Return the cache path so the user knows where it ended up
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, local_files_only=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Pre-download model weights")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-9B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-9B)",
    )
    args = parser.parse_args()

    ensure_hf_transfer()

    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Authenticated (HF_TOKEN set)")
    else:
        print("Warning: HF_TOKEN not set — download may be rate-limited")

    print(f"\nDownloading {args.model} (xet backend disabled, using plain HTTPS) ...")
    print("This is a one-time download — subsequent runs will use the cache.\n")

    try:
        path = download_with_hub_api(args.model)
        print(f"\nDone! Model cached at:\n  {path}")
        print("You can now run:  python run.py")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted — re-run this script to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
