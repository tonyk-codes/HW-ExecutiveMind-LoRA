"""Download the published LoRA adapter without committing model weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("LORA_ADAPTER", "tony-dev1/HW-ExecutiveMind-LoRA-Qwen3.6-27B"),
        help="Hugging Face adapter repository",
    )
    parser.add_argument(
        "--output-dir",
        default="models/executive-mindset-lora",
        help="Local directory for the adapter files",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        token=token,
        allow_patterns=["adapter_config.json", "adapter_model*.safetensors", "*.json", "*.jinja"],
    )
    print(f"Downloaded {args.repo_id} to {output_dir}")
    print(f"Set lora_adapter to {output_dir} in config.json for local loading.")


if __name__ == "__main__":
    main()