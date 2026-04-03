#!/usr/bin/env python3
"""
AutoLoRA — Sample Generator

Generates evaluation images from a trained LoRA checkpoint using fixed seeds
and prompts. Deterministic output enables fair comparison across experiments.

This file is READ-ONLY. The optimization agent must NOT modify it.

Usage:
    python generate_samples.py \
        --model outputs/checkpoint.safetensors \
        --prompts eval_prompts.txt \
        --output outputs/eval_images/
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.0


def parse_prompts_file(prompts_path: Path) -> list[tuple[int, str]]:
    """
    Parse eval_prompts.txt into [(seed, prompt), ...].

    Format: seed | prompt
    Lines starting with # are comments.
    """
    entries = []
    with open(prompts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^(\d+)\s*\|\s*(.+)$", line)
            if match:
                seed = int(match.group(1))
                prompt = match.group(2).strip()
                entries.append((seed, prompt))
            else:
                print(f"WARNING: Skipping malformed line: {line}")
    return entries


def load_pipeline(
    base_model: str,
    lora_path: Path,
    device: str = "cuda",
) -> FluxPipeline:
    """Load Flux pipeline with LoRA weights."""
    print(f"Loading base model: {base_model}")
    pipe = FluxPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA: {lora_path}")
    pipe.load_lora_weights(str(lora_path))

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_images(
    pipe: FluxPipeline,
    prompts: list[tuple[int, str]],
    output_dir: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
) -> list[Path]:
    """Generate images with fixed seeds. Returns list of output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for i, (seed, prompt) in enumerate(prompts):
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"[{i + 1}/{len(prompts)}] seed={seed} prompt={prompt[:60]}...")

        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save as seed_index.png for easy identification
        out_path = output_dir / f"seed{seed}_{i:02d}.png"
        image.save(out_path)
        generated.append(out_path)

        print(f"  → {out_path}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="AutoLoRA Sample Generator")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .safetensors LoRA checkpoint",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to eval_prompts.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model name or path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Inference steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE})",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        sys.exit(1)

    # Parse prompts
    prompts = parse_prompts_file(prompts_path)
    if not prompts:
        print("ERROR: No valid prompts found")
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts")

    # Load pipeline
    pipe = load_pipeline(args.base_model, model_path)

    # Generate
    generated = generate_images(
        pipe,
        prompts,
        output_dir,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance_scale=args.guidance,
    )

    # Cleanup GPU memory for subsequent evaluation
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nGenerated {len(generated)} images in {output_dir}")


if __name__ == "__main__":
    main()
