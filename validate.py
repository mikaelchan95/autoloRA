#!/usr/bin/env python3
"""
Pre-flight validation. Run this before starting the optimization loop.

Usage:
    python validate.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = []
warnings = []


def check(condition: bool, label: str, fail_msg: str, warn_only: bool = False):
    if condition:
        print(f"  {PASS} {label}")
    elif warn_only:
        print(f"  {WARN} {label} — {fail_msg}")
        warnings.append(fail_msg)
    else:
        print(f"  {FAIL} {label} — {fail_msg}")
        errors.append(fail_msg)


print("\n=== AutoLoRA Pre-Flight Check ===\n")

print("Files:")
check((ROOT / "config.yaml").exists(), "config.yaml exists", "Missing config.yaml")
check((ROOT / "program.md").exists(), "program.md exists", "Missing program.md")
check(
    (ROOT / "eval_prompts.txt").exists(),
    "eval_prompts.txt exists",
    "Missing eval_prompts.txt",
)
check(
    (ROOT / "run_experiment.py").exists(),
    "run_experiment.py exists",
    "Missing run_experiment.py",
)
check((ROOT / "evaluate.py").exists(), "evaluate.py exists", "Missing evaluate.py")
check(
    (ROOT / "generate_samples.py").exists(),
    "generate_samples.py exists",
    "Missing generate_samples.py",
)

print("\nAI Toolkit:")
toolkit_path = ROOT / "ai-toolkit" / "run.py"
check(
    toolkit_path.exists(),
    "ai-toolkit/run.py found",
    "Run: git clone https://github.com/ostris/ai-toolkit.git",
)

print("\nDataset:")
image_exts = {".png", ".jpg", ".jpeg", ".webp"}
dataset_images = [
    p for p in (ROOT / "dataset").rglob("*") if p.suffix.lower() in image_exts
]
dataset_captions = list((ROOT / "dataset").rglob("*.txt"))
check(
    len(dataset_images) > 0,
    f"Training images ({len(dataset_images)} found)",
    "Add images to dataset/",
)
check(
    len(dataset_captions) > 0,
    f"Caption files ({len(dataset_captions)} found)",
    "Add .txt captions to dataset/",
)
if dataset_images and dataset_captions:
    image_stems = {p.stem for p in dataset_images}
    caption_stems = {p.stem for p in dataset_captions}
    unmatched = image_stems - caption_stems
    check(
        len(unmatched) == 0,
        "All images have captions",
        f"{len(unmatched)} images missing captions: {list(unmatched)[:5]}",
    )

print("\nReference Images:")
ref_images = [
    p for p in (ROOT / "reference_images").rglob("*") if p.suffix.lower() in image_exts
]
check(
    len(ref_images) >= 5,
    f"Reference images ({len(ref_images)} found, need 5+)",
    "Add reference images to reference_images/",
)

print("\nConfig Syntax:")
try:
    import yaml

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    check(True, "config.yaml parses as valid YAML", "")
    check(
        config.get("job") == "extension",
        "job: extension",
        f"job is '{config.get('job')}', expected 'extension'",
    )
except yaml.YAMLError as e:
    check(False, "config.yaml is valid YAML", str(e))
except ImportError:
    check(False, "PyYAML installed", "pip install pyyaml")

print("\nPython Imports:")
imports = {
    "torch": "pip install torch",
    "diffusers": "pip install diffusers",
    "transformers": "pip install transformers",
    "open_clip": "pip install open-clip-torch",
    "yaml": "pip install pyyaml",
}
optional_imports = {
    "insightface": "pip install insightface onnxruntime-gpu",
    "aesthetics_predictor": "pip install simple-aesthetics-predictor",
}

for module, install_hint in imports.items():
    try:
        __import__(module)
        check(True, module, "")
    except ImportError:
        check(False, module, install_hint)

for module, install_hint in optional_imports.items():
    try:
        __import__(module)
        check(True, module, "")
    except ImportError:
        check(False, module, install_hint, warn_only=True)

print("\nGPU:")
try:
    import torch

    has_cuda = torch.cuda.is_available()
    check(
        has_cuda,
        "CUDA available",
        "No GPU detected — training will not work",
        warn_only=False,
    )
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        check(
            vram_gb >= 16,
            f"{gpu_name} ({vram_gb:.0f} GB)",
            f"Only {vram_gb:.0f} GB VRAM — may OOM at higher ranks",
        )
except Exception:
    check(False, "GPU check", "Could not query GPU")

print("\nGit:")
import subprocess

try:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(ROOT)
    )
    check(result.returncode == 0, "Git repo initialized", "Run: git init")
except FileNotFoundError:
    check(False, "Git installed", "Install git")

print()
if errors:
    print(f"{FAIL} {len(errors)} error(s) must be fixed before running.")
    for e in errors:
        print(f"  → {e}")
    sys.exit(1)
elif warnings:
    print(f"{WARN} {len(warnings)} warning(s) — may cause partial failures.")
    print(f"{PASS} Core checks passed. You can start experimenting.")
    sys.exit(0)
else:
    print(f"{PASS} All checks passed. Ready to go.")
    sys.exit(0)
