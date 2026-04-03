#!/usr/bin/env python3
"""
AutoLoRA — Experiment Orchestrator (Ratchet Loop)

This file is READ-ONLY. The optimization agent must NOT modify it.
It runs the train → generate → score → keep/revert loop.

Usage:
    python run_experiment.py                    # Run one experiment
    python run_experiment.py --loop             # Run ratchet loop indefinitely
    python run_experiment.py --loop --max-runs 20  # Run N experiments then stop
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "config.yaml"
EVAL_PROMPTS = ROOT / "eval_prompts.txt"
REFERENCE_DIR = ROOT / "reference_images"
OUTPUT_DIR = ROOT / "outputs"
EVAL_IMAGES_DIR = OUTPUT_DIR / "eval_images"
LOG_PATH = OUTPUT_DIR / "experiment_log.jsonl"
OSTRIS_RUN = ROOT / "ai-toolkit" / "run.py"  # Clone ostris/ai-toolkit here

# Hard timeout per training run (seconds). 25 min = 1500s.
TRAIN_TIMEOUT = 1500
# Hard timeout per generation pass (seconds). 5 min = 300s.
GENERATE_TIMEOUT = 300
# Hard timeout per evaluation pass (seconds).
EVAL_TIMEOUT = 120


def load_config() -> dict:
    """Load the current Ostris YAML config."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recently modified .safetensors file in output tree."""
    safetensors = sorted(
        output_dir.rglob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return safetensors[0] if safetensors else None


def train(config_path: Path) -> bool:
    """Run Ostris AI Toolkit training. Returns True on success."""
    if not OSTRIS_RUN.exists():
        print(f"ERROR: Ostris run.py not found at {OSTRIS_RUN}")
        print("Clone the repo: git clone https://github.com/ostris/ai-toolkit.git")
        return False

    cmd = [sys.executable, str(OSTRIS_RUN), "--config", str(config_path)]
    print(f"\n{'=' * 60}")
    print(f"TRAINING — {datetime.now().isoformat()}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            cmd,
            timeout=TRAIN_TIMEOUT,
            check=True,
            cwd=str(ROOT),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Training exceeded {TRAIN_TIMEOUT}s timeout")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        return False


def generate(checkpoint_path: Path) -> bool:
    """Generate evaluation images from the trained checkpoint."""
    EVAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "generate_samples.py"),
        "--model",
        str(checkpoint_path),
        "--prompts",
        str(EVAL_PROMPTS),
        "--output",
        str(EVAL_IMAGES_DIR),
    ]
    print(f"\n{'=' * 60}")
    print(f"GENERATING — {checkpoint_path.name}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            cmd,
            timeout=GENERATE_TIMEOUT,
            check=True,
            cwd=str(ROOT),
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"ERROR: Generation failed — {e}")
        return False


def evaluate() -> float | None:
    """Run evaluation and return the composite score."""
    cmd = [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--generated",
        str(EVAL_IMAGES_DIR),
        "--reference",
        str(REFERENCE_DIR),
    ]
    print(f"\n{'=' * 60}")
    print(f"EVALUATING")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            cmd,
            timeout=EVAL_TIMEOUT,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # evaluate.py prints a single float to stdout
        score = float(result.stdout.strip().split("\n")[-1])
        return score
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        print(f"ERROR: Evaluation failed — {e}")
        return None


def log_experiment(score: float | None, config: dict, notes: str = "") -> None:
    """Append experiment result to JSONL log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "notes": notes,
        "config": config,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nLogged: score={score}  notes={notes}")


def git_commit(message: str) -> bool:
    """Commit config.yaml with the given message."""
    try:
        subprocess.run(["git", "add", "config.yaml"], check=True, cwd=str(ROOT))
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True,
            cwd=str(ROOT),
        )
        return True
    except subprocess.CalledProcessError:
        return False


def git_revert() -> bool:
    """Revert config.yaml to the last committed version."""
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", "config.yaml"],
            check=True,
            cwd=str(ROOT),
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_one_experiment() -> float | None:
    """Execute a single train → generate → score cycle."""
    config = load_config()

    # 1. Train
    if not train(CONFIG_PATH):
        log_experiment(None, config, notes="training_failed")
        return None

    # 2. Find checkpoint
    # Ostris outputs to output/<name>/ by default
    experiment_name = config.get("config", {}).get("name", "autolora_experiment")
    checkpoint_dir = ROOT / "output" / experiment_name
    checkpoint = find_latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(OUTPUT_DIR)

    if checkpoint is None:
        print("ERROR: No .safetensors checkpoint found after training")
        log_experiment(None, config, notes="no_checkpoint")
        return None

    print(f"Checkpoint: {checkpoint}")

    # 3. Generate eval images
    if not generate(checkpoint):
        log_experiment(None, config, notes="generation_failed")
        return None

    # 4. Score
    score = evaluate()
    if score is None:
        log_experiment(None, config, notes="evaluation_failed")
        return None

    log_experiment(score, config)
    return score


def ratchet_loop(max_runs: int | None = None) -> None:
    """
    Ratchet loop: run experiments, keep improvements, revert failures.

    The agent modifies config.yaml BETWEEN calls to this function's iterations.
    In practice, the agent process watches for this script to finish one cycle,
    then edits config.yaml, then signals to run the next cycle.

    For fully autonomous operation, the agent is expected to:
    1. Read experiment_log.jsonl
    2. Read program.md for research directions
    3. Modify config.yaml
    4. Run this script with --loop
    """
    print("\n" + "=" * 60)
    print("AUTOLORA RATCHET LOOP")
    print(f"Max runs: {max_runs or 'unlimited'}")
    print("=" * 60)

    # Run baseline
    print("\n>>> Running baseline experiment...")
    best_score = run_one_experiment()

    if best_score is None:
        print("FATAL: Baseline experiment failed. Fix config.yaml and retry.")
        sys.exit(1)

    print(f"\nBaseline score: {best_score}")
    git_commit(f"baseline: score={best_score:.6f}")

    run_count = 1

    while True:
        if max_runs and run_count >= max_runs:
            print(f"\nReached max runs ({max_runs}). Stopping.")
            break

        run_count += 1
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {run_count} — best_score={best_score:.6f}")
        print(f"{'=' * 60}")

        # The agent should have modified config.yaml by now.
        # In autonomous mode, we pause here for the agent to make changes.
        # For now, we just run with whatever config.yaml currently has.

        score = run_one_experiment()

        if score is None:
            print("Experiment failed. Reverting config.")
            git_revert()
            log_experiment(None, load_config(), notes="reverted_after_failure")
            continue

        if score > best_score:
            improvement = score - best_score
            print(
                f"\n✓ IMPROVEMENT: {best_score:.6f} → {score:.6f} (+{improvement:.6f})"
            )
            best_score = score
            git_commit(f"improvement: score={score:.6f} (+{improvement:.6f})")
        else:
            regression = best_score - score
            print(f"\n✗ REGRESSION: {best_score:.6f} → {score:.6f} (-{regression:.6f})")
            git_revert()
            log_experiment(score, load_config(), notes=f"reverted: -{regression:.6f}")

        # Brief pause to let the agent modify config.yaml
        print("\nWaiting 10s for agent to propose next config change...")
        time.sleep(10)

    print(f"\n{'=' * 60}")
    print(f"LOOP COMPLETE — Best score: {best_score:.6f} over {run_count} experiments")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="AutoLoRA Experiment Orchestrator")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run ratchet loop (indefinitely or until --max-runs)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of experiments in loop mode",
    )
    args = parser.parse_args()

    # Sanity checks
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found")
        sys.exit(1)
    if not EVAL_PROMPTS.exists():
        print(f"ERROR: {EVAL_PROMPTS} not found")
        sys.exit(1)
    if not any(REFERENCE_DIR.iterdir()):
        print(f"WARNING: {REFERENCE_DIR} is empty — evaluation will fail")

    if args.loop:
        ratchet_loop(max_runs=args.max_runs)
    else:
        score = run_one_experiment()
        if score is not None:
            print(f"\nFinal score: {score}")
        else:
            print("\nExperiment failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
