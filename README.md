# AutoLoRA

Autonomous LoRA hyperparameter optimization using the [autoresearch](https://github.com/karpathy/autoresearch) ratchet loop pattern and [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) for training.

An AI agent proposes config changes, trains a LoRA, scores the result, and keeps improvements automatically — overnight, unattended.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  1. Agent reads program.md + experiment log             │
│  2. Agent modifies config.yaml (the ONLY editable file) │
│  3. run_experiment.py trains → generates → scores       │
│  4. Score improved? git commit. Worse? git revert.      │
│  5. Repeat.                                             │
└─────────────────────────────────────────────────────────┘
```

**Cycle time:** ~25 min per experiment on an RTX 5090.
**Overnight (8 hrs):** ~18-20 experiments.
**Weekend (48 hrs):** ~100+ experiments.

## Quickstart

```bash
# 1. Clone this repo
git clone https://github.com/youruser/autoloRA.git
cd autoloRA

# 2. Run setup (clones ai-toolkit, installs deps, validates env)
chmod +x setup.sh
./setup.sh

# 3. Add your data
#    - Training images + .txt captions → dataset/
#    - Reference images of your subject → reference_images/
#    See dataset/README.md and reference_images/README.md for format details.

# 4. Edit config.yaml
#    - Set model path (default: FLUX.1-dev)
#    - Adjust dataset path if needed

# 5. Edit eval_prompts.txt
#    - Replace "sks" with your trigger word if different

# 6. Run baseline experiment
python run_experiment.py

# 7. Start the ratchet loop
python run_experiment.py --loop

# Or limit to N experiments:
python run_experiment.py --loop --max-runs 20
```

## Repo Structure

```
autoloRA/
├── config.yaml              ← Agent modifies this (search space)
├── program.md               ← Human-authored research directions
├── eval_prompts.txt         ← Fixed prompts + seeds for scoring
│
├── run_experiment.py        ← Orchestrator: train → generate → score → keep/revert
├── evaluate.py              ← Composite scorer (CLIP-I, CLIP-T, aesthetic, ArcFace)
├── generate_samples.py      ← Deterministic inference with fixed seeds
│
├── setup.sh                 ← One-command setup
├── requirements.txt         ← Python dependencies
│
├── dataset/                 ← Your training images + captions
├── reference_images/        ← Target images for evaluation
├── outputs/                 ← Checkpoints, eval images, experiment log (gitignored)
└── ai-toolkit/              ← Ostris AI Toolkit (cloned by setup.sh, gitignored)
```

**Read-only files:** `evaluate.py`, `run_experiment.py`, `generate_samples.py` are `chmod 444`. The agent must not modify these — if it can edit the scoring function, it will game the metric instead of improving the LoRA.

## Scoring

Generated images are scored against reference images using a weighted composite:

| Metric | Weight | What It Measures |
|--------|--------|------------------|
| CLIP-I (image similarity) | 0.35 | Does the output look like the subject? |
| CLIP-T (text-image alignment) | 0.25 | Does the output match the prompt? |
| Aesthetic predictor | 0.20 | Is the image generally high quality? |
| ArcFace identity | 0.15 | Face identity preservation |
| Diversity penalty | 0.05 | Penalizes outputs that all look the same |

All diagnostic output goes to stderr. Only the final scalar score is printed to stdout, which `run_experiment.py` reads.

## Search Space

Parameters the agent can tune in `config.yaml`:

| Parameter | Range | Notes |
|-----------|-------|-------|
| Learning rate | `1e-5` → `5e-4` | Most impactful lever |
| LoRA rank | 4, 8, 16, 32, 64, 128 | Higher = more capacity, more VRAM |
| LoRA alpha | 0.5× to 2× of rank | Controls effective LR scaling |
| LR scheduler | constant, cosine, linear | Cosine often wins |
| Optimizer | adamw8bit, Prodigy, Adafactor | Prodigy is adaptive |
| Training steps | 500 → 2000 | Find the sweet spot before overfit |
| Batch size | 1–4 | Higher needs more VRAM |
| Resolution | 512, 768, 1024 | Match your target model |
| Gradient checkpointing | on / off | Saves VRAM at speed cost |
| Text encoder training | on / off | Can help or hurt |

## Requirements

- **GPU:** 24+ GB VRAM (tested on RTX 5090 32 GB)
- **Python:** 3.10+
- **Model:** FLUX.1-dev (default), FLUX.1-schnell, or FLUX.2 variants

Key dependencies:
- [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) — LoRA training
- [open-clip-torch](https://github.com/mlfoundations/open_clip) — CLIP scoring
- [insightface](https://github.com/deepinsight/insightface) — ArcFace identity
- [aesthetic-predictor-v2](https://github.com/discus0434/aesthetic-predictor-v2-5) — Image quality

## Using with an AI Agent

Point Claude Code, Cursor, or Aider at this repo with `program.md` as context. The agent should:

1. Read `program.md` for research directions and constraints
2. Read `outputs/experiment_log.jsonl` for past results
3. Modify **only** `config.yaml`
4. The ratchet loop handles training, evaluation, and git

## Tips

- **Calibrate scoring first.** Run 3-5 manual experiments with known good/bad LoRAs to validate that the composite metric ranks them correctly before running overnight.
- **Fixed seeds are critical.** Without them, variance between generations makes comparison impossible.
- **Start narrow.** Lock everything except LR and rank for the first night. Add more variables once you trust the eval pipeline.
- **Watch for mode collapse.** If the agent over-optimizes CLIP-I, outputs become near-copies of references. The diversity penalty helps, but monitor visually.
- **Log everything.** The JSONL experiment log is gold for post-hoc analysis.

## License

MIT
