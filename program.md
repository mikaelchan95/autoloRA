# LoRA Optimization Research Directions

## Priority 1: Core Hyperparameters
- Find optimal rank × alpha ratio for this subject
- Explore LR schedules — does cosine beat constant?
- Test learning rates in the 1e-4 to 4e-4 range first
- Try Prodigy optimizer (adaptive LR) vs AdamW with manual schedule

## Priority 2: Architecture
- Try different layer targeting strategies
- Test if training text encoder helps or hurts for this subject
- Compare full transformer vs single_transformer_blocks only
- Experiment with different LoRA rank values: 4, 8, 16, 32, 64, 128

## Priority 3: Training Dynamics
- Compare Prodigy vs AdamW convergence speed at 1000 steps
- Try noise offset values 0.0 → 0.1 in increments of 0.02
- Explore step counts: find the sweet spot before overfit (500, 750, 1000, 1500, 2000)
- Test batch size 1 vs 2 (we have VRAM headroom on 5090)
- Try min-SNR gamma weighting: 0, 1, 5

## Priority 4: Regularization & Stability
- Test gradient checkpointing on vs off (speed vs VRAM tradeoff)
- Explore weight decay values: 0.0, 0.01, 0.1
- Try gradient clipping: 0.0 (off), 1.0

## Constraints
- Do NOT change the dataset or captions
- Do NOT modify evaluate.py or run_experiment.py
- Keep total training time under 20 minutes per run
- Always use quantized base model (we need VRAM for eval too)
- One parameter change at a time for the first 10 experiments, then combinations

## What "Better" Means
- Subject should be recognizable across diverse prompts
- Style should NOT leak into non-subject elements
- Prompt adherence matters — if the prompt says "beach", show a beach
- Image quality should remain high (no artifacts, no color shifts)
- Diversity across different prompts (not mode collapse)
