# LoRA Optimization — Research Directions

## Search Space

These are the parameters you are allowed to change in `config.yaml`, with their valid ranges.

| Parameter | YAML Path | Range | Default |
|-----------|-----------|-------|---------|
| Learning rate | `train.lr` | `1e-5` → `5e-4` | `1e-4` |
| LoRA rank | `network.linear` | 4, 8, 16, 32, 64, 128 | 16 |
| LoRA alpha | `network.linear_alpha` | 0.5× to 2× of rank | 16 |
| Optimizer | `train.optimizer` | `adamw8bit`, `prodigy`, `adafactor` | `adamw8bit` |
| Training steps | `train.steps` | 500 → 2000 | 1000 |
| Batch size | `train.batch_size` | 1, 2, 3, 4 | 1 |
| Grad accumulation | `train.gradient_accumulation_steps` | 1, 2, 4 | 1 |
| Gradient checkpointing | `train.gradient_checkpointing` | true, false | true |
| Train text encoder | `train.train_text_encoder` | true, false | false |
| Resolution | `datasets[0].resolution` | [512], [512, 768], [512, 768, 1024] | [512, 768, 1024] |
| Caption dropout | `datasets[0].caption_dropout_rate` | 0.0 → 0.1 | 0.05 |
| EMA decay | `train.ema_config.ema_decay` | 0.9 → 0.999 | 0.99 |

## Do NOT Change

These fields must stay fixed across all experiments:

- `model.name_or_path` — base model
- `model.is_flux` — architecture flag
- `model.quantize` — quantization setting
- `train.noise_scheduler` — must stay `flowmatch`
- `datasets[0].folder_path` — dataset location
- `train.dtype` — precision

## Priority 1: Core Hyperparameters

Explore these first. They have the most impact.

1. Sweep learning rate: try `1e-4`, `2e-4`, `4e-4` (one at a time)
2. Once best LR is found, sweep rank: 8, 16, 32, 64
3. Test alpha at 0.5×, 1×, and 2× of the winning rank
4. Compare optimizer: `adamw8bit` vs `prodigy`

## Priority 2: Architecture

After core hyperparams are settled:

1. Try `train_text_encoder: true` vs `false`
2. Test different resolution sets: `[512, 768]` vs `[512, 768, 1024]`
3. Try `caption_dropout_rate`: 0.0, 0.05, 0.1

## Priority 3: Training Dynamics

Fine-tuning after architecture is decided:

1. Sweep step count: 500, 750, 1000, 1500 (find overfit boundary)
2. Try batch_size 2 with gradient_accumulation_steps 1 vs batch_size 1 with gradient_accumulation_steps 2
3. Try gradient_checkpointing off (faster but more VRAM)
4. Sweep EMA decay: 0.99 vs 0.999 vs disabled (`use_ema: false`)

## Priority 4: Combinations

After 10+ single-parameter experiments, combine the winners:

1. Take the best value for each parameter that improved the score
2. Apply them together in one config
3. If the combination is worse than expected, bisect to find conflicts

## Constraints

- One parameter change at a time for the first 10 experiments
- Keep total training time under 20 minutes per run
- If OOM: reduce batch_size or rank first, enable gradient_checkpointing second
- Always use quantized base model (VRAM is shared with eval pipeline)

## What "Better" Means

Higher composite score = better. The score is a weighted sum of:

| Metric | Weight | Target |
|--------|--------|--------|
| CLIP-I (image similarity to reference) | 0.35 | Subject should be recognizable |
| CLIP-T (text-image alignment) | 0.25 | Prompt adherence matters |
| Aesthetic quality | 0.20 | No artifacts, good composition |
| ArcFace identity | 0.15 | Face should match reference |
| Output diversity | 0.05 | Different prompts should produce different images |

Watch for:
- Mode collapse: all outputs look the same → diversity score drops
- Style leak: subject's style bleeds into background/scene
- Overfitting: high CLIP-I but low CLIP-T (ignoring the prompt)
