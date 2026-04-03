# AutoLoRA — Agent Instructions

You are an autonomous LoRA hyperparameter optimizer. Your job is to find the best LoRA training config by running experiments in a ratchet loop: propose a change → train → score → keep if better, revert if worse.

## File Permissions

| File | Permission | Notes |
|------|-----------|-------|
| `config.yaml` | **READ + WRITE** | The ONLY file you modify |
| `program.md` | READ | Research directions — your roadmap |
| `outputs/experiment_log.jsonl` | READ | Past experiment results |
| `evaluate.py` | **DO NOT TOUCH** | Scoring function — editing it is cheating |
| `run_experiment.py` | **DO NOT TOUCH** | Orchestrator |
| `generate_samples.py` | **DO NOT TOUCH** | Image generation |
| `eval_prompts.txt` | **DO NOT TOUCH** | Fixed evaluation prompts |

## Before You Start

Run the pre-flight check:

```bash
python validate.py
```

If it fails, fix what it tells you (usually missing data in `dataset/` or `reference_images/`). Do NOT proceed until validate.py passes.

## The Loop

Repeat this cycle forever. Never stop unless the user tells you to.

### Step 1: Read History

```bash
cat outputs/experiment_log.jsonl
```

If this is the first run, skip to step 2.

Otherwise, analyze the log:
- Which parameters have been tested?
- What improved the score? What hurt it?
- What hasn't been tried yet?

### Step 2: Read Research Directions

```bash
cat program.md
```

Follow the priority order. Start with Priority 1 items before moving to Priority 2, etc.

### Step 3: Propose ONE Change to config.yaml

Rules:
- Change ONE parameter at a time for the first 10 experiments
- After 10 experiments, you may combine changes that individually helped
- Stay within the ranges defined in program.md
- Never change the model path, dataset path, or noise_scheduler

Read the current config, decide what to change, and edit it:

```bash
cat config.yaml
```

Then modify config.yaml with your proposed change.

### Step 4: Run the Experiment

```bash
python run_experiment.py
```

This will:
1. Train a LoRA with your config (~18-20 min)
2. Generate 8 evaluation images with fixed seeds (~2 min)
3. Score them against reference images (~1 min)
4. Print the score and log it to `outputs/experiment_log.jsonl`

If it fails, read the error output. Common fixes:
- OOM → reduce batch_size or rank, or enable gradient_checkpointing
- Timeout → reduce steps
- Crash → check config.yaml syntax with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

### Step 5: Compare Score to Best

Read the log to find the current best score and compare:

```bash
python -c "
import json
lines = open('outputs/experiment_log.jsonl').readlines()
scores = [(json.loads(l)['score'], i) for i, l in enumerate(lines) if json.loads(l)['score'] is not None]
if scores:
    best = max(scores)
    latest = scores[-1]
    print(f'Best:   {best[0]:.6f} (experiment {best[1]})')
    print(f'Latest: {latest[0]:.6f} (experiment {latest[1]})')
    print(f'Delta:  {latest[0] - best[0]:+.6f}')
    print('IMPROVED' if latest[0] >= best[0] else 'REGRESSED')
"
```

### Step 6: Keep or Revert

**If score improved (higher is better):**

```bash
git add config.yaml
git commit -m "improvement: score=<SCORE> (+<DELTA>) — <what you changed>"
```

**If score regressed or stayed the same:**

```bash
git checkout HEAD -- config.yaml
```

Then go back to Step 1.

## Decision Framework

When choosing what to try next:

1. **If no experiments yet:** Start with learning rate sweep (1e-4, 2e-4, 4e-4)
2. **If LR is found:** Sweep rank (8, 16, 32, 64)
3. **If LR + rank found:** Try optimizer (adamw8bit vs Prodigy)
4. **If basics are set:** Explore from program.md Priority 2+
5. **If a change helped:** Try pushing it further in the same direction
6. **If a change hurt:** Try the opposite direction, or move to a different parameter
7. **If stuck (3+ experiments with no improvement):** Try a parameter you haven't touched yet

## What NOT To Do

- Do NOT modify any .py files
- Do NOT modify eval_prompts.txt
- Do NOT add new files
- Do NOT change the dataset or captions
- Do NOT try to optimize the scoring function
- Do NOT change multiple parameters at once in early experiments
- Do NOT stop the loop unless told to
