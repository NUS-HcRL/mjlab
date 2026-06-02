# AGENTS.md

## Training Monitoring And Tuning Workflow

When the user asks Codex to analyze training, monitor W&B data, tune RL/AMP
settings, or review fall-task training behavior, follow this workflow.

### 1. Refresh W&B Data First

Use the lightweight `RL` conda environment and the local diagnostics script:

```bash
conda run -n RL python scripts/wandb_amp_diagnostics.py --group pm1_falling_amp --tail 200 --samples 5000
```

If the user gives a specific run, use:

```bash
conda run -n RL python scripts/wandb_amp_diagnostics.py --run-path ENTITY/PROJECT/RUN_ID --tail 200 --samples 5000
```

The script writes data under:

```text
outputs/wandb_runs/<project>/<run_name>/
```

Read at least:

```text
diagnostics.md
history.csv
config.json
run.json
```

Use `--samples 5000` by default because full W&B history scans can be slow.
Only use a full download when the user explicitly asks for it or when sampled
data is insufficient.

### 2. Metrics To Inspect

For AMP/PPO tuning, inspect:

```text
Loss/policy_pred
Loss/expert_pred
Loss/accuracy_policy
Loss/accuracy_expert
Loss/amp_loss
Loss/grad_pen_loss
Loss/mean_kl_divergence
Debug/ratio_mean
Debug/ratio_std
Debug/actor_grad_norm
Policy/mean_noise_std
Train/mean_reward
Train/mean_task_reward
Train/mean_style_reward
Train/style_reward_std
Train/style_balance_scale
Train/task_weight_scale
```

For `src/mjlab/tasks/fall/fall_env_cfg.py`, inspect:

```text
Episode_Reward/*
Episode_Termination/*
Metrics/contact_force/*
Metrics/termination_force/*
Metrics/lower_then_upper/*
Train/*
```

Do not treat `Curriculum/*` or `Perf/*` as important by default for tuning
recommendations. Only mention them if the user asks or if they clearly explain
a training change.

### 3. Summarize Before Suggesting Changes

Always summarize the latest run before recommending edits:

- W&B run path, run name, state, and sampled row count.
- Tail-window values for the key metrics.
- Which reward terms dominate the total task reward.
- Whether AMP discriminator is too weak, usable, or saturated.
- Whether style reward appears to dominate task reward.
- Whether contact-force penalties are caused by high, medium, low, or specific
  tracked bodies.
- Whether forbidden contact terminations are mostly head, torso, left elbow, or
  right elbow.
- Whether `lower_then_upper_contact` shows lower-body buffering:
  `lower_contact_rate`, `upper_before_lower_rate`, `early_upper_rate`,
  `timely_upper_rate`, `late_upper_rate`, and `upper_delay_mean`.

Be concrete. Use metric names and numbers from `diagnostics.md` or `history.csv`.

### 4. Give Candidate Tuning Options, Do Not Apply Them Yet

After analysis, provide tuning suggestions as numbered options. Each option must
include:

- The config file and setting to change.
- The proposed value.
- The evidence from W&B metrics.
- The expected effect.
- The risk or tradeoff.

Do not edit files immediately after giving recommendations. Wait for the user to
choose which suggestions to apply.

Examples of acceptable recommendation shape:

```text
1. Reduce style dominance in rl_cfg.py
   Change reward_mix_scale_clip from (0.25, 4.0) to (0.10, 4.0).
   Evidence: Train/style_balance_scale is pinned near the lower clip.
   Expected effect: allows EMA mixing to reduce style reward further.
   Risk: policy may drift toward task reward and away from demo style.
```

### 5. Apply Only Selected Changes

When the user chooses one or more suggestions:

- Apply only the selected changes.
- Preserve unrelated local edits.
- Re-read the target file before editing.
- Use `apply_patch` for manual edits.
- Run a lightweight verification such as `python -m py_compile` on touched
  Python files.

If tests or imports require unavailable heavy dependencies, say so clearly and
report the checks that did run.

### 6. Keep Download Script Current

The W&B diagnostics script should automatically include these prefixes:

```text
Episode_Reward/
Episode_Termination/
Metrics/
Train/
```

Do not add `Curriculum/` or `Perf/` back to the default downloaded prefixes
unless the user asks for them.

### 7. Default Interpretation Heuristics

Use these as starting heuristics, not hard rules:

- If `Loss/expert_pred - Loss/policy_pred` is very small and accuracies are near
  chance, the discriminator is probably weak.
- If discriminator accuracies are very high and style reward is noisy, the
  discriminator may be too strong or too sharp.
- If `Train/style_balance_scale` is pinned near its lower clip, style reward is
  probably dominating task reward.
- If `Debug/ratio_std` is high, PPO updates may be too aggressive for the
  current reward mix.
- If `Metrics/contact_force/high_max` or tracked head/torso/elbow values are
  high, adjust contact penalties, reset curriculum, or forbidden thresholds with
  care.
- If `Metrics/lower_then_upper/upper_before_lower_rate` or
  `early_upper_rate` is high, the robot is not reliably using lower-body
  buffering before upper-body contact.

Always tie these judgments back to the latest downloaded metrics.
