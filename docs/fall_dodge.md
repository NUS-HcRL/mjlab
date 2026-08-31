# PM1 fall with ground-region avoidance

The `Mjlab-Falling-Flat-PM1-AMP-Dodge` task extends the original fall factory. It
adds a virtual ground region, a five-dimensional actor/critic observation, a
separate contact sensor, and one negative reward. Original robot physics, actions,
fall rewards, termination thresholds, curricula, AMP observations/demonstrations,
and PPO settings are unchanged. The original fall task IDs remain available.

This is a normal policy-training task, not a frozen-policy/residual controller.
Old fall checkpoint migration is **not implemented**: its smaller actor/critic
input layers and observation normalizers cannot simply be resumed into Dodge.
Use newly trained Dodge checkpoints for subsequent resumes. Unchanged settings
do not guarantee unchanged fall performance after training.

## Run

In the project's full training environment (with its AMP dependencies and motion
files), start a new run:

```bash
uv run train Mjlab-Falling-Flat-PM1-AMP-Dodge --env.scene.num-envs 4096
```

Preview the region and observations without a trained policy:

```bash
uv run play Mjlab-Falling-Flat-PM1-AMP-Dodge --agent zero --num-envs 16
```

The red ring is a debug visualization, supported by both native and Viser viewers.
It is only visible in environments with an active region. Play retains the same
20% sampling probability; reset or switch the displayed environment to find one.
The original fall play behavior, including its long episode and deterministic
initial push, is retained. Zero actions do not demonstrate learned avoidance.

Logs use experiment name `pm1_falling_amp_dodge`.

## Configuration

Environment/runner factory: `src/mjlab/tasks/fall/config/pm1/dodge_env_cfg.py`.
Region defaults and implementation: `src/mjlab/tasks/fall/mdp/dodge.py`.
Sampling settings are exposed under `env.commands.dodge` in the training config.

| Setting | Default | Meaning |
| --- | --- | --- |
| `probability` | `0.2` | Probability of requesting a region at reset |
| `radius_range` | `(0.08, 0.14)` m | Radius of the forbidden ground disc |
| `distance_range` | `(0.35, 0.70)` m | Distance along the estimated fall direction |
| `lateral_range` | `0.18` m | Symmetric sideways placement offset |
| `velocity_lookahead_s` | `0.35` s | Scale for initial horizontal velocity |
| `tilt_scale` | `0.5` m | Scale for horizontal projection of the body's up axis |
| `min_root_height` | `0.45` m | Suppress regions for late/low reset states |
| `initial_body_clearance` | `0.18` m | Clearance outside the disc around low body origins |
| `placement_attempts` | `8` | Candidate locations before disabling the region |
| reward `dodge_region_contact.weight` | `-1.0` | Bounded contact cost multiplier |

These are initial implementation defaults, not values selected from W&B tuning.
The actual active fraction can be lower than 20%: late fall resets and candidates
overlapping low bodies are rejected. Existing reset states are never resampled or
modified to accommodate an obstacle. Placement uses initial velocity and tilt as
a heuristic, with random directions for nearly stationary upright resets. It is
not a policy landing predictor and does not guarantee that every region is a
useful or feasible challenge. Low body origins plus clearance are an approximate
initial-overlap filter, not an exact collision-shape test.

The region is sampled **after** robot reset/push events by the command manager.
Its center stays fixed in world coordinates for the entire episode. Observations
are `[active, relative_x, relative_y, relative_z, radius]`; relative position is
in the robot root frame. All five values are zero when inactive. The new term has
no history or injected noise; existing proprioceptive histories/noise are kept.
AMP discriminator inputs are unchanged. Position comes from simulation state;
this does not implement a camera, object detector, or visibility/occlusion model.

## Contact reward and limitations

The separate `dodge_ground_contact` sensor includes all `LINK_*` bodies, including
feet, with eight contact slots per body. The cost is one if **any valid ground
contact point** is inside/on the disc, otherwise zero. Multiple contacts do not
increase this cost. RewardManager multiplies by the negative weight and control
`dt` (currently `0.02` s), so default contribution is `-0.02` per contacting step.
Contact force is logged but does not scale the new reward. The original AMP EMA
mixing still applies and may react to the added task penalty.

There is no physical obstacle, new termination, proximity penalty, directional
pose target, or distant-escape bonus. This models avoiding a ground footprint,
not colliding with an object of nonzero height. The original action-rate reward
does not impose a hard bound on displacement or guarantee a small dodge.

As with existing fall rewards, contact is read once per control step, so contacts
that begin and end between reads may be missed. More than eight simultaneous
contacts on one body can also hide points after reduction. The sensor's `found`
field carries the original match count, allowing slot overflow to be logged; see
the [MuJoCo contact sensor documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact).
This version makes no continuous collision-detection or safety guarantee.

## Metrics and evaluation

The new reward appears under `Episode_Reward/dodge_region_contact`. Additional
completed-episode statistics are under `Metrics/dodge/`:

- `requested_rate`, `active_rate`, `active_episodes`: sampling and valid placement.
- `contact_rate`: fraction of active episodes with at least one detected hit.
- `contact_time_fraction`: average contacting-step fraction in active episodes.
- `peak_region_contact_force`: mean episode peak force inside active regions.
- `slot_overflow_rate`: average fraction of steps exceeding the per-body slots,
  across all completed episodes, including those without regions.

Conditional metrics are zero when a reset batch contains no active episode;
interpret them with `active_episodes`. As with existing task metrics, logging
averages reset-batch summaries, not a globally episode-weighted dataset.

Keep fixed-seed evaluations for both `probability=0` and `probability=1` alongside
the original fall policy. Compare original head/torso/elbow forces, forbidden
terminations, lower-before-upper contact metrics, and action/landing displacement,
in addition to region hits. For avoidance claims, separate cases the original
policy would have hit from cases that were already safe. This evaluation protocol
is not an automated baseline/checkpoint migration tool in this change.

## Tests

```bash
uv run pytest tests/test_fall_dodge.py tests/test_task_configs.py -q
```

The focused tests exercise contact geometry/empty slots, coordinate transforms,
partial resets, fixed world positions, placement rejection, contact statistics,
and preservation of original fall/AMP config values. They require the normal
project Python dependencies even though the tensor checks themselves run on CPU.
