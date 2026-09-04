# PM1 fall with ground-region avoidance

The `Mjlab-Falling-Flat-PM1-AMP-Dodge` task extends the original fall factory. It
adds a virtual ground region, a five-dimensional actor/critic observation with
three-frame history, a separate contact sensor, and two negative rewards. Original robot physics, actions,
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
| `motion_observation_steps` | `3` | Post-reset frames observed before activation |
| `reference_frame` | `yaw` | Yaw-only or full-root relative observation frame |
| `radius_range` | `(0.08, 0.14)` m | Radius of the forbidden ground disc |
| `prediction_height` | `0.35` m | Root height used as the ballistic landing plane |
| `flight_time_range` | `(0.10, 0.80)` s | Bounds on estimated flight time |
| `landing_distance_range` | `(0.35, 0.90)` m | Bounds on predicted displacement |
| `landing_lead_distance` | `0.25` m | Extra distance from the root projection toward the fall direction |
| `placement_jitter` | `0.12` m | Candidate jitter around the prediction |
| `min_root_height` | `0.45` m | Suppress regions for late/low reset states |
| `initial_body_clearance` | `0.18` m | Clearance outside the disc around low body origins |
| `placement_attempts` | `8` | Candidate locations before disabling the region |
| reward `dodge_region_contact.weight` | `-1.0` | Bounded contact cost multiplier |
| reward `dodge_region_proximity.weight` | `-0.05` | Bounded pre-contact risk multiplier |

These are initial implementation defaults, not values selected from W&B tuning.
The actual active fraction can be lower than 20%: late fall resets and candidates
overlapping low bodies are rejected. Existing reset states are never resampled or
modified to accommodate an obstacle. Placement waits for three completed control
frames after reset, blends displacement-derived and current root velocity, then
projects the root ballistically until `prediction_height`. Candidates are jittered
around a point `0.25 m` beyond that root projection in the fall direction, placing
the region closer to the expected hand/forearm landing area. The force pulse can
still be active during this short window;
waiting for its full configured duration could make the region appear after impact.
The estimate omits future policy actions, articulation and contacts, so it is not
a landing guarantee. The low-body clearance check is also an approximation.

The request is sampled after robot reset/push events. The region remains inactive
during motion observation, then stays fixed in world coordinates after activation.
Observations are `[active, relative_x, relative_y, relative_z, radius]` in the
yaw-only `LINK_BASE` frame by default. All five raw values are zero when inactive.
The term keeps three frames (15 flattened values) and has no injected noise;
existing proprioceptive histories/noise are kept.
AMP discriminator inputs are unchanged. Position comes from simulation state;
this does not implement a camera, object detector, or visibility/occlusion model.

## Contact reward and limitations

The separate `dodge_ground_contact` sensor includes all `LINK_*` bodies, including
feet, with four contact slots per body. It exports only `found` and `pos`; the
original fall sensor continues to supply force. The cost is one if **any valid ground
contact point** is inside/on the disc, otherwise zero. Multiple contacts do not
increase this cost. RewardManager multiplies by the negative weight and control
`dt` (currently `0.02` s), so default contribution is `-0.02` per contacting step.
The bounded proximity term provides earlier credit only while a body is descending,
near the floor, and horizontally near the active disc. Its initial `-0.05` weight
is deliberately small relative to the dominant cached base-run reward terms. The
original AMP EMA mixing still applies and may react to the added task penalty.

There is no physical obstacle, new termination, directional pose target, or
distant-escape bonus. This models avoiding a ground footprint,
not colliding with an object of nonzero height. The original action-rate reward
does not impose a hard bound on displacement or guarantee a small dodge.

As with existing fall rewards, contact is read once per control step, so contacts
that begin and end between reads may be missed. More than four simultaneous
contacts on one body can also hide points after reduction. The sensor's `found`
field carries the original match count, allowing slot overflow to be logged; see
the [MuJoCo contact sensor documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact).
This version makes no continuous collision-detection or safety guarantee.

## Metrics and evaluation

The new rewards appear under `Episode_Reward/dodge_region_contact` and
`Episode_Reward/dodge_region_proximity`. Additional
completed-episode statistics are under `Metrics/dodge/`:

- `requested_rate`, `active_rate`: sampling and valid placement.
- `contact_rate`: fraction of active episodes with at least one detected hit.
- `contact_time_fraction`: average contacting-step fraction in active episodes.
- `slot_overflow_rate`: average fraction of steps exceeding the per-body slots,
  across all completed episodes, including those without regions.

Conditional metrics are zero when a reset batch contains no active episode;
interpret them with `active_rate`. As with existing task metrics, logging
averages reset-batch summaries, not a globally episode-weighted dataset.

During training analysis, inspect region hits together with the existing
head/torso/elbow forces, forbidden terminations, lower-before-upper contact metrics,
and action/landing displacement. No separate active-versus-inactive comparison
metrics are added by this change.

## Tests

```bash
uv run pytest tests/test_fall_dodge.py tests/test_task_configs.py -q
```

The focused tests exercise contact geometry/empty slots, coordinate transforms,
partial resets, fixed world positions, placement rejection, contact statistics,
and preservation of original fall/AMP config values. They require the normal
project Python dependencies even though the tensor checks themselves run on CPU.
