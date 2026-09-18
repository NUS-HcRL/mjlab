# PM1 ground-region avoidance

The task `Mjlab-Falling-Flat-PM1-AMP-Dodge` adds virtual ground regions, 15
observation dimensions (five values with three-frame history), contact costs and
landing-clearance shaping. Original fall task factories remain unchanged.

Dodge overrides impact cost to `2.0` and action-rate cost to `-0.15`. It removes
`forbidden_body_contact_force` termination and its event penalty, but retains
nonfinite/invalid-physics guards and time limits. There are no hard head/torso/elbow
impact limits in Dodge; a soft reward cannot guarantee safe impacts. The broad
invalid-physics guard still rejects extreme forces/speeds. Region request
probability stays at 0.5, AMP mix clip at (0.02, 0.25), and KL early stopping is
disabled. The two extra dodge reference clips remain included.

## Run

In the full project environment:

```bash
uv run train Mjlab-Falling-Flat-PM1-AMP-Dodge --env.scene.num-envs 4096
uv run play Mjlab-Falling-Flat-PM1-AMP-Dodge --agent zero --num-envs 16
```

The red ring is a debug visualization, not collision geometry. Play requests
regions with the same probability but retains its original reset/push behavior.
Zero actions do not demonstrate avoidance. Logs use `pm1_falling_amp_dodge`.
This is ordinary policy training, not a residual controller. Old fall checkpoint
migration is not provided.

## Placement

Factory: `src/mjlab/tasks/fall/config/pm1/dodge_env_cfg.py`.
Implementation: `src/mjlab/tasks/fall/mdp/dodge.py`.

| Setting | Default | Meaning |
| --- | --- | --- |
| probability | 0.5 (task override) | Request probability |
| motion_observation_steps | 3 | Complete unforced frames after pulse ends |
| radius_range | (0.08, 0.14) m | Unchanged disc radius |
| prediction_height | 0.10 m | Link landing plane |
| min_reaction_time | 0.12 s | Minimum estimated reaction time |
| max_flight_time | 0.65 s | Extrapolation cap |
| max_observation_time | 0.8 s | Unforced observation window |
| direction_lookahead | 0.15 s | Tilt/rotation/motion direction estimate |
| min_tilt / min_tipping_speed | 0.10 / 0.30 | Horizontal up-vector magnitude or tipping speed confirming a fall |
| placement_jitter | 0.12 m | Jitter around selected limb projection |
| min_root_height | 0.45 m | Reject low/late states |
| initial_body_clearance | 0.18 m | Clearance beyond radius from low bodies |
| placement_attempts | 8 | Candidates per eligible observation frame |

Placement waits for the pulse to end and three complete unforced steps.
Tilt, world angular velocity and observed/current translation estimate direction;
upright translation alone does not confirm a fall. A descending elbow pitch/end
or knee in that direction is selected with sufficient estimated reaction time.
Candidates lie near that link's ballistic landing point, not a fixed base offset
or the average between hands. Ambiguous/overlapping candidates are retried within
the observation window; late/low states are rejected. Active fraction can thus
be below 50%. Robot reset states are never altered to accommodate a region.

Once active, the region is world-fixed for the episode. Observation is
`[active, relative_x, relative_y, relative_z, radius]` in yaw-only LINK_BASE
coordinates, with three frames and no added noise. Inactive raw values are zero.
Existing proprioception stays unchanged. This uses simulation state, not camera
perception or visibility modeling.

## Rewards

- `dodge_region_contact`, weight -1: any valid contact point inside/on the disc
  incurs a per-step cost (currently -0.02 after dt scaling).
- `dodge_region_first_contact`, weight -0.75: fixed first-hit cost per episode.
- `dodge_landing_clearance`, weight +1: replaces the old predicted-risk term.

Clearance shaping projects torso, knees and elbow pitch/end links up to 0.65 s.
Grounded links use their current footprint. Head is excluded as requested;
head force still contributes to impact cost. Clearance is distance to center
minus radius and a 0.04 m body margin. Potential is
`Phi = -max(sigmoid((0.05 - clearance) / 0.04))`, bounded in [-1, 0].
Reward is `(gamma * Phi_next - Phi_previous) / dt`; division cancels
RewardManager's dt scaling. Gamma matches default PPO (0.99); keep both aligned
if overriding PPO gamma from the command line.

There is no downward-speed gate or positive-only clipping: moving away supplies
positive feedback; moving back costs reward. This is potential shaping, not a
new terminal success objective; actual contact costs still define avoidance.
Activation snapshots establish the baseline before the first dodge action,
without an appearance bonus. Partial reset clears it. True terminals use zero potential; time limits
retain potential for PPO bootstrapping. Terminal compensation completes the
telescoping potential difference, not an independent failure bonus.

Predictions omit future control, articulation and contacts. Reaction time is an
estimate, not a guarantee; no hard bound ensures a small dodge. The separate
contact sensor includes feet, with four slots per body and position/presence
only. Contacts between control reads or beyond slots can be missed. There is no
physical obstacle or continuous collision/safety guarantee.

## Metrics and verification

Rewards log under `Episode_Reward/` with the three names above. Existing
`Metrics/dodge/` fields are requested/active rates, contact rate, contact time
fraction and slot overflow rate. Conditional rates average reset-batch summaries,
not globally episode-weighted data; interpret them alongside active rate.
No extra counts or active-versus-inactive comparisons are added.

Inspect impact costs, head/torso/elbow forces, physics failures and actual motion.
Missing forbidden-force termination metrics reflect its removal, not lower
forces. Low region contact alone does not prove reactive avoidance.

```bash
uv run pytest tests/test_fall_dodge.py tests/test_task_configs.py -q
```

Tests cover pulse timing, limb projection, late placement, observation frames,
partial resets, contact costs, potential progress/retreat and terminal handling,
finite-state handling and Dodge override isolation. CPU tensor tests still
require the project's import dependencies.
