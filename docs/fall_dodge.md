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
disabled. Training now uses only ordinary front-fall and the two front-fall dodge
reference clips. Training resets mix exact standing and forward_walk.npy at 50/50;
CSV resets and adaptive disturbance replay are disabled in this task.

The training pulse is fixed in the reset yaw frame within +/-30 degrees forward.
Its horizontal magnitude is uniform from zero to the current curriculum cap
(30, 80, 120, then 220 N); Z force and pulse duration stages are unchanged.
This retains the stage caps, not the old independent XY magnitude distribution.
The independent world-axis velocity push is removed in training. The direction
is sampled even on episodes without a region and does not track later rotation.
Play reset/push logic and its reference pool are intentionally unchanged.

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
| at_reset | true (task override) | Visible in the first observation after reset |
| forward_half_angle_deg | 30 | Forward push/region half cone |
| region_angle_jitter_deg | 10 | Region offset around push angle, clipped to cone |
| forward_distance_range | (1.00, 2.00) m | Distance from reset base, aimed farther toward torso/head landing area |
| radius_range | (0.08, 0.14) m | Ground disc radius |
| min_root_height | 0.45 m | Reject low/late states |
| initial_body_clearance | 0.18 m | Extra clearance from low collision bounds |
| placement_attempts | 8 | Candidates at reset, then disable on failure |

Placement no longer waits for the pulse, motion frames or a predicted reaction
time. Reset samples the force direction; region candidates are sampled in the
same forward sector. The candidate check happens after reset forward kinematics,
before observations (including manual env.reset). Low collidable geoms are
represented by conservative compiled bounding spheres, including foot extent,
and candidates overlapping their horizontal bounds plus clearance are rejected.
No future trajectory simulation is needed. Low resets are still rejected; actual
active fraction may be below 50%. This only avoids initial overlap, not future
impact. Reset robot states are never moved to accommodate a region. The legacy
post-pulse placement path is retained behind at_reset=false, but is not used by
the Dodge task. Ballistic prediction remains in the unchanged shaping reward.

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
