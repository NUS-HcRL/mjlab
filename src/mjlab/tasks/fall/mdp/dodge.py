"""Episode-fixed ground regions for the fall task's optional dodge extension.

Regions are virtual: they do not add collision geometry or change the robot.
Only actor/critic observations and a ground-contact cost depend on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def region_contact_mask(
  active: torch.Tensor,
  center_w: torch.Tensor,
  radius: torch.Tensor,
  found: torch.Tensor,
  pos_w: torch.Tensor,
) -> torch.Tensor:
  """Return [env, contact slot] membership using real, valid contact points."""
  distance_sq = (pos_w[..., :2] - center_w[:, None, :2]).square().sum(dim=-1)
  return active[:, None] & (found > 0) & (distance_sq <= radius[:, None].square())


class DodgeRegionCommand(CommandTerm):
  """Activate one predicted landing region shortly after reset, then keep it fixed."""

  cfg: DodgeRegionCommandCfg

  def __init__(self, cfg: DodgeRegionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    if self.robot.data.is_fixed_base:
      raise ValueError("Dodge regions require a floating-base robot")
    self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.requested = torch.zeros_like(self.active)
    self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.radius = torch.zeros(self.num_envs, device=self.device)
    self._steps = torch.zeros(self.num_envs, device=self.device)
    self._contact_steps = torch.zeros_like(self._steps)
    self._overflow_steps = torch.zeros_like(self._steps)
    self.pending = torch.zeros_like(self.active)
    self._motion_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._motion_start_pos_w = torch.zeros_like(self.center_w)

  @property
  def command(self) -> torch.Tensor:
    """Return [active, relative position (3), radius] in the configured frame."""
    root_quat = self.robot.data.root_link_quat_w
    if self.cfg.reference_frame == "yaw":
      root_quat = yaw_quat(root_quat)
    relative = quat_apply_inverse(
      root_quat, self.center_w - self.robot.data.root_link_pos_w
    )
    obs = torch.cat((self.active[:, None], relative, self.radius[:, None]), dim=-1)
    # Mask after transforming: absent regions must be exactly zero even when the
    # robot is far from the world origin. No observation noise is added to this term.
    return torch.where(self.active[:, None], obs, torch.zeros_like(obs))

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    ids = torch.arange(self.num_envs, device=self.device)
    if env_ids is not None:
      ids = ids[env_ids]
    if ids.numel() == 0:
      return {}
    completed = ids[self._steps[ids] > 0]
    extras = {}
    if completed.numel() > 0:
      active = self.active[completed].float()
      active_count = active.sum()
      denominator = active_count.clamp(min=1.0)
      fractions = self._contact_steps[completed] / self._steps[completed]
      extras = {
        "requested_rate": self.requested[completed].float().mean().item(),
        "active_rate": active.mean().item(),
        # Conditional rates are zero when this reset batch has no active episodes.
        # Always interpret them together with active_rate.
        "contact_rate": (
          ((self._contact_steps[completed] > 0).float() * active).sum() / denominator
        ).item(),
        "contact_time_fraction": (
          fractions.mul(active).sum() / denominator
        ).item(),
        "slot_overflow_rate": (
          (self._overflow_steps[completed] / self._steps[completed]).mean()
        ).item(),
      }
    self._steps[ids] = 0
    self._contact_steps[ids] = 0
    self._overflow_steps[ids] = 0
    super().reset(ids)
    return extras

  def compute(self, dt: float) -> None:
    # The force-pulse interval event starts after this method on the reset step.
    # Skip episode_length==0, then observe real motion for a few completed control
    # steps before estimating a landing point and making the region visible.
    pending_ids = torch.nonzero(
      self.pending & (self._env.episode_length_buf > 0), as_tuple=False
    ).squeeze(-1)
    if pending_ids.numel() == 0:
      return
    self._motion_steps[pending_ids] += 1
    ready = pending_ids[
      self._motion_steps[pending_ids] >= self.cfg.motion_observation_steps
    ]
    if ready.numel() > 0:
      self._activate_regions(ready, dt)

  def _update_metrics(self) -> None:
    # Contact statistics are collected by the reward before terminated envs reset.
    pass

  def _update_command(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    n = env_ids.numel()
    self.active[env_ids] = False
    self.center_w[env_ids] = 0
    self.radius[env_ids] = 0
    self.requested[env_ids] = torch.rand(n, device=self.device) < cfg.probability
    data = self.robot.data
    root_pos = data.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[:3]]
    ground_z = self._env.scene.env_origins[env_ids, 2]
    eligible = (root_pos[:, 2] - ground_z) >= cfg.min_root_height
    self.pending[env_ids] = self.requested[env_ids] & eligible
    self._motion_steps[env_ids] = 0
    self._motion_start_pos_w[env_ids] = root_pos

  def _activate_regions(self, env_ids: torch.Tensor, dt: float) -> None:
    """Estimate the future root landing projection from observed post-pulse motion."""
    cfg = self.cfg
    data = self.robot.data
    root_pos = data.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[:3]]
    root_quat = data.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[3:7]]
    current_velocity = data.data.qvel[env_ids][
      :, self.robot.indexing.free_joint_v_adr[:3]
    ]
    elapsed = self._motion_steps[env_ids].to(root_pos.dtype).clamp_min(1) * dt
    observed_velocity = (
      root_pos - self._motion_start_pos_w[env_ids]
    ) / elapsed[:, None]
    velocity = (
      cfg.current_velocity_mix * current_velocity
      + (1.0 - cfg.current_velocity_mix) * observed_velocity
    )
    ground_z = self._env.scene.env_origins[env_ids, 2]

    # Ballistic root projection until it reaches prediction_height above ground.
    # Clamps bound model error from contacts, control action, and non-ballistic motion.
    height = (root_pos[:, 2] - ground_z - cfg.prediction_height).clamp_min(0)
    vz = velocity[:, 2]
    flight_time = (
      vz + torch.sqrt(vz.square() + 2.0 * cfg.gravity_magnitude * height)
    ) / cfg.gravity_magnitude
    flight_time = flight_time.clamp(*cfg.flight_time_range)
    displacement = velocity[:, :2] * flight_time[:, None]

    # When the first few frames have little translation, use root heading as a
    # deterministic fallback so the region is not placed under the standing feet.
    displacement_norm = torch.linalg.vector_norm(
      displacement, dim=-1, keepdim=True
    )
    root_forward = torch.zeros(len(env_ids), 3, device=self.device)
    root_forward[:, 0] = 1.0
    # Applying yaw only keeps this fallback horizontal even when the robot tilts.
    root_forward_w = quat_apply(yaw_quat(root_quat), root_forward)
    direction = torch.where(
      displacement_norm > 1e-5,
      displacement / displacement_norm.clamp_min(1e-6),
      root_forward_w[:, :2],
    )
    base_landing_distance = displacement_norm.squeeze(-1).clamp(
      *cfg.landing_distance_range
    )
    # The root projection tends to put the region beneath the torso. Lead it
    # farther along the fall direction so the target is closer to the expected
    # hand/forearm landing area and can be avoided with a small lateral motion.
    target_distance = base_landing_distance + cfg.landing_lead_distance
    predicted_xy = root_pos[:, :2] + direction * target_distance[:, None]

    n = len(env_ids)
    random = torch.rand(n, cfg.placement_attempts, 3, device=self.device)
    angle = random[..., 0] * (2.0 * math.pi)
    jitter_radius = torch.sqrt(random[..., 1]) * cfg.placement_jitter
    jitter = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    candidates = predicted_xy[:, None, :] + jitter * jitter_radius[..., None]
    radii = cfg.radius_range[0] + random[..., 2] * (
      cfg.radius_range[1] - cfg.radius_range[0]
    )

    body_pos = data.body_link_pos_w[env_ids]
    low_body = (body_pos[..., 2] - ground_z[:, None]) < cfg.low_body_height
    clearance_sq = (candidates[:, :, None] - body_pos[:, None, :, :2]).square().sum(-1)
    intersects = clearance_sq <= (radii[..., None] + cfg.initial_body_clearance).square()
    valid = ~(intersects & low_body[:, None]).any(dim=-1)
    choice = valid.long().argmax(dim=-1)
    rows = torch.arange(n, device=self.device)
    active = valid.any(dim=-1)
    centers = torch.cat((candidates[rows, choice], ground_z[:, None]), dim=-1)
    self.active[env_ids] = active
    self.center_w[env_ids] = torch.where(active[:, None], centers, 0.0)
    self.radius[env_ids] = torch.where(active, radii[rows, choice], 0.0)
    self.pending[env_ids] = False

  def record_contacts(self, hit: torch.Tensor, overflow: torch.Tensor) -> None:
    self._steps += 1
    self._contact_steps += hit.float()
    self._overflow_steps += overflow.float()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    idx = visualizer.env_idx
    if not self.active[idx].item():
      return
    center = self.center_w[idx].detach().cpu().clone()
    center[2] += 0.008  # Draw just above the plane to avoid z fighting.
    radius = self.radius[idx].item()
    # The common native/Viser interface provides arrows; short segments form a
    # ring without adding a geom, mocap body, or viewer-specific implementation.
    angles = torch.linspace(0, 2 * math.pi, 33)
    points = center + radius * torch.stack(
      (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    for i in range(len(points) - 1):
      visualizer.add_arrow(
        points[i],
        points[i + 1],
        color=(1.0, 0.2, 0.1, 0.85),
        width=0.003,
        label=f"dodge_region_{i}",
      )


@dataclass(kw_only=True)
class DodgeRegionCommandCfg(CommandTermCfg):
  class_type: type[CommandTerm] = DodgeRegionCommand
  # Required by CommandTermCfg; compute() disables mid-episode resampling.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True
  asset_name: str = "robot"
  probability: float = 0.2
  reference_frame: Literal["yaw", "full"] = "yaw"
  motion_observation_steps: int = 3
  radius_range: tuple[float, float] = (0.08, 0.14)
  prediction_height: float = 0.35
  gravity_magnitude: float = 9.81
  current_velocity_mix: float = 0.5
  flight_time_range: tuple[float, float] = (0.10, 0.80)
  landing_distance_range: tuple[float, float] = (0.35, 0.90)
  landing_lead_distance: float = 0.35
  placement_jitter: float = 0.12
  min_root_height: float = 0.45
  low_body_height: float = 0.15
  initial_body_clearance: float = 0.18
  placement_attempts: int = 8

  def __post_init__(self) -> None:
    if not 0 <= self.probability <= 1:
      raise ValueError("Dodge probability must be in [0, 1]")
    if self.reference_frame not in ("yaw", "full"):
      raise ValueError("reference_frame must be 'yaw' or 'full'")
    if self.motion_observation_steps < 1:
      raise ValueError("motion_observation_steps must be positive")
    for name in (
      "radius_range", "flight_time_range", "landing_distance_range"
    ):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise ValueError(f"{name} must contain ordered positive finite values")
    for name in (
      "prediction_height",
      "gravity_magnitude",
      "landing_lead_distance",
      "placement_jitter",
      "min_root_height",
      "low_body_height",
      "initial_body_clearance",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if self.gravity_magnitude <= 0:
      raise ValueError("gravity_magnitude must be positive")
    if not 0 <= self.current_velocity_mix <= 1:
      raise ValueError("current_velocity_mix must be in [0, 1]")
    if self.placement_attempts < 1:
      raise ValueError("placement_attempts must be positive")


def dodge_region_observation(
  env: ManagerBasedRlEnv, command_name: str = "dodge"
) -> torch.Tensor:
  return env.command_manager.get_command(command_name)


def dodge_region_contact_cost(
  env: ManagerBasedRlEnv,
  command_name: str = "dodge",
  sensor_name: str = "dodge_ground_contact",
) -> torch.Tensor:
  """Bounded [0, 1] cost per control step; configure a negative reward weight.

  Contacts are read at control frequency, as with existing fall rewards. A contact
  that starts and ends between those reads can be missed; this is not a continuous
  collision/safety guarantee. Slot overflow is reported separately.
  """
  region = cast(DodgeRegionCommand, env.command_manager.get_term(command_name))
  sensor = cast(ContactSensor, env.scene[sensor_name])
  data = sensor.data
  assert data.found is not None and data.pos is not None
  mask = region_contact_mask(
    region.active, region.center_w, region.radius, data.found, data.pos
  )
  hit = mask.any(dim=-1)
  # MuJoCo repeats the pre-reduction match count in each occupied found slot.
  overflow = (data.found > sensor.cfg.num_slots).any(dim=-1)
  region.record_contacts(hit, overflow)
  return hit.float()


class DodgeFirstContactCost:
  """Penalize the first region contact once per episode.

  RewardManager multiplies reward terms by ``dt``. Returning ``1 / step_dt``
  therefore makes the configured weight the fixed penalty for the event instead
  of making it depend on how long contact lasts.
  """

  def __init__(
    self,
    command_name: str = "dodge",
    sensor_name: str = "dodge_ground_contact",
  ) -> None:
    self.command_name = command_name
    self.sensor_name = sensor_name
    self._contacted_once: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._contacted_once is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self._contacted_once[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    region = cast(
      DodgeRegionCommand, env.command_manager.get_term(self.command_name)
    )
    sensor = cast(ContactSensor, env.scene[self.sensor_name])
    data = sensor.data
    assert data.found is not None and data.pos is not None
    if (
      self._contacted_once is None
      or self._contacted_once.shape[0] != env.num_envs
    ):
      self._contacted_once = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )
    hit = region_contact_mask(
      region.active, region.center_w, region.radius, data.found, data.pos
    ).any(dim=-1)
    first_hit = hit & ~self._contacted_once
    self._contacted_once |= hit
    return first_hit.float() / env.step_dt


class DodgePredictedLandingRisk:
  """Predict selected descending links' landing risk around the region."""

  def __init__(
    self,
    body_names: tuple[str, ...],
    command_name: str = "dodge",
    asset_name: str = "robot",
    prediction_height: float = 0.10,
    gravity_magnitude: float = 9.81,
    flight_time_range: tuple[float, float] = (0.05, 0.45),
    body_margin: float = 0.04,
    temperature: float = 0.04,
    min_downward_speed: float = 0.05,
    downward_speed_scale: float = 0.5,
  ) -> None:
    if not body_names:
      raise ValueError("body_names must not be empty")
    lo, hi = flight_time_range
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
      raise ValueError("flight_time_range must contain ordered positive values")
    for name, value in (
      ("prediction_height", prediction_height),
      ("body_margin", body_margin),
      ("min_downward_speed", min_downward_speed),
    ):
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (
      ("gravity_magnitude", gravity_magnitude),
      ("temperature", temperature),
      ("downward_speed_scale", downward_speed_scale),
    ):
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    self.body_names = body_names
    self.command_name = command_name
    self.asset_name = asset_name
    self.prediction_height = prediction_height
    self.gravity_magnitude = gravity_magnitude
    self.flight_time_range = flight_time_range
    self.body_margin = body_margin
    self.temperature = temperature
    self.min_downward_speed = min_downward_speed
    self.downward_speed_scale = downward_speed_scale
    self._body_ids: list[int] | None = None

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    region = cast(
      DodgeRegionCommand, env.command_manager.get_term(self.command_name)
    )
    asset = cast(Entity, env.scene[self.asset_name])
    if self._body_ids is None:
      self._body_ids, matched = asset.find_bodies(
        self.body_names, preserve_order=True
      )
      if len(matched) != len(self.body_names):
        missing = sorted(set(self.body_names) - set(matched))
        raise ValueError(f"Dodge risk bodies not found: {missing}")
    assert self._body_ids is not None

    pos = asset.data.body_link_pos_w[:, self._body_ids]
    vel = asset.data.body_link_lin_vel_w[:, self._body_ids]
    ground_z = env.scene.env_origins[:, 2]
    height = (
      pos[..., 2] - ground_z[:, None] - self.prediction_height
    ).clamp_min(0.0)
    vz = vel[..., 2]
    flight_time = (
      vz + torch.sqrt(vz.square() + 2.0 * self.gravity_magnitude * height)
    ) / self.gravity_magnitude
    flight_time = flight_time.clamp(*self.flight_time_range)
    predicted_xy = pos[..., :2] + vel[..., :2] * flight_time[..., None]

    distance = torch.linalg.vector_norm(
      predicted_xy - region.center_w[:, None, :2], dim=-1
    )
    clearance = distance - region.radius[:, None] - self.body_margin
    spatial_risk = torch.sigmoid(-clearance / self.temperature)
    downward_gate = (
      (-vz - self.min_downward_speed) / self.downward_speed_scale
    ).clamp(0.0, 1.0)
    region_finite = torch.isfinite(region.center_w).all(dim=-1) & torch.isfinite(
      region.radius
    )
    finite = (
      torch.isfinite(pos).all(dim=-1)
      & torch.isfinite(vel).all(dim=-1)
      & region_finite[:, None]
    )
    body_risk = torch.where(finite, spatial_risk * downward_gate, 0.0)
    risk = body_risk.max(dim=-1).values
    return risk * region.active.float()
