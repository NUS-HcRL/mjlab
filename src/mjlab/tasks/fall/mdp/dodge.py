"""Episode-fixed ground regions for the fall task's optional dodge extension.

Regions are virtual: they do not add collision geometry or change the robot.
Only actor/critic observations and a ground-contact cost depend on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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
  """Sample one region after reset events and leave it fixed until the next reset."""

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
    self._peak_force = torch.zeros_like(self._steps)
    self._overflow_steps = torch.zeros_like(self._steps)

  @property
  def command(self) -> torch.Tensor:
    """[active, relative position in the root frame (3), radius], current frame."""
    relative = quat_apply_inverse(
      self.robot.data.root_link_quat_w,
      self.center_w - self.robot.data.root_link_pos_w,
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
        "active_episodes": active_count.item(),
        # Conditional rates are zero when this reset batch has no active episodes.
        # Always interpret them together with active_episodes / active_rate.
        "contact_rate": (
          ((self._contact_steps[completed] > 0).float() * active).sum() / denominator
        ).item(),
        "contact_time_fraction": (fractions.mul(active).sum() / denominator).item(),
        "peak_region_contact_force": (
          self._peak_force[completed].mul(active).sum() / denominator
        ).item(),
        "slot_overflow_rate": (
          (self._overflow_steps[completed] / self._steps[completed]).mean()
        ).item(),
      }
    self._steps[ids] = 0
    self._contact_steps[ids] = 0
    self._peak_force[ids] = 0
    self._overflow_steps[ids] = 0
    super().reset(ids)
    return extras

  def compute(self, dt: float) -> None:
    # Intentionally ignore timers, including in indefinitely long play episodes.
    # Relative observations are computed on demand; world positions never follow
    # the robot and regions never appear/disappear partway through a fall.
    del dt

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

    # Read the free joint directly: reset pushes have already written qvel, but
    # derived cvel-based velocities may still precede that push until forward().
    data = self.robot.data
    root_pos = data.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[:3]]
    root_quat = data.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[3:7]]
    velocity = data.data.qvel[env_ids][:, self.robot.indexing.free_joint_v_adr[:3]]
    ground_z = self._env.scene.env_origins[env_ids, 2]
    eligible = (root_pos[:, 2] - ground_z) >= cfg.min_root_height

    up = torch.zeros(n, 3, device=self.device)
    up[:, 2] = 1
    tilt_xy = quat_apply(root_quat, up)[:, :2]
    heading = velocity[:, :2] * cfg.velocity_lookahead_s + tilt_xy * cfg.tilt_scale
    norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    angle = torch.rand(n, device=self.device) * (2 * math.pi)
    fallback = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    direction = torch.where(norm > 0.05, heading / norm.clamp(min=1e-6), fallback)
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)

    # Try a small fixed batch of candidates, without altering robot/reset states.
    # The heading is only a heuristic, not a prediction of the policy's landing.
    random = torch.rand(n, cfg.placement_attempts, 3, device=self.device)
    distance = cfg.distance_range[0] + random[..., 0] * (
      cfg.distance_range[1] - cfg.distance_range[0]
    )
    offset = (2 * random[..., 1] - 1) * cfg.lateral_range
    radii = cfg.radius_range[0] + random[..., 2] * (
      cfg.radius_range[1] - cfg.radius_range[0]
    )
    candidates = (
      root_pos[:, None, :2]
      + direction[:, None, :] * distance[..., None]
      + lateral[:, None, :] * offset[..., None]
    )
    body_pos = data.body_link_pos_w[env_ids]
    low_body = (body_pos[..., 2] - ground_z[:, None]) < cfg.low_body_height
    clearance_sq = (candidates[:, :, None] - body_pos[:, None, :, :2]).square().sum(-1)
    intersects = clearance_sq <= (radii[..., None] + cfg.initial_body_clearance).square()
    valid = ~(intersects & low_body[:, None]).any(dim=-1)
    choice = valid.long().argmax(dim=-1)
    rows = torch.arange(n, device=self.device)
    active = self.requested[env_ids] & eligible & valid.any(dim=-1)
    centers = torch.cat((candidates[rows, choice], ground_z[:, None]), dim=-1)
    self.active[env_ids] = active
    self.center_w[env_ids] = torch.where(active[:, None], centers, 0.0)
    self.radius[env_ids] = torch.where(active, radii[rows, choice], 0.0)

  def record_contacts(
    self, hit: torch.Tensor, peak_force: torch.Tensor, overflow: torch.Tensor
  ) -> None:
    self._steps += 1
    self._contact_steps += hit.float()
    self._peak_force = torch.maximum(self._peak_force, peak_force)
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
  radius_range: tuple[float, float] = (0.08, 0.14)
  distance_range: tuple[float, float] = (0.35, 0.70)
  lateral_range: float = 0.18
  velocity_lookahead_s: float = 0.35
  tilt_scale: float = 0.5
  min_root_height: float = 0.45
  low_body_height: float = 0.15
  initial_body_clearance: float = 0.18
  placement_attempts: int = 8

  def __post_init__(self) -> None:
    if not 0 <= self.probability <= 1:
      raise ValueError("Dodge probability must be in [0, 1]")
    for name in ("radius_range", "distance_range"):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise ValueError(f"{name} must contain ordered positive finite values")
    for name in (
      "lateral_range",
      "velocity_lookahead_s",
      "tilt_scale",
      "min_root_height",
      "low_body_height",
      "initial_body_clearance",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
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
  assert data.found is not None and data.pos is not None and data.force is not None
  mask = region_contact_mask(
    region.active, region.center_w, region.radius, data.found, data.pos
  )
  hit = mask.any(dim=-1)
  force = torch.linalg.vector_norm(data.force, dim=-1)
  peak_force = torch.where(mask, force, 0.0).max(dim=-1).values
  # MuJoCo repeats the pre-reduction match count in each occupied found slot.
  overflow = (data.found > sensor.cfg.num_slots).any(dim=-1)
  region.record_contacts(hit, peak_force, overflow)
  return hit.float()
