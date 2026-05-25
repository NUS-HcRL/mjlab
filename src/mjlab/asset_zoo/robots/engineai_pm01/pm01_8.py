"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "engineai_pm01" / "xmls" / "pm01.xml"
)
assert PM_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PM_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PM_XML))
  # Ensure free joint has a valid name to avoid downstream lookups failing.
  try:
    for j in spec.joints:
      if j.type == mujoco.mjtJoint.mjJNT_FREE and (j.name is None or j.name == ""):
        j.name = "FREE_BASE"
        break
  except Exception:
    pass
  spec.assets = get_assets(spec.meshdir)
  return spec




# Initial pose (matched to pm.py configuration for consistency)
PM_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.82),  # Updated base height from pm.py
  joint_pos={
    #    "J00_HIP_PITCH_L": -0.06,
    # "J01_HIP_ROLL_L": 0.15,
    # "J02_HIP_YAW_L": -0.06,
    # "J03_KNEE_PITCH_L": 0.12,
    # "J04_ANKLE_PITCH_L": -0.06,
    # "J05_ANKLE_ROLL_L": 0.15,
    # "J06_HIP_PITCH_R": -0.06,
    # "J07_HIP_ROLL_R": 0.15,
    # "J08_HIP_YAW_R": -0.06,g
    # "J09_KNEE_PITCH_R": 0.12,
    # "J10_ANKLE_PITCH_R": -0.0,
    # "J11_ANKLE_ROLL_R":  0.15,
    # "J12_WAIST_YAW": 0.0,
    # "J13_SHOULDER_PITCH_L": 0.0,
    # "J14_SHOULDER_ROLL_L": 0.15,
    # "J15_SHOULDER_YAW_L": 0.0,
    # "J16_ELBOW_PITCH_L": -0.25,
    # "J17_ELBOW_YAW_L": 0.0,
    # "J18_SHOULDER_PITCH_R": 0.0,
    # "J19_SHOULDER_ROLL_R": -0.15,
    # "J20_SHOULDER_YAW_R": 0.0,
    # "J21_ELBOW_PITCH_R": -0.25,
    # "J22_ELBOW_YAW_R": 0.0,
    # "J23_HEAD_YAW": 0.0,
    "J00_HIP_PITCH_L": -0.06,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12,
    "J04_ANKLE_PITCH_L": -0.06,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12,
    "J10_ANKLE_PITCH_R": -0.06,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_WAIST_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": -0.25,
    "J22_ELBOW_YAW_R": 0.0,
    "J23_HEAD_YAW": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Physical Parameters (based on pm.py motor specs)
# High-torque joints: Q90 motor (HIP_PITCH, HIP_ROLL, KNEE_PITCH)
ARMATURE_Q90 = 0.0453
EFFORT_LIMIT_Q90 = 164.0
# EFFORT_LIMIT_Q90 = 147.6  # 164.0 * 0.9
# EFFORT_LIMIT_Q90 = 139.4  # 164.0 * 0.85
# EFFORT_LIMIT_Q90 = 131.2  # 164.0 * 0.8
VELOCITY_LIMIT_Q90 = 26.3

# Low-torque joints: Q25 motor (HIP_YAW, ANKLE, WAIST, SHOULDER, ELBOW, HEAD)
ARMATURE_Q25 = 0.0067
EFFORT_LIMIT_Q25 = 52.0
# EFFORT_LIMIT_Q25 = 46.8 # 52.0 * 0.9
# EFFORT_LIMIT_Q25 = 44.2  # 52.0 * 0.85
# EFFORT_LIMIT_Q25 = 41.6  # 52.0 * 0.8
VELOCITY_LIMIT_Q25 = 35.2

# Control parameters: 10Hz natural frequency with critical damping
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# Calculate stiffness and damping based on natural frequency and damping ratio
STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2  # ≈ 178.5
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2  # ≈ 26.4
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ  # ≈ 11.4
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ  # ≈ 1.69

# Actuator groupings: values matched to pm.py configuration
PM_ACTUATOR_HIP_PITCH_KNEE = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    "J00_HIP_PITCH_L",
    "J06_HIP_PITCH_R",
    "J03_KNEE_PITCH_L",
    "J09_KNEE_PITCH_R",
  ),
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
)

PM_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  joint_names_expr=("J01_HIP_ROLL_L", "J07_HIP_ROLL_R"),
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
)

PM_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  joint_names_expr=("J02_HIP_YAW_L", "J08_HIP_YAW_R"),
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
)

PM_ACTUATOR_ANKLES = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
  ),
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
  stiffness=STIFFNESS_Q25,
  damping=0.5,  # Lower damping for ankle compliance (from pm.py)
)

PM_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  joint_names_expr=("J12_WAIST_YAW",),
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
)

PM_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
  ),
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
)

# Head actuator to reach 24 total controls (match motion with head channel)
PM_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  joint_names_expr=("J23_HEAD_YAW",),
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
)

# Collision: all robot collision geoms use prefix `collision_*` (serial_links + base lower in
# serial_pm_v2). `solref` lists every geom explicitly so you can override one link without a
# catch-all (e.g. add SOLREF_CONTACT_HEAD = (...) and set collision_head1 to it).
# solref = (timeconst, dampratio)  larger timeconst => softer
# solimp = (d0, dmax, width)  larger width => thicker compliant layer
SOLIMP_CONTACT_SOFT_6mm = (0.9, 0.95, 0.001)  # 6 mm compliant layer
SOLREF_CONTACT_SOFT_6mm = (0.02, 1.0)
# default solref
SOLIMP_CONTACT_DEFAULT = (0.9, 0.95, 0.001)  # MuJoCo 默认 0.9, 0.95, 0.001
SOLREF_CONTACT_DEFAULT = (0.02, 1.0) # MuJoCo 默认 0.02, 1.0
# Sole + toe contact (not ankle proxy spheres).
SOLIMP_CONTACT_FOOT = (0.9, 0.95, 0.023)
SOLREF_CONTACT_FOOT = (0.0005, 1.0)
# PM_FEET_ONLY_COLLISION = CollisionCfg(
#   geom_names_expr=[
#     r"^collision_left_foot$",
#     r"^collision_left_foot_toe$",
#     r"^collision_right_foot$",
#     r"^collision_right_foot_toe$",
#   ],
#   contype=0,
#   conaffinity=1,
#   condim=3,
#   priority=1,
#   friction=(0.6,),
#   disable_other_geoms=False,
# )

PM_NAMED_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r"^collision_",),
  condim={
    r"^collision_left_foot$": 3,
    r"^collision_left_foot_toe$": 3,
    r"^collision_right_foot$": 3,
    r"^collision_right_foot_toe$": 3,
  },
  priority={
    r"^collision_left_foot$": 1,
    r"^collision_left_foot_toe$": 1,
    r"^collision_right_foot$": 1,
    r"^collision_right_foot_toe$": 1,
  },
  friction={
    r"^collision_left_foot$": (0.6,),
    r"^collision_left_foot_toe$": (0.6,),
    r"^collision_right_foot$": (0.6,),
    r"^collision_right_foot_toe$": (0.6,),
  },
  # solimp={
  #   r"^collision_left_foot$": (0.1, 0.95, 0.00023),
  #   r"^collision_left_foot_toe$": (0.1, 0.95, 0.00023),
  #   r"^collision_right_foot$": (0.1, 0.95, 0.00023),
  #   r"^collision_right_foot_toe$": (0.1, 0.95, 0.00023),
  # },
  # solref={
  #   r"^collision_left_foot$": (0.002, 0.6),
  #   r"^collision_left_foot_toe$":  (0.002, 0.6),
  #   r"^collision_right_foot$":  (0.002, 0.6),
  #   r"^collision_right_foot_toe$":  (0.002, 0.6),
  # },
  solimp={
    # --- feet (sole + toe): wide compliant layer, matches SOLREF_CONTACT_FOOT ---
    r"^collision_left_foot$": SOLIMP_CONTACT_FOOT,
    r"^collision_left_foot_toe$": SOLIMP_CONTACT_FOOT,
    r"^collision_right_foot$": SOLIMP_CONTACT_FOOT,
    r"^collision_right_foot_toe$": SOLIMP_CONTACT_FOOT,
    # --- SOLIMP_CONTACT_SOFT_6mm (hip yaw / knee / shoulder yaw / elbow sphere) ---
    r"^collision_left_hip_yaw$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_right_hip_yaw$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_left_knee_pitch$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_left_knee1$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_right_knee_pitch$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_right_knee1$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_left_elbow_pitch$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_right_elbow_pitch$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_left_shoulder_yaw$": SOLIMP_CONTACT_SOFT_6mm,
    r"^collision_right_shoulder_yaw$": SOLIMP_CONTACT_SOFT_6mm,
    # --- SOLIMP_CONTACT_DEFAULT (everything else) ---
    r"^collision_base_lower$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_head1$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_elbow_capsule$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_elbow_end$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_elbow_yaw$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_hip$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_hip_roll$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_shoulder_roll$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_left_shoulder_roll1$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_elbow_capsule$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_elbow_end$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_elbow_yaw$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_hip$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_hip_roll$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_shoulder_roll$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_right_shoulder_roll1$": SOLIMP_CONTACT_DEFAULT,
    r"^collision_torso_upper$": SOLIMP_CONTACT_DEFAULT,
  },
  # solimp={
  #   r"^collision_left_foot$": (0.95, 0.95, 0.01),
  #   r"^collision_left_foot_toe$": (0.95, 0.95, 0.01),
  #   r"^collision_right_foot$": (0.95, 0.95, 0.01),
  #   r"^collision_right_foot_toe$": (0.95, 0.95, 0.01),
  # },
  # solref={
  #   r"^collision_left_foot$": (0.0005, 0.9),
  #   r"^collision_left_foot_toe$": (0.0005, 0.9),
  #   r"^collision_right_foot$": (0.0005, 0.9),
  #   r"^collision_right_foot_toe$": (0.0005, 0.9),
  # },
  solref={
    # --- SOLREF_CONTACT_SOFT_6mm (hip yaw / knee / shoulder yaw / elbow sphere) ---
    r"^collision_left_hip_yaw$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_right_hip_yaw$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_left_knee_pitch$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_left_knee1$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_right_knee_pitch$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_right_knee1$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_left_elbow_pitch$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_right_elbow_pitch$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_left_shoulder_yaw$": SOLREF_CONTACT_SOFT_6mm,
    r"^collision_right_shoulder_yaw$": SOLREF_CONTACT_SOFT_6mm,
    # --- feet (sole + toe) ---
    r"^collision_left_foot$": SOLREF_CONTACT_FOOT,
    r"^collision_left_foot_toe$": SOLREF_CONTACT_FOOT,
    r"^collision_right_foot$": SOLREF_CONTACT_FOOT,
    r"^collision_right_foot_toe$": SOLREF_CONTACT_FOOT,
    # --- SOLREF_CONTACT_DEFAULT (everything else; edit one line to tune a single link) ---
    r"^collision_base_lower$": SOLREF_CONTACT_DEFAULT,
    r"^collision_head1$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_elbow_capsule$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_elbow_end$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_elbow_yaw$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_hip$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_hip_roll$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_shoulder_roll$": SOLREF_CONTACT_DEFAULT,
    r"^collision_left_shoulder_roll1$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_elbow_capsule$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_elbow_end$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_elbow_yaw$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_hip$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_hip_roll$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_shoulder_roll$": SOLREF_CONTACT_DEFAULT,
    r"^collision_right_shoulder_roll1$": SOLREF_CONTACT_DEFAULT,
    r"^collision_torso_upper$": SOLREF_CONTACT_DEFAULT,
  },
  # solimp="0.9 0.95" solimp="0.0005, 1"
  disable_other_geoms=False,
)

PM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PM_ACTUATOR_HIP_PITCH_KNEE,
    PM_ACTUATOR_HIP_ROLL,
    PM_ACTUATOR_HIP_YAW,
    PM_ACTUATOR_ANKLES,
    PM_ACTUATOR_WAIST_YAW,
    PM_ACTUATOR_ARMS,
    PM_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Indices into `Entity.actuators` for actuator groups that use EFFORT_LIMIT_Q25
# (same order as PM_ARTICULATION.actuators above).
PM_Q25_ACTUATOR_INDICES: tuple[int, ...] = (2, 3, 4, 5, 6)

PM_ROBOT_CFG = EntityCfg(
  init_state=PM_HOME_KEYFRAME,
  collisions=(PM_NAMED_FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=PM_ARTICULATION,
)

# Action scaling similar to @unitree_g1
PM_ACTION_SCALE: dict[str, float] = {}
for a in PM_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  assert e is not None
  for n in names:
    PM_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(PM_ROBOT_CFG)
  viewer.launch(robot.spec.compile())
