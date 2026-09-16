"""CPU-only motion clock tests; no simulator or torch installation required."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_PATH = Path(__file__).resolve().parents[1] / "src/mjlab/utils/motion_resampling.py"
_SPEC = importlib.util.spec_from_file_location("motion_resampling", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
resample_motion_arrays = _MODULE.resample_motion_arrays


class MotionResamplingTests(unittest.TestCase):
  def make_motion(self, fps=30.0, count=31):
    times = np.arange(count, dtype=np.float64) / fps
    yaw = times * np.pi / 2.0
    quat = np.stack(
      (np.cos(yaw / 2), times * 0, times * 0, np.sin(yaw / 2)), axis=-1
    )
    return {
      "fps": np.array([fps]),
      "joint_pos": (2.0 * times)[:, None],
      "joint_vel": np.full((count, 1), 2.0),
      "body_pos_w": np.repeat(times[:, None, None], 3, axis=2),
      "body_quat_w": quat[:, None, :],
      "body_lin_vel_w": np.ones((count, 1, 3)),
      "body_ang_vel_w": np.tile([0.0, 0.0, np.pi / 2], (count, 1, 1)),
    }

  def test_physical_time_and_velocity_are_preserved(self):
    data = self.make_motion()
    result = resample_motion_arrays(data, 0.02)
    times = np.arange(51) * 0.02
    np.testing.assert_allclose(result["joint_pos"][:, 0], 2.0 * times)
    np.testing.assert_allclose(result["joint_vel"], 2.0)
    np.testing.assert_allclose(result["body_pos_w"][:, 0, 0], times)
    np.testing.assert_allclose(result["body_lin_vel_w"], 1.0)
    np.testing.assert_allclose(result["body_ang_vel_w"][:, 0, 2], np.pi / 2)
    yaw = 2 * np.arctan2(
      result["body_quat_w"][:, 0, 3], result["body_quat_w"][:, 0, 0]
    )
    np.testing.assert_allclose(yaw, times * np.pi / 2, atol=1e-6)
    self.assertEqual(len(data["joint_pos"]), 31)

  def test_quaternion_sign_flips_take_short_arc(self):
    data = self.make_motion(fps=1.0, count=2)
    data["body_quat_w"][1] *= -1
    result = resample_motion_arrays(data, 0.5)
    np.testing.assert_allclose(
      result["body_quat_w"][1, 0],
      [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-7,
    )
    np.testing.assert_allclose(np.linalg.norm(result["body_quat_w"], axis=-1), 1)

  def test_matching_clock_is_unchanged(self):
    data = self.make_motion(fps=50.0)
    self.assertIs(resample_motion_arrays(data, 0.02), data)

  def test_fractional_tail_is_not_stretched_or_extrapolated(self):
    result = resample_motion_arrays(self.make_motion(count=2), 0.02)
    np.testing.assert_allclose(result["joint_pos"][:, 0], [0.0, 0.04])

  def test_batched_root_layout_and_downsampling(self):
    data = self.make_motion(fps=100.0, count=101)
    data["joint_pos"] = data["joint_pos"][None]
    data["joint_vel"] = data["joint_vel"][None]
    data["root_pos"] = data["body_pos_w"][:, 0][None]
    data["root_quat"] = data["body_quat_w"][:, 0][None]
    result = resample_motion_arrays(data, 0.02)
    self.assertEqual(result["joint_pos"].shape, (1, 51, 1))
    self.assertEqual(result["root_quat"].shape, (1, 51, 4))
    np.testing.assert_allclose(result["root_pos"][0, :, 0], np.arange(51) * 0.02)

  def test_single_frame_is_supported(self):
    result = resample_motion_arrays(self.make_motion(count=1), 0.02)
    self.assertEqual(result["joint_pos"].shape, (1, 1))
    self.assertEqual(float(result["fps"]), 50.0)

  def test_invalid_clock_fails_instead_of_silently_changing_speed(self):
    for fps in (0.0, -30.0, float("nan"), float("inf")):
      data = self.make_motion()
      data["fps"] = np.array(fps)
      with self.subTest(fps=fps), self.assertRaises(ValueError):
        resample_motion_arrays(data, 0.02)
    data = self.make_motion()
    del data["fps"]
    with self.assertRaises(ValueError):
      resample_motion_arrays(data, 0.02)


if __name__ == "__main__":
  unittest.main()
