#!/usr/bin/env python3
"""
Straight-Line Corrector — IMU PID + Wheel Velocity PI

Ported from jetracer_controlv1.2/scripts/ultimate_straight.py

Fuses IMU heading PID (with gyroscope D-term) and wheel-velocity PI
to keep the robot driving straight. Applied to every velocity command:
  - Full correction when desired action is 'straight'
  - Scaled correction (configurable) when turning

Usage:
    corrector = StraightLineCorrector(config)
    corrector.lock_heading(current_yaw_degrees)
    corrected_angular = corrector.compute_correction(
        desired_angular_z, imu_yaw_deg, gyro_z_rad, left_vel, right_vel
    )
"""

import math
import time
import logging
import threading


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle_deg(a):
    """Wrap angle into [-180, +180] degrees."""
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def quaternion_to_yaw_deg(x, y, z, w):
    """Quaternion -> yaw in degrees (-180 .. +180)."""
    return math.degrees(math.atan2(2.0 * (w * z + x * y),
                                   1.0 - 2.0 * (y * y + z * z)))


class StraightLineCorrector:
    """
    Fuses IMU heading, wheel velocity, and gyroscope into a steering correction.

    Layers:
      1. Calibrated base offset       — static mechanical bias
      2. IMU yaw PID (gyro-enhanced)  — primary heading hold
      3. Wheel-velocity PI (gated)    — secondary, only when heading nearly on-target
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)

        # Load parameters from config (ultimate_straight section)
        ust = config.get('ultimate_straight', {})

        self.base_offset = ust.get('base_offset', -0.0445)

        # IMU PID gains
        self.imu_kp = ust.get('imu_kp', 0.12)
        self.imu_ki = ust.get('imu_ki', 0.008)
        self.imu_kd = ust.get('imu_kd', 0.05)
        self.imu_max_correction = ust.get('imu_max_correction', 0.45)
        self.imu_max_integral = ust.get('imu_max_integral', 10.0)
        self.imu_kd_max = ust.get('imu_kd_max', 0.20)
        self.imu_integral_zone = ust.get('imu_integral_zone', 3.0)
        self.imu_integral_decay = ust.get('imu_integral_decay', 0.9)

        # Yaw spike filter
        self.max_yaw_rate_dps = ust.get('max_yaw_rate_dps', 60.0)

        # Wheel velocity PI
        self.whl_kp = ust.get('whl_kp', 0.0004)
        self.whl_ki = ust.get('whl_ki', 0.00001)
        self.whl_max_correction = ust.get('whl_max_correction', 0.08)
        self.whl_max_integral = ust.get('whl_max_integral', 5000.0)
        self.whl_gate_deg = ust.get('whl_gate_deg', 2.0)

        # Safety
        self.error_stop_deg = ust.get('error_stop_deg', 15.0)

        # Combined output clamp
        self.total_max_steer = ust.get('total_max_steer', 0.50)

        # Turn scaling — reduce correction when robot is intentionally turning
        unified = config.get('unified_controller', {})
        self.turn_scale = unified.get('straight_correction_turn_scale', 0.5)

        # --- internal state ---
        self._lock = threading.Lock()
        self.locked = False
        self.target_yaw = 0.0

        # IMU PID accumulators
        self.imu_integral = 0.0
        self.imu_prev_error = 0.0
        self.yaw_error = 0.0

        # Wheel PI accumulators
        self.whl_integral = 0.0
        self.whl_active = False

        # Spike filter state
        self.prev_valid_yaw = None
        self.last_imu_time = None
        self.spike_count = 0

        # Safety flag
        self.auto_stopped = False

        self.logger.info(
            f"StraightLineCorrector: base_offset={self.base_offset:+.4f} "
            f"IMU(Kp={self.imu_kp} Ki={self.imu_ki} Kd={self.imu_kd}) "
            f"WHL(Kp={self.whl_kp} Ki={self.whl_ki} gate={self.whl_gate_deg}deg)"
        )

    # ------------------------------------------------------------------
    # Heading lock
    # ------------------------------------------------------------------

    def lock_heading(self, current_yaw_deg: float):
        """Lock heading to current yaw value."""
        with self._lock:
            self.target_yaw = current_yaw_deg
            self.locked = True
            self._reset_accumulators()

    def unlock_heading(self):
        """Release heading lock."""
        with self._lock:
            self.locked = False
            self._reset_accumulators()

    def update_target_yaw(self, yaw_deg: float):
        """Update the target yaw without resetting accumulators."""
        with self._lock:
            self.target_yaw = yaw_deg

    def is_locked(self) -> bool:
        with self._lock:
            return self.locked

    def _reset_accumulators(self):
        self.imu_integral = 0.0
        self.imu_prev_error = 0.0
        self.yaw_error = 0.0
        self.whl_integral = 0.0
        self.whl_active = False
        self.auto_stopped = False

    # ------------------------------------------------------------------
    # Spike filter
    # ------------------------------------------------------------------

    def filter_yaw(self, raw_yaw_deg: float) -> float:
        """Apply slew-rate spike filter to reject IMU glitches."""
        now = time.time()
        with self._lock:
            if self.prev_valid_yaw is not None and self.last_imu_time is not None:
                dt = max(now - self.last_imu_time, 0.005)
                max_jump = self.max_yaw_rate_dps * dt
                delta = normalize_angle_deg(raw_yaw_deg - self.prev_valid_yaw)
                if abs(delta) > max_jump:
                    sign = 1.0 if delta > 0 else -1.0
                    raw_yaw_deg = normalize_angle_deg(
                        self.prev_valid_yaw + sign * max_jump)
                    self.spike_count += 1
            self.prev_valid_yaw = raw_yaw_deg
            self.last_imu_time = now
        return raw_yaw_deg

    # ------------------------------------------------------------------
    # Correction layers
    # ------------------------------------------------------------------

    def _compute_imu_pid(self, imu_yaw_deg: float, gyro_z_rad: float) -> float:
        """
        Primary correction: PID on yaw heading error.
          P = Kp * yaw_error
          I = Ki * sum(yaw_error) — zone-gated, with zero-crossing reset
          D = -Kd * gyro_z         — gyro measures rotation rate directly
        """
        if not self.locked:
            return 0.0

        error = normalize_angle_deg(self.target_yaw - imu_yaw_deg)
        self.yaw_error = error

        # Zero-crossing reset
        if self.imu_prev_error != 0.0 and (error * self.imu_prev_error < 0):
            self.imu_integral = 0.0

        # P
        p_term = self.imu_kp * error

        # I — zone-gated with anti-windup
        if abs(error) < self.imu_integral_zone:
            self.imu_integral += error
        else:
            self.imu_integral *= self.imu_integral_decay
        self.imu_integral = _clamp(self.imu_integral,
                                   -self.imu_max_integral, self.imu_max_integral)
        i_term = self.imu_ki * self.imu_integral

        # D — from gyroscope (opposes current rotation)
        d_term = -self.imu_kd * gyro_z_rad
        d_term = _clamp(d_term, -self.imu_kd_max, self.imu_kd_max)

        self.imu_prev_error = error

        raw = p_term + i_term + d_term
        return _clamp(raw, -self.imu_max_correction, self.imu_max_correction)

    def _compute_wheel_pi(self, left_vel: float, right_vel: float) -> float:
        """
        Secondary correction: PI on wheel-velocity difference.
        Gated — only active when |yaw_error| < whl_gate_deg.
        """
        if not self.locked:
            self.whl_active = False
            return 0.0

        if abs(self.yaw_error) > self.whl_gate_deg:
            self.whl_integral *= 0.5   # bleed while gated
            self.whl_active = False
            return 0.0

        self.whl_active = True
        error = left_vel - right_vel     # positive = left faster

        self.whl_integral += error
        self.whl_integral = _clamp(self.whl_integral,
                                   -self.whl_max_integral, self.whl_max_integral)

        raw = -(self.whl_kp * error + self.whl_ki * self.whl_integral)
        return _clamp(raw, -self.whl_max_correction, self.whl_max_correction)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_correction(self,
                           desired_angular_z: float,
                           imu_yaw_deg: float,
                           gyro_z_rad: float,
                           left_vel: float = 0.0,
                           right_vel: float = 0.0,
                           is_turning: bool = False) -> float:
        """
        Compute corrected angular.z given desired angular, IMU data, and wheel data.

        Args:
            desired_angular_z: The angular velocity the high-level controller wants.
            imu_yaw_deg: Current yaw in degrees (spike-filtered recommended).
            gyro_z_rad: IMU gyroscope z-axis in rad/s.
            left_vel: Left wheel velocity (encoder ticks or m/s).
            right_vel: Right wheel velocity (encoder ticks or m/s).
            is_turning: True if the robot is intentionally turning (reduces correction).

        Returns:
            Corrected angular.z value to publish.
        """
        with self._lock:
            if not self.locked:
                return desired_angular_z

            # Safety: auto-stop if error too large
            imu_c = self._compute_imu_pid(imu_yaw_deg, gyro_z_rad)
            if abs(self.yaw_error) > self.error_stop_deg:
                if not self.auto_stopped:
                    self.auto_stopped = True
                    self.logger.warning(
                        f"SAFETY: yaw error {self.yaw_error:+.1f} deg > "
                        f"{self.error_stop_deg} deg — correction only (no stop)"
                    )
                # Still apply correction but don't override desired
                # The main controller decides whether to stop

            whl_c = self._compute_wheel_pi(left_vel, right_vel)

            correction = self.base_offset + imu_c + whl_c

            # Scale correction down when intentionally turning
            if is_turning:
                correction *= self.turn_scale

            total = desired_angular_z + correction
            return _clamp(total, -self.total_max_steer, self.total_max_steer)

    def get_status(self) -> dict:
        """Get current corrector status for display."""
        with self._lock:
            return {
                'locked': self.locked,
                'target_yaw': self.target_yaw,
                'yaw_error': self.yaw_error,
                'whl_active': self.whl_active,
                'spike_count': self.spike_count,
                'auto_stopped': self.auto_stopped,
            }
