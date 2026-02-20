#!/usr/bin/env python3
"""
JetRacer IMU Straight-Line Controller

Uses IMU yaw heading + PID to keep the robot driving perfectly straight.
When driving forward/backward, the controller locks the current IMU yaw
heading and continuously corrects angular.z (steering) so the robot
never veers left or right.

Also monitors wheel velocities for diagnostics display.

Controls:
  W / S     - Drive forward / backward  (locks yaw heading)
  SPACE     - Stop  (releases yaw lock)
  + / -     - Adjust throttle speed
  P / p     - Increase / decrease Kp
  I / i     - Increase / decrease Ki
  D / d     - Increase / decrease Kd
  R         - Reset PID state & re-lock current heading
  L         - Log current state to console
  Q         - Quit
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
import math
import sys
import termios
import tty
import threading
import time


# ── defaults ────────────────────────────────────────────────
BASE_OFFSET  = -0.0445   # static steering bias (from calibration)
SPEED_STEP   = 0.05
MIN_SPEED    = 0.10
MAX_SPEED    = 0.60

# PID gains for yaw correction (tune on the robot)
DEFAULT_KP = 0.020       # proportional: degrees of yaw error → steering
DEFAULT_KI = 0.002       # integral: removes persistent drift
DEFAULT_KD = 0.005       # derivative: dampens oscillation

MAX_CORRECTION = 0.40    # clamp so PID can't yank steering too far
MAX_INTEGRAL   = 30.0    # anti-windup limit (in accumulated degree-error)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion orientation to yaw angle in degrees (-180 to 180)."""
    yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw_rad)


def normalize_angle(angle):
    """Normalize angle to [-180, 180] range."""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def read_key(settings):
    """Read a single keypress (non-blocking-ish via raw mode)."""
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        sys.stdin.read(2)                     # consume escape sequence
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return ch


class IMUStraightLineController:
    def __init__(self):
        rospy.init_node("imu_straight_line", anonymous=True)

        # ── publishers ────────────────────────────────────
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ── IMU subscriber ────────────────────────────────
        self.imu_lock  = threading.Lock()
        self.current_yaw = 0.0
        self.imu_ready = False
        rospy.Subscriber("/imu", Imu, self._cb_imu)

        # ── wheel velocity subscribers (for diagnostics) ──
        self.wheel_lock = threading.Lock()
        self.lvel = 0
        self.rvel = 0
        rospy.Subscriber("/motor/lvel", Int32, self._cb_lvel)
        rospy.Subscriber("/motor/rvel", Int32, self._cb_rvel)

        # ── driving state ─────────────────────────────────
        self.speed   = 0.20
        self.driving = 0           # -1 rev, 0 stop, 1 fwd

        # ── yaw lock state ────────────────────────────────
        self.yaw_locked  = False
        self.target_yaw  = 0.0

        # ── PID state ─────────────────────────────────────
        self.kp = DEFAULT_KP
        self.ki = DEFAULT_KI
        self.kd = DEFAULT_KD
        self.integral   = 0.0
        self.prev_error = 0.0
        self.correction = 0.0      # last PID output

    # ── IMU callback ──────────────────────────────────────
    def _cb_imu(self, msg):
        o = msg.orientation
        yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        with self.imu_lock:
            self.current_yaw = yaw
            self.imu_ready = True

    def _get_yaw(self):
        with self.imu_lock:
            return self.current_yaw

    # ── wheel callbacks ───────────────────────────────────
    def _cb_lvel(self, msg):
        with self.wheel_lock:
            self.lvel = msg.data

    def _cb_rvel(self, msg):
        with self.wheel_lock:
            self.rvel = msg.data

    def _get_vel(self):
        with self.wheel_lock:
            return self.lvel, self.rvel

    # ── yaw lock helpers ──────────────────────────────────
    def lock_yaw(self):
        """Lock onto the current IMU heading."""
        self.target_yaw = self._get_yaw()
        self.yaw_locked = True
        self.integral   = 0.0
        self.prev_error = 0.0
        self.correction = 0.0

    def unlock_yaw(self):
        """Release the yaw lock."""
        self.yaw_locked = False
        self.integral   = 0.0
        self.prev_error = 0.0
        self.correction = 0.0

    # ── PID controller ────────────────────────────────────
    def compute_correction(self):
        """
        PID on yaw error.
        error = target_yaw - current_yaw  (positive = robot turned right of target)
        correction is applied to angular.z so robot steers back on heading.
        """
        if not self.yaw_locked:
            self.correction = 0.0
            return 0.0

        current = self._get_yaw()
        error = normalize_angle(self.target_yaw - current)

        # Proportional
        p_out = self.kp * error

        # Integral (with anti-windup)
        self.integral += error
        self.integral = clamp(self.integral, -MAX_INTEGRAL, MAX_INTEGRAL)
        i_out = self.ki * self.integral

        # Derivative
        d_out = self.kd * (error - self.prev_error)
        self.prev_error = error

        # Total PID output (clamped)
        raw = p_out + i_out + d_out
        self.correction = clamp(raw, -MAX_CORRECTION, MAX_CORRECTION)
        return self.correction

    def reset_pid(self):
        """Reset PID accumulator and re-lock current heading."""
        self.integral   = 0.0
        self.prev_error = 0.0
        self.correction = 0.0
        if self.yaw_locked:
            self.target_yaw = self._get_yaw()

    # ── HUD ───────────────────────────────────────────────
    def print_hud(self):
        lv, rv = self._get_vel()
        wheel_diff = lv - rv
        yaw_now    = self._get_yaw()
        direction  = {1: "FWD", -1: "REV"}.get(self.driving, "---")
        state      = "DRIVE" if self.driving != 0 else "STOP "
        steer      = BASE_OFFSET + self.correction

        if self.yaw_locked:
            yaw_error = normalize_angle(self.target_yaw - yaw_now)
            line = (
                f"\r  [{state} {direction}]  spd:{self.speed:.2f}  "
                f"Yaw:{yaw_now:+7.2f}° → {self.target_yaw:+7.2f}° "
                f"err:{yaw_error:+6.2f}°  "
                f"corr:{self.correction:+.4f} steer:{steer:+.4f}  "
                f"Lvel:{lv:+6d} Rvel:{rv:+6d} Δ:{wheel_diff:+5d}  "
                f"Kp:{self.kp:.4f} Ki:{self.ki:.4f} Kd:{self.kd:.4f}"
            )
        else:
            line = (
                f"\r  [{state} {direction}]  spd:{self.speed:.2f}  "
                f"Yaw:{yaw_now:+7.2f}° (free)  "
                f"steer:{steer:+.4f}  "
                f"Lvel:{lv:+6d} Rvel:{rv:+6d} Δ:{wheel_diff:+5d}"
            )
        print(f"{line:<180}", end="", flush=True)

    # ── main loop ─────────────────────────────────────────
    def run(self):
        settings = termios.tcgetattr(sys.stdin)

        print(__doc__)
        print(f"  Base steering offset : {BASE_OFFSET:+.4f}")
        print(f"  PID gains            : Kp={self.kp:.4f}  Ki={self.ki:.4f}  Kd={self.kd:.4f}")
        print(f"  Max correction clamp : ±{MAX_CORRECTION}")
        print(f"  Subscribed to /imu, /motor/lvel, /motor/rvel")
        print(f"  Publishing to /cmd_vel\n")

        # Wait for IMU data before proceeding
        print("  ⏳ Waiting for IMU data...")
        timeout = time.time() + 10
        while not self.imu_ready and time.time() < timeout:
            time.sleep(0.1)

        if self.imu_ready:
            print(f"  ✅ IMU ready!  Current yaw: {self._get_yaw():+.2f}°\n")
        else:
            print("  ⚠️  No IMU data received after 10 s — yaw lock will NOT work!\n")

        time.sleep(0.5)

        twist = Twist()
        rate  = rospy.Rate(20)          # 20 Hz control loop

        try:
            while not rospy.is_shutdown():
                key = read_key(settings)

                if key is None:
                    pass

                # --- driving ---
                elif key in ("w", "W"):
                    self.driving = 1
                    if not self.yaw_locked:
                        self.lock_yaw()
                elif key in ("s", "S"):
                    self.driving = -1
                    if not self.yaw_locked:
                        self.lock_yaw()
                elif key == " ":
                    self.driving = 0
                    self.unlock_yaw()

                # --- speed ---
                elif key in ("+", "="):
                    self.speed = clamp(self.speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
                elif key in ("-", "_"):
                    self.speed = clamp(self.speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)

                # --- tune Kp ---
                elif key == "P":
                    self.kp += 0.002
                    print(f"\n  Kp = {self.kp:.4f}")
                elif key == "p":
                    self.kp = max(0, self.kp - 0.002)
                    print(f"\n  Kp = {self.kp:.4f}")

                # --- tune Ki ---
                elif key == "I":
                    self.ki += 0.001
                    print(f"\n  Ki = {self.ki:.4f}")
                elif key == "i":
                    self.ki = max(0, self.ki - 0.001)
                    print(f"\n  Ki = {self.ki:.4f}")

                # --- tune Kd ---
                elif key == "D":
                    self.kd += 0.001
                    print(f"\n  Kd = {self.kd:.4f}")
                elif key == "d":
                    self.kd = max(0, self.kd - 0.001)
                    print(f"\n  Kd = {self.kd:.4f}")

                # --- reset PID ---
                elif key in ("r", "R"):
                    self.reset_pid()
                    print(f"\n  [i] PID reset — new target yaw: {self.target_yaw:+.2f}°")

                # --- log ---
                elif key in ("l", "L"):
                    lv, rv = self._get_vel()
                    yaw_now = self._get_yaw()
                    err = normalize_angle(self.target_yaw - yaw_now) if self.yaw_locked else 0.0
                    print(
                        f"\n  LOG  Kp={self.kp:.4f} Ki={self.ki:.4f} Kd={self.kd:.4f}  "
                        f"corr={self.correction:+.4f} integral={self.integral:+.2f}  "
                        f"yaw={yaw_now:+.2f}° target={self.target_yaw:+.2f}° err={err:+.2f}°  "
                        f"Lvel={lv} Rvel={rv}"
                    )

                # --- quit ---
                elif key in ("q", "Q", "\x03"):
                    break

                # ── compute correction while moving ───────
                if self.driving != 0:
                    correction = self.compute_correction()
                else:
                    correction = 0.0

                # ── apply: base offset + PID correction ───
                steer = BASE_OFFSET + correction
                if self.driving < 0:
                    steer = -steer           # negate in reverse

                twist.linear.x  = self.speed * self.driving
                twist.angular.z = steer
                self.pub.publish(twist)

                self.print_hud()
                rate.sleep()

        except Exception as e:
            print(f"\nError: {e}")

        finally:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            print(f"\n\n  🛑 Robot stopped.")
            print(f"  Final correction : {self.correction:+.4f}")
            print(f"  Final gains      : Kp={self.kp:.4f}  Ki={self.ki:.4f}  Kd={self.kd:.4f}\n")


if __name__ == "__main__":
    ctrl = IMUStraightLineController()
    ctrl.run()
