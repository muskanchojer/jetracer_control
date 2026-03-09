#!/usr/bin/env python3
"""
JetRacer — Ultimate Straight-Line Controller  v2
═════════════════════════════════════════════════════
Fuses every available sensor to keep the robot on a perfectly straight path.
Outputs an exact steering angle (angular.z) every cycle — no "left/right"
commands, only precise numeric corrections.

Architecture:
  1. Calibrated base offset       — static mechanical bias
  2. IMU yaw PID (gyro-enhanced)  — primary heading hold
       P  = Kp × yaw_error          (proportional to heading deviation)
       I  = Ki × Σ(yaw_error)       (kills persistent drift)
       D  = −Kd × gyro_z            (gyroscope measures rotation rate directly)
  3. Wheel-velocity PI (gated)    — only active when yaw error is small
  4. Safety: auto-stop if error > threshold (prevents spinning)

The final steering angle published each cycle is:
    steer = base_offset + imu_pid + wheel_pi

Controls:
  W / S       — drive forward / backward  (locks heading automatically)
  SPACE       — stop  (releases heading lock)
  + / −       — adjust throttle speed
  1 / 2       — increase / decrease  IMU Kp
  3 / 4       — increase / decrease  IMU Ki
  5 / 6       — increase / decrease  IMU Kd (gyro D-gain)
  7 / 8       — increase / decrease  Wheel Kp
  9 / 0       — increase / decrease  Wheel Ki
  R           — reset all PID/PI state & re-lock current heading
  L           — log full state snapshot to console
  Q / Ctrl+C  — quit
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
import datetime


# ═══════════════════════════════════════════════════════════
#  TUNABLE DEFAULTS  (adjust to your robot)
# ═══════════════════════════════════════════════════════════

# Static steering bias from calibration
BASE_OFFSET = -0.0445

# Speed limits
SPEED_STEP = 0.05
MIN_SPEED  = 0.10
MAX_SPEED  = 0.60

# ── IMU yaw PID (primary) ─────────────────────────────────
#    P & I operate on yaw error (degrees)
#    D uses raw gyroscope angular_velocity.z (rad/s) for accuracy
IMU_KP = 0.12           # 1° error → 0.12 steer
IMU_KI = 0.008          # accumulated °error → steer
IMU_KD = 0.05           # gyro rad/s → steer (opposes rotation rate)
IMU_MAX_CORRECTION = 0.45
IMU_MAX_INTEGRAL   = 10.0   # anti-windup (reduced to prevent overshoot)
IMU_KD_MAX         = 0.20   # D-term clamp
IMU_INTEGRAL_ZONE  = 3.0    # only accumulate I when |error| < this (deg)
IMU_INTEGRAL_DECAY = 0.9    # shrink integral per tick when error outside zone

# ── Yaw spike filter ─────────────────────────────────────
#    Rate-limit yaw changes to reject IMU glitches / sensor resets
MAX_YAW_RATE_DPS   = 60.0   # max °/s the robot can physically rotate

# ── Wheel-velocity PI (secondary, gated) ──────────────────
#    Only active when |yaw_error| < WHL_GATE_DEG
WHL_KP = 0.0004         # tick-diff → steer
WHL_KI = 0.00001        # accumulated tick-diff → steer
WHL_MAX_CORRECTION = 0.08
WHL_MAX_INTEGRAL   = 5000.0
WHL_GATE_DEG       = 2.0    # disable wheel PI when |yaw error| > this

# ── Safety ────────────────────────────────────────────────
ERROR_WARN_DEG  = 5.0    # print warning when error exceeds this
ERROR_STOP_DEG  = 15.0   # auto-stop robot when error exceeds this

# ── Combined output clamp ─────────────────────────────────
TOTAL_MAX_STEER = 0.50   # absolute max angular.z we ever send

# Control-loop frequency
LOOP_HZ = 30

# ── Log file (inside Docker) ──────────────────────────────
LOG_FILE_PATH = "/tmp/jetracer_run.log"


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quaternion_to_yaw(x, y, z, w):
    """Quaternion → yaw in degrees (−180 … +180)."""
    return math.degrees(math.atan2(2.0 * (w * z + x * y),
                                   1.0 - 2.0 * (y * y + z * z)))


def normalize_angle(a):
    """Wrap angle into [−180, +180]."""
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def read_key(settings):
    """Read one keypress; returns None for unrecognised escape seqs."""
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        sys.stdin.read(2)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return ch


# ═══════════════════════════════════════════════════════════
#  CONTROLLER
# ═══════════════════════════════════════════════════════════

class UltimateStraightLine:
    """Fuses IMU heading, wheel velocity, and gyroscope into one steering angle."""

    def __init__(self):
        rospy.init_node("ultimate_straight_line", anonymous=True)

        # ── publisher ─────────────────────────────────────
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ── IMU state ─────────────────────────────────────
        self.imu_lock    = threading.Lock()
        self.current_yaw = 0.0
        self.gyro_z      = 0.0        # raw angular velocity about z (rad/s)
        self.imu_ready   = False
        rospy.Subscriber("/imu", Imu, self._cb_imu)

        # ── wheel velocity state ──────────────────────────
        self.whl_lock = threading.Lock()
        self.lvel = 0
        self.rvel = 0
        rospy.Subscriber("/motor/lvel", Int32, self._cb_lvel)
        rospy.Subscriber("/motor/rvel", Int32, self._cb_rvel)

        # ── driving ───────────────────────────────────────
        self.speed   = 0.20
        self.driving = 0              # −1 rev | 0 stop | 1 fwd

        # ── heading lock ──────────────────────────────────
        self.locked     = False
        self.target_yaw = 0.0

        # ── IMU PID state ─────────────────────────────────
        self.imu_kp = IMU_KP
        self.imu_ki = IMU_KI
        self.imu_kd = IMU_KD          # this is the gyro-based D gain
        self.imu_integral   = 0.0
        self.imu_prev_error = 0.0     # for zero-crossing detection
        self.imu_correction = 0.0
        self.imu_p = 0.0              # for HUD display
        self.imu_i = 0.0
        self.imu_d = 0.0

        # ── Wheel PI state ────────────────────────────────
        self.whl_kp = WHL_KP
        self.whl_ki = WHL_KI
        self.whl_integral   = 0.0
        self.whl_correction = 0.0
        self.whl_active     = False   # whether wheel PI is engaged

        # ── bookkeeping ───────────────────────────────────
        self.total_steer  = BASE_OFFSET
        self.cycle_count  = 0
        self.yaw_error    = 0.0       # latest error for HUD / safety
        self.auto_stopped = False     # safety-stop flag

        # ── spike filter state ────────────────────────────
        self.prev_valid_yaw = None
        self.last_imu_time  = None
        self.spike_count    = 0

        # ── file logger ──────────────────────────────────
        self.log_file = None

    # ── callbacks ─────────────────────────────────────────
    def _cb_imu(self, msg):
        o = msg.orientation
        raw_yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        gz  = msg.angular_velocity.z          # rad/s about vertical axis
        now = time.time()
        with self.imu_lock:
            # ── spike / slew-rate filter ──────────────────
            if self.prev_valid_yaw is not None and self.last_imu_time is not None:
                dt = max(now - self.last_imu_time, 0.005)
                max_jump = MAX_YAW_RATE_DPS * dt
                delta = normalize_angle(raw_yaw - self.prev_valid_yaw)
                if abs(delta) > max_jump:
                    sign = 1.0 if delta > 0 else -1.0
                    raw_yaw = normalize_angle(
                        self.prev_valid_yaw + sign * max_jump)
                    self.spike_count += 1
            self.current_yaw    = raw_yaw
            self.prev_valid_yaw = raw_yaw
            self.gyro_z         = gz
            self.last_imu_time  = now
            self.imu_ready      = True

    def _cb_lvel(self, msg):
        with self.whl_lock:
            self.lvel = msg.data

    def _cb_rvel(self, msg):
        with self.whl_lock:
            self.rvel = msg.data

    # ── thread-safe readers ───────────────────────────────
    def _get_yaw(self):
        with self.imu_lock:
            return self.current_yaw

    def _get_gyro_z(self):
        with self.imu_lock:
            return self.gyro_z

    def _get_vel(self):
        with self.whl_lock:
            return self.lvel, self.rvel

    # ── heading lock helpers ──────────────────────────────
    def lock_heading(self):
        self.target_yaw = self._get_yaw()
        self.locked = True
        self._reset_all()

    def unlock_heading(self):
        self.locked = False
        self._reset_all()

    def _reset_all(self):
        """Zero every accumulator."""
        self.imu_integral   = 0.0
        self.imu_prev_error = 0.0
        self.imu_correction = 0.0
        self.imu_p = 0.0
        self.imu_i = 0.0
        self.imu_d = 0.0
        self.whl_integral   = 0.0
        self.whl_correction = 0.0
        self.whl_active     = False
        self.yaw_error      = 0.0
        self.auto_stopped   = False

    # ═════════════════════════════════════════════════════
    #  CORRECTION LAYERS
    # ═════════════════════════════════════════════════════

    def _compute_imu_pid(self):
        """
        Primary correction: PID on yaw heading error.
          P = Kp × yaw_error          (degrees — proportional to deviation)
          I = Ki × Σ(yaw_error)       (degrees — removes persistent offset)
          D = −Kd × gyro_z            (rad/s — gyroscope measures rate directly,
                                       far more accurate than differencing yaw)

        Positive error = robot turned CW past target → positive correction (steer left).
        Gyro D-term: opposes current rotation rate to dampen oscillation.
        """
        if not self.locked or not self.imu_ready:
            self.imu_correction = 0.0
            return 0.0

        current = self._get_yaw()
        error = normalize_angle(self.target_yaw - current)
        self.yaw_error = error

        # ── Zero-crossing reset: FULL reset when error crosses zero ──
        #    Old integral was accumulated for the wrong direction — dump it
        if self.imu_prev_error != 0.0 and (error * self.imu_prev_error < 0):
            self.imu_integral = 0.0

        # P — proportional to heading deviation
        self.imu_p = self.imu_kp * error

        # I — zone-gated integral with anti-windup
        #    Only accumulate when error is small (prevents windup during spikes)
        #    Actively decay stored integral when error is large
        if abs(error) < IMU_INTEGRAL_ZONE:
            self.imu_integral += error
        else:
            self.imu_integral *= IMU_INTEGRAL_DECAY
        self.imu_integral = clamp(self.imu_integral, -IMU_MAX_INTEGRAL, IMU_MAX_INTEGRAL)
        self.imu_i = self.imu_ki * self.imu_integral

        # D — from gyroscope (NOT differenced yaw)
        #   gyro_z > 0 → rotating CCW → oppose with negative steering
        #   gyro_z < 0 → rotating CW  → oppose with positive steering
        gyro_z = self._get_gyro_z()
        self.imu_d = -self.imu_kd * gyro_z
        self.imu_d = clamp(self.imu_d, -IMU_KD_MAX, IMU_KD_MAX)

        self.imu_prev_error = error

        raw = self.imu_p + self.imu_i + self.imu_d
        self.imu_correction = clamp(raw, -IMU_MAX_CORRECTION, IMU_MAX_CORRECTION)
        return self.imu_correction

    def _compute_wheel_pi(self):
        """
        Secondary correction: PI on wheel-velocity difference (encoder ticks).
        GATED: only active when |yaw_error| < WHL_GATE_DEG.

        When the IMU PID is actively steering (large yaw error), wheel speeds
        WILL be asymmetric on purpose. The wheel PI must NOT fight that.
        It only engages when the heading is nearly on-target to handle
        subtle mechanical drift the IMU can't see.
        """
        if not self.locked:
            self.whl_correction = 0.0
            self.whl_active = False
            return 0.0

        # Gate: disable when yaw error is large
        if abs(self.yaw_error) > WHL_GATE_DEG:
            self.whl_integral *= 0.5   # bleed off integral while gated
            self.whl_correction = 0.0
            self.whl_active = False
            return 0.0

        self.whl_active = True
        lv, rv = self._get_vel()
        error = lv - rv                       # positive = left faster

        # I (anti-windup)
        self.whl_integral += error
        self.whl_integral = clamp(self.whl_integral, -WHL_MAX_INTEGRAL, WHL_MAX_INTEGRAL)

        raw = -(self.whl_kp * error + self.whl_ki * self.whl_integral)
        self.whl_correction = clamp(raw, -WHL_MAX_CORRECTION, WHL_MAX_CORRECTION)
        return self.whl_correction

    # ═════════════════════════════════════════════════════
    #  FUSED OUTPUT
    # ═════════════════════════════════════════════════════

    def compute_steering(self):
        """
        Combine all layers into one exact steering angle:
            steer = base_offset + imu_pid + wheel_pi
        Then clamp to TOTAL_MAX_STEER.
        """
        imu_c  = self._compute_imu_pid()
        whl_c  = self._compute_wheel_pi()

        raw = BASE_OFFSET + imu_c + whl_c
        self.total_steer = clamp(raw, -TOTAL_MAX_STEER, TOTAL_MAX_STEER)
        return self.total_steer

    def check_safety(self):
        """
        Auto-stop if yaw error exceeds safety threshold.
        Returns True if safe to continue, False if auto-stopped.
        """
        if not self.locked:
            return True

        abs_err = abs(self.yaw_error)

        if abs_err > ERROR_STOP_DEG:
            # Critical: robot has deviated too far, stop immediately
            if not self.auto_stopped:
                self.auto_stopped = True
                self.driving = 0
                self.unlock_heading()
                stop_msg = (f"SAFETY STOP — yaw error {self.yaw_error:+.1f}° exceeded "
                            f"±{ERROR_STOP_DEG}°!")
                print(f"\n\n  🚨 {stop_msg}")
                print(f"  Robot is off course. Press W to re-lock heading and try again.\n")
                self._log(stop_msg)
            return False

        if abs_err > ERROR_WARN_DEG and self.cycle_count % 10 == 0:
            warn_msg = f"Large yaw error: {self.yaw_error:+.1f}° — correction active"
            print(f"\n  ⚠️  {warn_msg}")
            self._log(f"WARNING: {warn_msg}")

        return True

    # ═════════════════════════════════════════════════
    #  FILE LOGGING
    # ═════════════════════════════════════════════════

    def _open_log(self):
        """Open the run log file."""
        try:
            self.log_file = open(LOG_FILE_PATH, "w")
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log(f"=== JetRacer Ultimate Straight-Line Controller v2 ===")
            self._log(f"Started: {ts}")
            self._log(f"Base offset: {BASE_OFFSET:+.4f}")
            self._log(f"IMU PID: Kp={self.imu_kp:.4f} Ki={self.imu_ki:.4f} Kd={self.imu_kd:.4f}")
            self._log(f"IMU limits: max_corr={IMU_MAX_CORRECTION} max_int={IMU_MAX_INTEGRAL} "
                      f"int_zone={IMU_INTEGRAL_ZONE}° Kd_max={IMU_KD_MAX}")
            self._log(f"Wheel PI: Kp={self.whl_kp:.5f} Ki={self.whl_ki:.6f} gate={WHL_GATE_DEG}°")
            self._log(f"Safety: warn={ERROR_WARN_DEG}° stop={ERROR_STOP_DEG}°")
            self._log(f"Spike filter: max_rate={MAX_YAW_RATE_DPS}°/s")
            self._log(f"Loop rate: {LOOP_HZ} Hz")
            self._log("")
            print(f"  📝 Log file: {LOG_FILE_PATH}")
        except Exception as e:
            print(f"\n  ⚠️  Could not open log file: {e}")
            self.log_file = None

    def _log(self, msg):
        """Write a timestamped line to the log file."""
        if self.log_file:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_file.write(f"[{ts}] {msg}\n")

    def _close_log(self):
        """Flush and close the log file."""
        if self.log_file:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log(f"")
            self._log(f"Stopped: {ts}  Cycles: {self.cycle_count}  Spikes filtered: {self.spike_count}")
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    # ═════════════════════════════════════════════════════
    #  HUD
    # ═════════════════════════════════════════════════════

    def print_hud(self):
        lv, rv = self._get_vel()
        yaw    = self._get_yaw()
        gz     = self._get_gyro_z()
        direction = {1: "FWD", -1: "REV"}.get(self.driving, "---")
        state     = "DRIVE" if self.driving else "STOP "

        if self.locked:
            err = normalize_angle(self.target_yaw - yaw)
            whl_tag = "ON " if self.whl_active else "off"
            line = (
                f"\r  [{state} {direction}] spd:{self.speed:.2f} │ "
                f"Yaw:{yaw:+7.2f}°→{self.target_yaw:+7.2f}° err:{err:+6.2f}° │ "
                f"P:{self.imu_p:+.4f} I:{self.imu_i:+.4f} D:{self.imu_d:+.4f} "
                f"={self.imu_correction:+.4f} │ "
                f"WHL[{whl_tag}]:{self.whl_correction:+.4f} │ "
                f"steer:{self.total_steer:+.4f} │ "
                f"L:{lv:+5d} R:{rv:+5d} Δ:{lv - rv:+5d} gz:{gz:+.3f}"
            )
        else:
            line = (
                f"\r  [{state} {direction}] spd:{self.speed:.2f} │ "
                f"Yaw:{yaw:+7.2f}° (free) │ "
                f"steer:{self.total_steer:+.4f} │ "
                f"L:{lv:+5d} R:{rv:+5d} Δ:{lv - rv:+5d}"
            )
        print(f"{line:<200}", end="", flush=True)
        self._log(line.strip())
    # ═════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═════════════════════════════════════════════════════

    def run(self):
        settings = termios.tcgetattr(sys.stdin)

        print("""
╔═══════════════════════════════════════════════════════════════╗
║      JetRacer — Ultimate Straight-Line Controller  v2        ║
╠═══════════════════════════════════════════════════════════════╣
║  W / S   — forward / backward  (auto-locks heading)         ║
║  SPACE   — stop  (releases lock)                             ║
║  + / −   — adjust speed                                      ║
║                                                               ║
║  Gain tuning:                                                 ║
║   1/2 — IMU Kp ↑↓    3/4 — IMU Ki ↑↓    5/6 — Gyro Kd ↑↓   ║
║   7/8 — Whl Kp ↑↓    9/0 — Whl Ki ↑↓                        ║
║                                                               ║
║  R — reset PID & re-lock heading                              ║
║  L — log full state snapshot                                  ║
║  Q — quit                                                     ║
║                                                               ║
║  Safety: auto-stops if yaw error > ±%.0f°                    ║
╚═══════════════════════════════════════════════════════════════╝
""" % ERROR_STOP_DEG)
        print(f"  Base offset      : {BASE_OFFSET:+.4f}")
        print(f"  IMU  PID gains   : Kp={self.imu_kp:.4f}  Ki={self.imu_ki:.4f}  Kd(gyro)={self.imu_kd:.4f}")
        print(f"  IMU  limits      : max_corr={IMU_MAX_CORRECTION}  max_integral={IMU_MAX_INTEGRAL}  Kd_max={IMU_KD_MAX}")
        print(f"  Wheel PI gains   : Kp={self.whl_kp:.5f}  Ki={self.whl_ki:.6f}  gate={WHL_GATE_DEG}°")
        print(f"  Safety           : warn={ERROR_WARN_DEG}°  stop={ERROR_STOP_DEG}°")
        print(f"  Total steer clamp: ±{TOTAL_MAX_STEER}")
        print(f"  Loop rate        : {LOOP_HZ} Hz")
        print(f"  Subscribed to    : /imu  /motor/lvel  /motor/rvel")
        print(f"  Publishing to    : /cmd_vel\n")

        # ── wait for IMU ──────────────────────────────────
        print("  ⏳ Waiting for IMU data…")
        t0 = time.time()
        while not self.imu_ready and (time.time() - t0) < 10:
            time.sleep(0.1)

        if self.imu_ready:
            print(f"  ✅ IMU online — yaw: {self._get_yaw():+.2f}°\n")
        else:
            print("  ⚠️  No IMU data after 10 s — heading correction disabled!\n")

        time.sleep(0.3)

        self._open_log()

        twist = Twist()
        rate  = rospy.Rate(LOOP_HZ)

        try:
            while not rospy.is_shutdown():
                key = read_key(settings)
                self.cycle_count += 1

                # ── key handling ──────────────────────────
                if key is None:
                    pass

                # driving
                elif key in ("w", "W"):
                    self.driving = 1
                    self.auto_stopped = False
                    if not self.locked:
                        self.lock_heading()
                elif key in ("s", "S"):
                    self.driving = -1
                    self.auto_stopped = False
                    if not self.locked:
                        self.lock_heading()
                elif key == " ":
                    self.driving = 0
                    self.unlock_heading()

                # speed
                elif key in ("+", "="):
                    self.speed = clamp(self.speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
                elif key in ("-", "_"):
                    self.speed = clamp(self.speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)

                # ── gain tuning (exact numeric keys) ──────
                elif key == "1":
                    self.imu_kp += 0.01
                    print(f"\n  IMU Kp ↑ = {self.imu_kp:.4f}")
                elif key == "2":
                    self.imu_kp = max(0, self.imu_kp - 0.01)
                    print(f"\n  IMU Kp ↓ = {self.imu_kp:.4f}")
                elif key == "3":
                    self.imu_ki += 0.002
                    print(f"\n  IMU Ki ↑ = {self.imu_ki:.4f}")
                elif key == "4":
                    self.imu_ki = max(0, self.imu_ki - 0.002)
                    print(f"\n  IMU Ki ↓ = {self.imu_ki:.4f}")
                elif key == "5":
                    self.imu_kd += 0.005
                    print(f"\n  Gyro Kd ↑ = {self.imu_kd:.4f}")
                elif key == "6":
                    self.imu_kd = max(0, self.imu_kd - 0.005)
                    print(f"\n  Gyro Kd ↓ = {self.imu_kd:.4f}")
                elif key == "7":
                    self.whl_kp += 0.0001
                    print(f"\n  Whl Kp ↑ = {self.whl_kp:.5f}")
                elif key == "8":
                    self.whl_kp = max(0, self.whl_kp - 0.0001)
                    print(f"\n  Whl Kp ↓ = {self.whl_kp:.5f}")
                elif key == "9":
                    self.whl_ki += 0.000005
                    print(f"\n  Whl Ki ↑ = {self.whl_ki:.6f}")
                elif key == "0":
                    self.whl_ki = max(0, self.whl_ki - 0.000005)
                    print(f"\n  Whl Ki ↓ = {self.whl_ki:.6f}")

                # reset
                elif key in ("r", "R"):
                    self._reset_all()
                    if self.locked:
                        self.target_yaw = self._get_yaw()
                    print(f"\n  ↻ PID/PI reset — target yaw: {self.target_yaw:+.2f}°")

                # log
                elif key in ("l", "L"):
                    self._print_log()

                # quit
                elif key in ("q", "Q", "\x03"):
                    break

                # ── log key events ───────────────────────────
                if key and self.log_file:
                    if key in ('w', 'W', 's', 'S', ' '):
                        labels = {'w': 'FWD', 'W': 'FWD', 's': 'REV', 'S': 'REV', ' ': 'STOP'}
                        self._log(f"KEY: {labels.get(key, key)} driving={self.driving} "
                                  f"locked={self.locked} target={self.target_yaw:+.2f}°")
                    elif key in ('r', 'R'):
                        self._log(f"KEY: RESET target={self.target_yaw:+.2f}°")
                    elif key in '1234567890':
                        self._log(f"KEY: GAIN IMU(Kp={self.imu_kp:.4f} Ki={self.imu_ki:.4f} "
                                  f"Kd={self.imu_kd:.4f}) WHL(Kp={self.whl_kp:.5f} Ki={self.whl_ki:.6f})")

                # ── compute fused steering ────────────────
                if self.driving != 0:
                    steer = self.compute_steering()

                    # Safety check (may auto-stop)
                    if not self.check_safety():
                        steer = BASE_OFFSET
                    elif self.driving < 0:
                        steer = -steer       # mirror for reverse
                else:
                    steer = BASE_OFFSET      # idle: just the static bias

                twist.linear.x  = self.speed * self.driving
                twist.angular.z = steer
                self.pub.publish(twist)

                self.print_hud()
                rate.sleep()

        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        finally:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            self._close_log()
            self._print_summary()

    # ═════════════════════════════════════════════════════
    #  LOGGING
    # ═════════════════════════════════════════════════════

    def _print_log(self):
        lv, rv = self._get_vel()
        yaw = self._get_yaw()
        gz  = self._get_gyro_z()
        err = normalize_angle(self.target_yaw - yaw) if self.locked else 0.0
        print(
            f"\n  ───────── LOG (cycle {self.cycle_count}) ─────────\n"
            f"  Heading  : yaw={yaw:+.2f}°  target={self.target_yaw:+.2f}°  err={err:+.2f}°\n"
            f"  Gyro z   : {gz:+.4f} rad/s\n"
            f"  Wheels   : L={lv}  R={rv}  Δ={lv - rv}  (PI {'ON' if self.whl_active else 'gated'})\n"
            f"  IMU PID  : Kp={self.imu_kp:.4f} Ki={self.imu_ki:.4f} Kd(gyro)={self.imu_kd:.4f}\n"
            f"             P={self.imu_p:+.4f}  I={self.imu_i:+.4f}  D={self.imu_d:+.4f}  "
            f"total={self.imu_correction:+.4f}  integral={self.imu_integral:+.2f}\n"
            f"  Wheel PI : Kp={self.whl_kp:.5f} Ki={self.whl_ki:.6f}  "
            f"integral={self.whl_integral:+.1f}  corr={self.whl_correction:+.4f}\n"
            f"  Output   : base={BASE_OFFSET:+.4f}  total_steer={self.total_steer:+.4f}\n"
            f"  ─────────────────────────────────────────"
        )

    def _print_summary(self):
        print(
            f"\n\n  🛑 Robot stopped.\n"
            f"  Cycles run       : {self.cycle_count}\n"
            f"  Final steer      : {self.total_steer:+.4f}\n"
            f"  IMU PID gains    : Kp={self.imu_kp:.4f}  Ki={self.imu_ki:.4f}  Kd(gyro)={self.imu_kd:.4f}\n"
            f"  Wheel PI gains   : Kp={self.whl_kp:.5f}  Ki={self.whl_ki:.6f}\n"
        )


if __name__ == "__main__":
    ctrl = UltimateStraightLine()
    ctrl.run()
