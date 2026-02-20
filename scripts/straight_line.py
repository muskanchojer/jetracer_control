#!/usr/bin/env python3
"""
JetRacer Straight-Line Auto-Correction

Reads left/right wheel velocities in real time and applies a PI controller
to angular.z so both wheels spin at the same speed → robot drives straight.

Controls:
  W / S     - Drive forward / backward
  SPACE     - Stop
  + / -     - Adjust throttle speed
  P / p     - Increase / decrease Kp
  I / i     - Increase / decrease Ki
  R         - Reset PI state
  L         - Log current gains & correction to console
  Q         - Quit
"""

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import sys
import termios
import tty
import threading
import time


# ── defaults ────────────────────────────────────────────────
BASE_OFFSET  = -0.0445   # static steering bias you already found
SPEED_STEP   = 0.05
MIN_SPEED    = 0.10
MAX_SPEED    = 0.60

# PI gains (tune on the robot — start small)
DEFAULT_KP = 0.0005     # proportional: per-tick correction per unit of vel diff
DEFAULT_KI = 0.00002    # integral: removes persistent drift over time

MAX_CORRECTION = 0.15   # clamp so PI can't yank steering too far
MAX_INTEGRAL   = 5000   # anti-windup limit (in accumulated tick-error units)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_key(settings):
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        sys.stdin.read(2)                     # consume escape sequence
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return ch


class StraightLineController:
    def __init__(self):
        rospy.init_node("straight_line_autocorrect", anonymous=True)

        # publisher
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # wheel velocity (Int32)
        self.lock = threading.Lock()
        self.lvel = 0
        self.rvel = 0

        rospy.Subscriber("/motor/lvel", Int32, self._cb_lvel)
        rospy.Subscriber("/motor/rvel", Int32, self._cb_rvel)

        # driving state
        self.speed   = 0.20
        self.driving = 0           # -1 rev, 0 stop, 1 fwd

        # PI state
        self.kp = DEFAULT_KP
        self.ki = DEFAULT_KI
        self.integral = 0.0
        self.correction = 0.0      # last PI output

    # ── callbacks ──────────────────────────────────────────
    def _cb_lvel(self, msg):
        with self.lock:
            self.lvel = msg.data

    def _cb_rvel(self, msg):
        with self.lock:
            self.rvel = msg.data

    def _get_vel(self):
        with self.lock:
            return self.lvel, self.rvel

    # ── PI controller ──────────────────────────────────────
    def compute_correction(self):
        """
        error = lvel - rvel
          > 0 → left wheel faster → robot veers right → need negative correction
          < 0 → right wheel faster → robot veers left → need positive correction
        """
        lv, rv = self._get_vel()
        error = lv - rv                       # positive = left faster

        # Integral (with anti-windup)
        self.integral += error
        self.integral = clamp(self.integral, -MAX_INTEGRAL, MAX_INTEGRAL)

        # PI output
        raw = -(self.kp * error + self.ki * self.integral)
        self.correction = clamp(raw, -MAX_CORRECTION, MAX_CORRECTION)
        return self.correction

    def reset_pi(self):
        self.integral   = 0.0
        self.correction = 0.0

    # ── display ────────────────────────────────────────────
    def print_hud(self):
        lv, rv = self._get_vel()
        diff = lv - rv
        direction = {1: "FWD", -1: "REV"}.get(self.driving, "---")
        state = "DRIVE" if self.driving != 0 else "STOP "
        steer = BASE_OFFSET + self.correction

        line = (
            f"\r  [{state} {direction}]  spd:{self.speed:.2f}  "
            f"Lvel:{lv:+6d} Rvel:{rv:+6d} diff:{diff:+6d}  |  "
            f"base:{BASE_OFFSET:+.4f} corr:{self.correction:+.5f} "
            f"steer:{steer:+.5f}  "
            f"Kp:{self.kp:.5f} Ki:{self.ki:.6f}"
        )
        print(f"{line:<160}", end="", flush=True)

    # ── main loop ──────────────────────────────────────────
    def run(self):
        settings = termios.tcgetattr(sys.stdin)

        print(__doc__)
        print(f"  Base steering offset : {BASE_OFFSET:+.4f}")
        print(f"  PI gains             : Kp={self.kp:.5f}  Ki={self.ki:.6f}")
        print(f"  Max correction clamp : ±{MAX_CORRECTION}")
        print(f"  Subscribed to /motor/lvel, /motor/rvel")
        print(f"  Publishing to /cmd_vel\n")

        time.sleep(1.0)

        twist = Twist()
        rate  = rospy.Rate(20)

        try:
            while not rospy.is_shutdown():
                key = read_key(settings)

                if key is None:
                    pass

                # --- driving ---
                elif key in ("w", "W"):
                    self.driving = 1
                    self.reset_pi()        # fresh start each time
                elif key in ("s", "S"):
                    self.driving = -1
                    self.reset_pi()
                elif key == " ":
                    self.driving = 0
                    self.reset_pi()

                # --- speed ---
                elif key in ("+", "="):
                    self.speed = clamp(self.speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
                elif key in ("-", "_"):
                    self.speed = clamp(self.speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)

                # --- tune Kp ---
                elif key == "P":
                    self.kp += 0.0001
                    print(f"\n  Kp = {self.kp:.5f}")
                elif key == "p":
                    self.kp = max(0, self.kp - 0.0001)
                    print(f"\n  Kp = {self.kp:.5f}")

                # --- tune Ki ---
                elif key == "I":
                    self.ki += 0.000005
                    print(f"\n  Ki = {self.ki:.6f}")
                elif key == "i":
                    self.ki = max(0, self.ki - 0.000005)
                    print(f"\n  Ki = {self.ki:.6f}")

                # --- reset PI ---
                elif key in ("r", "R"):
                    self.reset_pi()
                    print("\n  [i] PI state reset")

                # --- log ---
                elif key in ("l", "L"):
                    lv, rv = self._get_vel()
                    print(f"\n  LOG  Kp={self.kp:.5f} Ki={self.ki:.6f} "
                          f"corr={self.correction:+.5f} "
                          f"integral={self.integral:+.1f} "
                          f"Lvel={lv} Rvel={rv}")

                # --- quit ---
                elif key in ("q", "Q", "\x03"):
                    break

                # compute correction only while moving
                if self.driving != 0:
                    correction = self.compute_correction()
                else:
                    correction = 0.0

                # apply: base offset + PI correction
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
            print(f"\n\n  Robot stopped.")
            print(f"  Final correction: {self.correction:+.5f}")
            print(f"  Final gains: Kp={self.kp:.5f}  Ki={self.ki:.6f}\n")


if __name__ == "__main__":
    ctrl = StraightLineController()
    ctrl.run()
