#!/usr/bin/env python3
"""
JetRacer Motor Diagnostics

Live monitor of left/right wheel velocities while driving.
Helps identify wheel asymmetry causing the robot to veer.

Controls:
  W / S       - Drive forward / backward
  A / D       - Nudge steering offset  (coarse:  0.05)
  LEFT / RIGHT- Nudge steering offset  (fine:    0.01)
  UP / DOWN   - Nudge steering offset  (finer:   0.001)
  [ / ]       - Nudge steering offset  (finest:  0.0001)
  + / -       - Adjust throttle speed
  SPACE       - Stop driving (keeps offset)
  R           - Reset offset to 0
  Q / Ctrl+C  - Quit
"""

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import sys
import os
import termios
import tty
import threading
import time


# ── step sizes ──────────────────────────────────────────────
COARSE_STEP  = 0.05
FINE_STEP    = 0.01
FINER_STEP   = 0.001
FINEST_STEP  = 0.0001
SPEED_STEP   = 0.05
MIN_SPEED    = 0.10
MAX_SPEED    = 0.60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_key(settings):
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(2)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        mapping = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}
        return mapping.get(seq)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return ch


class MotorDiagnostics:
    def __init__(self):
        rospy.init_node("motor_diagnostics", anonymous=True)

        # publisher
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # motor data (thread-safe)
        self.lock = threading.Lock()
        self.lvel = 0.0   # left wheel actual velocity
        self.rvel = 0.0   # right wheel actual velocity
        self.lset = 0.0   # left wheel set command
        self.rset = 0.0   # right wheel set command

        # subscribers (Int32 topics on jetracer ROS)
        rospy.Subscriber("/motor/lvel", Int32, self._cb_lvel)
        rospy.Subscriber("/motor/rvel", Int32, self._cb_rvel)
        rospy.Subscriber("/motor/lset", Int32, self._cb_lset)
        rospy.Subscriber("/motor/rset", Int32, self._cb_rset)

        # driving state
        self.offset  = -0.0450   # start from your known-good value
        self.speed   = 0.20
        self.driving = 0         # -1 rev, 0 stop, 1 fwd

    # ── callbacks ──────────────────────────────────────────
    def _cb_lvel(self, msg): 
        with self.lock: self.lvel = msg.data
    def _cb_rvel(self, msg): 
        with self.lock: self.rvel = msg.data
    def _cb_lset(self, msg): 
        with self.lock: self.lset = msg.data
    def _cb_rset(self, msg): 
        with self.lock: self.rset = msg.data

    def _get_motors(self):
        with self.lock:
            return self.lvel, self.rvel, self.lset, self.rset

    # ── display ────────────────────────────────────────────
    def print_hud(self):
        lv, rv, ls, rs = self._get_motors()
        diff_vel = lv - rv
        diff_set = ls - rs
        direction = {1: "FWD", -1: "REV"}.get(self.driving, "---")
        state = "DRIVE" if self.driving != 0 else "STOP "

        # bar for offset
        bar_len = 30
        pos = int((self.offset + 1.0) / 2.0 * bar_len)
        pos = clamp(pos, 0, bar_len)
        bar = "-" * pos + "|" + "-" * (bar_len - pos)

        line = (
            f"\r  [{state} {direction}] "
            f"offset:{self.offset:+.4f} [{bar}] "
            f"spd:{self.speed:.2f}  |  "
            f"Lvel:{lv:+6d}  Rvel:{rv:+6d}  diff:{diff_vel:+6d}  |  "
            f"Lset:{ls:+6d}  Rset:{rs:+6d}  diff:{diff_set:+6d}"
        )
        # pad to overwrite previous content
        print(f"{line:<160}", end="", flush=True)

    # ── main loop ──────────────────────────────────────────
    def run(self):
        settings = termios.tcgetattr(sys.stdin)

        print(__doc__)
        print(f"  Starting offset: {self.offset:+.4f}")
        print(f"  Subscribed to /motor/lvel, /motor/rvel, /motor/lset, /motor/rset")
        print(f"  Publishing to  /cmd_vel\n")

        # wait briefly for subscribers to connect
        time.sleep(1.0)

        twist = Twist()
        rate  = rospy.Rate(20)

        try:
            while not rospy.is_shutdown():
                key = read_key(settings)
                if key is None:
                    pass

                # driving
                elif key in ("w", "W"):
                    self.driving = 1
                elif key in ("s", "S"):
                    self.driving = -1
                elif key == " ":
                    self.driving = 0

                # coarse offset
                elif key in ("a", "A"):
                    self.offset = clamp(self.offset + COARSE_STEP, -1.0, 1.0)
                elif key in ("d", "D"):
                    self.offset = clamp(self.offset - COARSE_STEP, -1.0, 1.0)

                # fine
                elif key == "LEFT":
                    self.offset = clamp(self.offset + FINE_STEP, -1.0, 1.0)
                elif key == "RIGHT":
                    self.offset = clamp(self.offset - FINE_STEP, -1.0, 1.0)

                # finer
                elif key == "UP":
                    self.offset = clamp(self.offset + FINER_STEP, -1.0, 1.0)
                elif key == "DOWN":
                    self.offset = clamp(self.offset - FINER_STEP, -1.0, 1.0)

                # finest
                elif key == "[":
                    self.offset = clamp(self.offset + FINEST_STEP, -1.0, 1.0)
                elif key == "]":
                    self.offset = clamp(self.offset - FINEST_STEP, -1.0, 1.0)

                # speed
                elif key in ("+", "="):
                    self.speed = clamp(self.speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
                elif key in ("-", "_"):
                    self.speed = clamp(self.speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)

                # reset
                elif key in ("r", "R"):
                    self.offset = 0.0
                    print("\n  [i] Offset reset to 0")

                # quit
                elif key in ("q", "Q", "\x03"):
                    break

                # publish — negate offset in reverse
                twist.linear.x  = self.speed * self.driving
                twist.angular.z = self.offset if self.driving >= 0 else -self.offset
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
            print(f"\n\n  Robot stopped.  Final offset: {self.offset:+.4f}\n")


if __name__ == "__main__":
    diag = MotorDiagnostics()
    diag.run()
