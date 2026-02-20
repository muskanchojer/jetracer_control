#!/usr/bin/env python3
"""
JetRacer Steering Calibration Tool

Adjust the steering offset in fine increments until the robot drives straight.
The calibrated value is saved to a file so other scripts can use it.

Controls:
  W / S          - Drive forward / backward at current speed
  A / D          - Nudge offset LEFT / RIGHT  (coarse:     0.05)
  LEFT / RIGHT   - Nudge offset LEFT / RIGHT  (fine:       0.01)
  UP / DOWN      - Nudge offset LEFT / RIGHT  (finer:      0.001)
  [ / ]          - Nudge offset LEFT / RIGHT  (finest:     0.0001)
  + / -          - Increase / decrease throttle speed
  SPACE          - Stop driving (keeps offset)
  R              - Reset steering offset to 0
  V              - Save current offset to config
  Q / Ctrl+C     - Quit (auto-stops robot)
"""

import rospy
from geometry_msgs.msg import Twist
import sys
import os
import termios
import tty
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "robot_config.yaml")

COARSE_STEP  = 0.05    # A/D nudge
FINE_STEP    = 0.01    # LEFT/RIGHT arrow nudge
FINER_STEP   = 0.001   # UP/DOWN arrow nudge
FINEST_STEP  = 0.0001  # [ / ] nudge
SPEED_STEP   = 0.05
MIN_SPEED    = 0.10
MAX_SPEED    = 0.60


def read_key(settings):
    """Read a single keypress, handling arrow-key escape sequences."""
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    if ch == "\x1b":                       # ESC sequence (arrow keys)
        seq = sys.stdin.read(2)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if seq == "[D":                    # LEFT arrow
            return "LEFT"
        elif seq == "[C":                  # RIGHT arrow
            return "RIGHT"
        elif seq == "[A":                  # UP arrow
            return "UP"
        elif seq == "[B":                  # DOWN arrow
            return "DOWN"
        return None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return ch


def load_existing_offset():
    """Try to read a previously saved servo_bias from of config."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("robot_params", {}).get("steering_offset", 0.0))
    except Exception:
        return 0.0


def save_offset_to_config(offset):
    """Write the steering_offset into the YAML config file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}

        cfg.setdefault("robot_params", {})["steering_offset"] = round(offset, 4)

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"\n  [!] Failed to save config: {e}")
        return False


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def print_hud(offset, speed, driving):
    """Single-line heads-up display, reprinted in place."""
    state = "DRIVING" if driving != 0 else "STOPPED"
    direction = {1: "FWD", -1: "REV"}.get(driving, "---")
    bar_len = 40
    # Map offset from [-1, 1] to bar position
    pos = int((offset + 1.0) / 2.0 * bar_len)
    pos = clamp(pos, 0, bar_len)
    bar = "-" * pos + "|" + "-" * (bar_len - pos)
    print(f"\r  [{state} {direction}]  offset: {offset:+.4f}  speed: {speed:.2f}  [{bar}]    ", end="", flush=True)


def main():
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node("steering_calibration", anonymous=True)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    offset = load_existing_offset()
    speed  = 0.20
    driving = 0        # -1 reverse, 0 stopped, 1 forward

    print(__doc__)
    print(f"  Loaded steering offset: {offset:+.4f}")
    print(f"  Config file: {os.path.abspath(CONFIG_PATH)}")
    print()

    twist = Twist()
    rate  = rospy.Rate(20)

    try:
        while not rospy.is_shutdown():
            key = read_key(settings)
            if key is None:
                pass

            # --- driving ---
            elif key in ("w", "W"):
                driving = 1
            elif key in ("s", "S"):
                driving = -1
            elif key == " ":
                driving = 0

            # --- coarse offset (A/D) ---
            elif key in ("a", "A"):
                offset = clamp(offset + COARSE_STEP, -1.0, 1.0)
            elif key in ("d", "D"):
                offset = clamp(offset - COARSE_STEP, -1.0, 1.0)

            # --- fine offset (LEFT/RIGHT arrows) ---
            elif key == "LEFT":
                offset = clamp(offset + FINE_STEP, -1.0, 1.0)
            elif key == "RIGHT":
                offset = clamp(offset - FINE_STEP, -1.0, 1.0)

            # --- finer offset (UP/DOWN arrows) ---
            elif key == "UP":
                offset = clamp(offset + FINER_STEP, -1.0, 1.0)
            elif key == "DOWN":
                offset = clamp(offset - FINER_STEP, -1.0, 1.0)

            # --- finest offset ([ / ]) ---
            elif key == "[":
                offset = clamp(offset + FINEST_STEP, -1.0, 1.0)
            elif key == "]":
                offset = clamp(offset - FINEST_STEP, -1.0, 1.0)

            # --- speed ---
            elif key in ("+", "="):
                speed = clamp(speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
            elif key in ("-", "_"):
                speed = clamp(speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)

            # --- utilities ---
            elif key in ("r", "R"):
                offset = 0.0
                print("\n  [i] Steering offset reset to 0.0000")
            elif key in ("v", "V"):
                if save_offset_to_config(offset):
                    print(f"\n  [OK] Saved steering_offset = {offset:+.4f} to config")
            elif key in ("q", "Q", "\x03"):
                break

            # publish — negate offset in reverse so steering tracks the same
            twist.linear.x  = speed * driving
            twist.angular.z = offset if driving >= 0 else -offset
            pub.publish(twist)

            print_hud(offset, speed, driving)
            rate.sleep()

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print(f"\n\n  Robot stopped.  Final steering offset: {offset:+.4f}")
        print("  (Press V before quitting to save, or run again.)\n")


if __name__ == "__main__":
    main()
