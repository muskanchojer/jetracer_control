#!/usr/bin/env python3
"""
JetRacer Keyboard Control with Yaw Lock
When moving forward (W), the robot locks onto the current IMU yaw heading
and uses a PID controller to correct steering, keeping it perfectly straight.
Any other direction command (A/D/S/SPACE) releases the yaw lock.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import math
import sys
import termios
import tty
import threading
import time


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in degrees (-180 to 180)"""
    yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw_rad)


def normalize_angle(angle):
    """Normalize angle to [-180, 180] range"""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class YawLockedController:
    def __init__(self):
        rospy.init_node("yaw_locked_keyboard_teleop")

        # Publishers / Subscribers
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/imu", Imu, self.imu_callback)

        # Speed settings
        self.linear_speed = 0.3
        self.angular_speed = 0.8

        # Current commanded values
        self.current_linear = 0.0
        self.current_angular = 0.0

        # IMU state
        self.current_yaw = 0.0
        self.imu_ready = False
        self.imu_lock = threading.Lock()

        # Yaw-lock state
        self.yaw_locked = False
        self.target_yaw = 0.0

        # PID gains for yaw correction (tune these for your robot)
        self.kp = 0.5     # Proportional gain  (1° error → 0.5 steering)
        self.ki = 0.05    # Integral gain      (persistent drift buildup)
        self.kd = 0.15    # Derivative gain    (dampen oscillation)

        # PID state
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.pid_max_integral = 20.0   # Anti-windup limit
        self.pid_max_output = 1.0      # Max steering correction (was 0.5)

        # Direction labels for display
        self.direction_label = "STOPPED"

    def imu_callback(self, msg):
        """Update current yaw from IMU data"""
        o = msg.orientation
        yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        with self.imu_lock:
            self.current_yaw = yaw
            self.imu_ready = True

    def get_yaw(self):
        """Thread-safe read of current yaw"""
        with self.imu_lock:
            return self.current_yaw

    def lock_yaw(self):
        """Lock onto the current yaw heading"""
        self.target_yaw = self.get_yaw()
        self.yaw_locked = True
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0

    def unlock_yaw(self):
        """Release the yaw lock"""
        self.yaw_locked = False
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0

    def compute_yaw_correction(self):
        """PID controller: compute angular.z correction to maintain target yaw"""
        if not self.yaw_locked:
            return 0.0

        current = self.get_yaw()
        error = normalize_angle(self.target_yaw - current)

        # Proportional
        p_out = self.kp * error

        # Integral (with anti-windup)
        self.pid_integral += error
        self.pid_integral = max(-self.pid_max_integral,
                                min(self.pid_max_integral, self.pid_integral))
        i_out = self.ki * self.pid_integral

        # Derivative
        d_out = self.kd * (error - self.pid_prev_error)
        self.pid_prev_error = error

        # Total correction (clamped)
        correction = p_out + i_out + d_out
        correction = max(-self.pid_max_output,
                         min(self.pid_max_output, correction))

        return correction

    def print_status(self):
        """Print current status line"""
        yaw_now = self.get_yaw()
        if self.yaw_locked:
            error = normalize_angle(self.target_yaw - yaw_now)
            status = (f"{self.direction_label} | YAW-LOCKED  "
                      f"Target: {self.target_yaw:+7.2f}°  "
                      f"Current: {yaw_now:+7.2f}°  "
                      f"Error: {error:+6.2f}°  "
                      f"Correction: {self.current_angular:+.3f}")
        else:
            status = (f"{self.direction_label} | "
                      f"Yaw: {yaw_now:+7.2f}°  "
                      f"Speed: {self.current_linear:+.2f}  "
                      f"Steer: {self.current_angular:+.2f}")
        print(status)

    def run(self):
        """Main control loop"""
        settings = termios.tcgetattr(sys.stdin)

        print("""
=====================================================
   JetRacer Keyboard Control  (YAW-LOCK MODE)
=====================================================
  W - Forward  (LOCKS yaw heading, auto-corrects)
  S - Backward (no yaw lock)
  A - Steer Left        D - Steer Right
  X - Center steering   SPACE - Full Stop

  + / - : Adjust speed
  P / O / I : Adjust PID gains (Kp / Ki / Kd)
  Q - Quit
=====================================================
  When W is pressed, the robot locks its current
  heading and uses IMU + PID to drive perfectly
  straight until you give another command.
=====================================================
  Current Speed: Linear=%.2f  Angular=%.2f
  PID Gains: Kp=%.4f  Ki=%.4f  Kd=%.4f
=====================================================
""" % (self.linear_speed, self.angular_speed,
       self.kp, self.ki, self.kd))

        # Wait for IMU data
        print("Waiting for IMU data...")
        timeout = time.time() + 10
        while not self.imu_ready and time.time() < timeout:
            time.sleep(0.1)

        if self.imu_ready:
            print("IMU ready!  Yaw: %.2f°\n" % self.get_yaw())
        else:
            print("WARNING: No IMU data received! Yaw-lock will not work.\n")

        rate = rospy.Rate(20)  # 20 Hz control loop
        twist = Twist()

        try:
            while not rospy.is_shutdown():
                # Non-blocking key read
                key = self._getKey(settings)

                if key == "w" or key == "W":
                    # FORWARD with yaw lock
                    self.current_linear = self.linear_speed
                    if not self.yaw_locked:
                        self.lock_yaw()
                    self.direction_label = "FORWARD"

                elif key == "s" or key == "S":
                    # BACKWARD - release yaw lock
                    self.unlock_yaw()
                    self.current_linear = -self.linear_speed
                    self.current_angular = 0.0
                    self.direction_label = "BACKWARD"

                elif key == "a" or key == "A":
                    # STEER LEFT - release yaw lock
                    self.unlock_yaw()
                    self.current_angular = self.angular_speed
                    self.direction_label = "STEER LEFT"

                elif key == "d" or key == "D":
                    # STEER RIGHT - release yaw lock
                    self.unlock_yaw()
                    self.current_angular = -self.angular_speed
                    self.direction_label = "STEER RIGHT"

                elif key == "x" or key == "X":
                    # CENTER STEERING - release yaw lock
                    self.unlock_yaw()
                    self.current_angular = 0.0
                    self.direction_label = "CENTERED"

                elif key == " ":
                    # FULL STOP - release yaw lock
                    self.unlock_yaw()
                    self.current_linear = 0.0
                    self.current_angular = 0.0
                    self.direction_label = "STOPPED"

                elif key == "+" or key == "=":
                    self.linear_speed = min(self.linear_speed + 0.1, 1.0)
                    self.angular_speed = min(self.angular_speed + 0.2, 2.0)
                    print("Speed UP: Linear=%.2f Angular=%.2f" % (self.linear_speed, self.angular_speed))

                elif key == "-" or key == "_":
                    self.linear_speed = max(self.linear_speed - 0.1, 0.1)
                    self.angular_speed = max(self.angular_speed - 0.2, 0.2)
                    print("Speed DOWN: Linear=%.2f Angular=%.2f" % (self.linear_speed, self.angular_speed))

                # PID tuning keys
                elif key == "p":
                    self.kp += 0.005
                    print("Kp increased: %.4f" % self.kp)
                elif key == "P":
                    self.kp = max(0, self.kp - 0.005)
                    print("Kp decreased: %.4f" % self.kp)
                elif key == "o":
                    self.ki += 0.001
                    print("Ki increased: %.4f" % self.ki)
                elif key == "O":
                    self.ki = max(0, self.ki - 0.001)
                    print("Ki decreased: %.4f" % self.ki)
                elif key == "i":
                    self.kd += 0.002
                    print("Kd increased: %.4f" % self.kd)
                elif key == "I":
                    self.kd = max(0, self.kd - 0.002)
                    print("Kd decreased: %.4f" % self.kd)

                elif key == "q" or key == "Q":
                    print("Quitting...")
                    break

                elif key == "\x03":  # Ctrl+C
                    break

                # If yaw-locked (forward), PID controls the steering
                if self.yaw_locked:
                    self.current_angular = self.compute_yaw_correction()

                # Publish command
                twist.linear.x = self.current_linear
                twist.angular.z = self.current_angular
                self.pub.publish(twist)

                # Print status
                self.print_status()

                rate.sleep()

        except Exception as e:
            print("Error: %s" % e)

        finally:
            # Stop the robot
            twist.linear.x = 0
            twist.angular.z = 0
            self.pub.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            print("\nRobot stopped. Yaw lock released.")

    @staticmethod
    def _getKey(settings):
        """Read a single keypress"""
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key


if __name__ == "__main__":
    try:
        controller = YawLockedController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
