#!/usr/bin/env python3
"""
JetRacer Unified Controller
============================
Single entry point that merges all modes:
  - Autonomous exploration with vision-based obstacle avoidance
  - Automatic return-to-home every N minutes
  - IMU + wheel straight-line correction on every command
  - Inline WASD manual override with auto-resume
  - Runtime home-point reset via 'H' key

State machine:  AUTONOMOUS -> RETURNING_HOME -> AT_HOME -> AUTONOMOUS
                     |                                      ^
                     v                                      |
               MANUAL_OVERRIDE  -----(idle timeout)--------+

Usage:
    python main.py                      # defaults
    python main.py --no-check           # skip Docker check
    python main.py --return-interval 60 # return home every 60s
"""

import os
import sys
import time
import math
import yaml
import signal
import logging
import threading
import subprocess
import termios
import tty
import select
from enum import Enum
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports from auto/ modules
# ---------------------------------------------------------------------------
from auto.ros_bridge import ROSTopicBridge
from auto.control_adapter import ControlAdapter, SafetyMonitor
from auto.navigation import WaypointNavigator
from auto.fusion_controller import FusionController, VisionCommand
from auto.homing_controller import HomingController, HomingMode
from auto.straight_line_corrector import quaternion_to_yaw_deg
from auto.display import DisplayWindow
from auto.exploration import ExplorationStrategy


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class RobotState(Enum):
    AUTONOMOUS = "autonomous"
    RETURNING_HOME = "returning_home"
    AT_HOME = "at_home"
    MANUAL_OVERRIDE = "manual_override"


# ---------------------------------------------------------------------------
# Non-blocking keyboard input (threaded)
# ---------------------------------------------------------------------------
class KeyboardHandler:
    """Threaded non-blocking keyboard reader for same-terminal WASD control."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_key = None
        self._running = False
        self._thread = None
        self._old_settings = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def poll(self):
        """Return the latest key and clear it. Returns None if no key."""
        with self._lock:
            k = self._latest_key
            self._latest_key = None
            return k

    def _read_loop(self):
        try:
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                    if ch == '\x1b':
                        # skip escape sequences
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            sys.stdin.read(2)
                        continue
                    with self._lock:
                        self._latest_key = ch
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Unified Controller
# ---------------------------------------------------------------------------
class UnifiedController:
    """
    Merges autonomous exploration, homing, straight-line correction,
    and manual override into a single control loop.
    """

    def __init__(self, return_interval=None, no_check=False):
        self.running = False
        self.setup_logging()
        self.load_config()

        # Override return interval from CLI if given
        if return_interval is not None:
            self.config.setdefault('homing', {})['return_interval_seconds'] = return_interval

        self.no_check = no_check

        # --- unified controller params ---
        uc = self.config.get('unified_controller', {})
        self.manual_timeout = uc.get('manual_override_timeout', 3.0)
        self.home_pause = uc.get('home_pause_seconds', 2.0)
        self.status_hz = uc.get('status_display_hz', 2)

        # --- state ---
        self.state = RobotState.AUTONOMOUS
        self.last_manual_time = 0.0
        self.home_arrival_time = 0.0

        # Current manual velocities
        self.manual_linear = 0.0
        self.manual_angular = 0.0

        # Speed settings
        ctrl = self.robot_config.get('control', {})
        self.linear_speed = ctrl.get('default_linear_speed', 0.3)
        self.angular_speed = ctrl.get('default_angular_speed', 0.8)

        # --- components ---
        use_tinygrad = self.config.get('inference', {}).get('use_tinygrad', True)
        vision_mode = self.config.get('vision', {}).get('mode', 'mobilenet')

        self.logger.info("=" * 60)
        self.logger.info("Initializing Unified JetRacer Controller")
        self.logger.info(f"Vision: {vision_mode} | Backend: {'tinygrad' if use_tinygrad else 'PyTorch'}")
        self.logger.info("=" * 60)

        # ROS bridge
        self.ros_bridge = ROSTopicBridge()

        # Control adapter (with integrated straight-line corrector)
        self.control = ControlAdapter(self.config)
        self.control.set_ros_bridge(self.ros_bridge)

        # Navigator
        self.navigator = WaypointNavigator(self.config)

        # Fusion controller
        self.fusion = FusionController(self.config)

        # Homing controller
        self.homing = HomingController(self.config)

        # Safety monitor
        self.safety = SafetyMonitor(self.config)

        # Vision + policy (lazy-loaded in _init_vision)
        self.vision = None
        self.policy = None
        self.evaluator = None
        self.camera = None
        self.lightweight_vision = None
        self.use_lightweight = False
        self._vision_mode = vision_mode
        self._use_tinygrad = use_tinygrad

        # Last vision prediction
        self.last_action = 'straight'
        self.last_confidence = 0.0
        self.last_frame = None

        # Display window
        cam_w = self.config.get('vision', {}).get('camera_width', 640)
        cam_h = self.config.get('vision', {}).get('camera_height', 480)
        self.display = DisplayWindow(self.config, width=cam_w, height=cam_h)

        # Exploration strategy (gives autonomous mode purpose)
        self.explorer = ExplorationStrategy(self.config)

        # Keyboard handler
        self.keyboard = KeyboardHandler()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup_logging(self):
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"unified_{ts}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger("unified")
        self.logger.info(f"Log file: {log_file}")

    def load_config(self):
        config_dir = Path(__file__).parent / "config"
        with open(config_dir / "robot_config.yaml") as f:
            self.robot_config = yaml.safe_load(f)
        with open(config_dir / "autonomous_config.yaml") as f:
            self.config = yaml.safe_load(f)
        # Merge robot_params into config so corrector can read wheel_base etc.
        self.config['robot_params'] = self.robot_config.get('robot_params', {})

    def _init_vision(self):
        """Lazy-initialize vision pipeline (heavy imports)."""
        from auto.vision import VisionProcessor, CameraStream
        from auto.policy import PolicyNetwork, PolicyEvaluator

        if self._vision_mode == 'direct':
            from auto.lightweight_vision import LightweightVisionProcessor
            self.lightweight_vision = LightweightVisionProcessor(
                self.config, mode='direct', use_tinygrad=self._use_tinygrad)
            self.use_lightweight = True
            self.logger.info("Vision: DirectPolicyNet (end-to-end)")
        elif self._vision_mode == 'mobilenet':
            from auto.lightweight_vision import LightweightVisionProcessor
            self.lightweight_vision = LightweightVisionProcessor(
                self.config, mode='mobilenet', use_tinygrad=False)
            self.policy = PolicyNetwork(self.config, use_tinygrad=self._use_tinygrad)
            self.use_lightweight = True
            self.logger.info("Vision: MobileNetV3 + TinyNet")
        else:
            self.vision = VisionProcessor(self.config, use_tinygrad=self._use_tinygrad)
            self.policy = PolicyNetwork(self.config, use_tinygrad=self._use_tinygrad)
            self.logger.info("Vision: YOLOv8 + TinyNet")

        self.evaluator = PolicyEvaluator(self.config)

        cam_dev = self.robot_config['hardware']['camera_device']
        cam_w = self.config['vision']['camera_width']
        cam_h = self.config['vision']['camera_height']
        self.camera = CameraStream(device=cam_dev, width=cam_w, height=cam_h)
        self.camera.start()
        self.logger.info("Camera started")

    # ------------------------------------------------------------------
    # Prerequisites
    # ------------------------------------------------------------------

    def check_docker(self):
        try:
            r = subprocess.run(
                ["docker", "ps", "--filter", "name=jetracer_ros", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5)
            if "jetracer_ros" in r.stdout:
                print("  Docker container running")
                return True
            print("  Docker container 'jetracer_ros' not running")
            print("  Start with: ~/start_jetracer_docker.sh")
            return False
        except Exception as e:
            print(f"  Docker check error: {e}")
            return False

    def start_ros_infra(self):
        """Ensure roscore + robot base are up."""
        try:
            r = subprocess.run(
                ["docker", "exec", "jetracer_ros", "bash", "-c",
                 "source /root/catkin_ws/devel/setup.bash && rostopic list"],
                capture_output=True, timeout=3)
            if r.returncode == 0:
                self.logger.info("ROS master already running")
                return
        except Exception:
            pass

        print("  Starting ROS core ...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && roscore"], check=False)
        time.sleep(3)
        print("  Starting robot base ...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && rosrun jetracer jetracer"], check=False)
        time.sleep(2)
        print("  ROS infrastructure ready")

    # ------------------------------------------------------------------
    # Vision prediction
    # ------------------------------------------------------------------

    def _predict_action(self, frame):
        """Run vision pipeline on a camera frame. Returns (action, confidence)."""
        if frame is None:
            return self.last_action, 0.0

        try:
            if self.use_lightweight:
                if self.lightweight_vision is not None:
                    if self._vision_mode == 'direct':
                        action, confidence, _ = self.lightweight_vision.predict(frame)
                    else:
                        features = self.lightweight_vision.process_frame(frame)
                        if features is None:
                            return self.last_action, 0.0
                        action, confidence, _ = self.policy.predict(features)
                else:
                    return self.last_action, 0.0
            else:
                features = self.vision.process_frame(frame)
                if features is None:
                    return self.last_action, 0.0
                action, confidence, _ = self.policy.predict(features)
        except Exception as e:
            self.logger.debug(f"Vision error: {e}")
            return self.last_action, 0.0

        self.last_action = action
        self.last_confidence = confidence
        return action, confidence

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_key(self, key):
        """Process a single key press. Returns True if robot should quit."""
        k = key.lower()

        if k == 'q' or key == '\x03':
            return True

        now = time.time()

        # Manual movement keys
        if k in ('w', 'a', 's', 'd', ' '):
            if k == 'w':
                self.manual_linear = self.linear_speed
                self.manual_angular = 0.0
            elif k == 's':
                self.manual_linear = -self.linear_speed
                self.manual_angular = 0.0
            elif k == 'a':
                self.manual_linear = 0.0
                self.manual_angular = self.angular_speed
            elif k == 'd':
                self.manual_linear = 0.0
                self.manual_angular = -self.angular_speed
            elif k == ' ':
                self.manual_linear = 0.0
                self.manual_angular = 0.0

            self.last_manual_time = now
            if self.state != RobotState.MANUAL_OVERRIDE:
                self.state = RobotState.MANUAL_OVERRIDE
                self.logger.info("Manual override activated")
            return False

        # Home reset
        if k == 'h':
            x, y, theta = self.navigator.get_position()
            self.homing.set_home(x, y, theta)
            print(f"\n  Home reset to ({x:.2f}, {y:.2f})")
            return False

        # Speed adjust
        if key in ('+', '='):
            self.linear_speed = min(self.linear_speed + 0.05, 1.0)
            self.angular_speed = min(self.angular_speed + 0.1, 2.0)
            print(f"\n  Speed up: L={self.linear_speed:.2f} A={self.angular_speed:.2f}")
        elif key in ('-', '_'):
            self.linear_speed = max(self.linear_speed - 0.05, 0.1)
            self.angular_speed = max(self.angular_speed - 0.1, 0.2)
            print(f"\n  Speed down: L={self.linear_speed:.2f} A={self.angular_speed:.2f}")

        return False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _maybe_resume_autonomous(self):
        """Resume autonomous after manual idle timeout."""
        if self.state != RobotState.MANUAL_OVERRIDE:
            return
        if time.time() - self.last_manual_time >= self.manual_timeout:
            self.state = RobotState.AUTONOMOUS
            self.manual_linear = 0.0
            self.manual_angular = 0.0
            self.logger.info("Manual idle timeout -> autonomous")

    def _maybe_return_home(self):
        """Check if it's time to go home."""
        if self.state != RobotState.AUTONOMOUS:
            return
        if self.homing.check_should_return_home():
            hx, hy, _ = self.homing.get_home()
            self.navigator.set_goal(hx, hy)
            self.homing.start_return_home()
            self.state = RobotState.RETURNING_HOME
            self.logger.info(f"Time to return home -> ({hx:.2f}, {hy:.2f})")

    def _check_home_arrival(self):
        """While RETURNING_HOME, check whether we've arrived."""
        if self.state != RobotState.RETURNING_HOME:
            return
        hx, hy, _ = self.homing.get_home()
        dist = self.navigator.distance_to(hx, hy)
        tolerance = self.config.get('homing', {}).get('home_tolerance', 0.3)
        if dist < tolerance:
            self.state = RobotState.AT_HOME
            self.home_arrival_time = time.time()
            self.homing.mark_home_reached()
            self.navigator.clear_waypoints()
            self.control.stop()
            self.logger.info("Arrived home")

    def _check_home_pause_done(self):
        """After AT_HOME pause, resume exploration."""
        if self.state != RobotState.AT_HOME:
            return
        if time.time() - self.home_arrival_time >= self.home_pause:
            self.homing.resume_exploration()
            self.state = RobotState.AUTONOMOUS
            self.logger.info("Home pause done -> autonomous")

    # ------------------------------------------------------------------
    # Stuck recovery (during RETURNING_HOME)
    # ------------------------------------------------------------------

    def _check_stuck(self):
        if self.state != RobotState.RETURNING_HOME:
            return
        if self.homing.is_stuck():
            self.logger.warning("Robot appears stuck - running recovery")
            if self.homing.start_stuck_recovery():
                while True:
                    cmd = self.homing.get_next_recovery_command()
                    if cmd is None:
                        break
                    self.control.send_velocity_command(
                        cmd.linear_velocity, cmd.angular_velocity)
                    time.sleep(cmd.duration)
                self.control.stop()
                # Re-set goal after recovery
                hx, hy, _ = self.homing.get_home()
                self.navigator.set_goal(hx, hy)

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def _print_status(self):
        x, y, theta = self.navigator.get_position()
        hx, hy, _ = self.homing.get_home()
        dist_home = math.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        ttr = self.homing.get_time_until_return()
        corr = self.control.corrector.get_status()
        err_str = f"err:{corr['yaw_error']:+5.1f}d" if corr['locked'] else "free"

        line = (
            f"\r  [{self.state.value:^16s}] "
            f"pos:({x:+5.2f},{y:+5.2f}) th:{math.degrees(theta):+6.1f}d | "
            f"home:({hx:+5.2f},{hy:+5.2f}) dist:{dist_home:4.2f}m | "
            f"act:{self.last_action:>8s} c:{self.last_confidence:.2f} | "
            f"IMU:{err_str} | "
            f"ret:{ttr:5.0f}s"
        )
        print(f"{line:<160}", end="", flush=True)

    def _update_display(self, linear, angular):
        """Push current state into the camera display window."""
        x, y, theta = self.navigator.get_position()
        hx, hy, _ = self.homing.get_home()
        dist_home = math.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        ttr = self.homing.get_time_until_return()
        corr = self.control.corrector.get_status()

        # Grab latest frame for display (use cached last_frame)
        frame = self.last_frame
        if frame is None and self.camera:
            frame = self.camera.read()

        self.display.update(
            frame,
            state=self.state.value,
            action=self.last_action,
            confidence=self.last_confidence,
            pos=(x, y, theta),
            home=(hx, hy),
            dist_home=dist_home,
            linear=linear,
            angular=angular,
            imu_locked=corr['locked'],
            imu_yaw_err=corr.get('yaw_error', 0.0),
            time_to_return=ttr,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        # Pre-flight
        if not self.no_check:
            if not self.check_docker():
                return False
        self.start_ros_infra()

        # Start ROS bridge
        self.ros_bridge.start()
        time.sleep(1)

        # Initialize navigator ROS connection
        self.navigator.initialize_ros()

        # Initialize vision
        self._init_vision()

        # Start homing session
        self.homing.start_session()

        # Lock heading to current IMU yaw for straight-line correction
        imu_yaw, _, _ = self.ros_bridge.get_imu_yaw_gyro()
        self.control.corrector.lock_heading(imu_yaw)

        # Safety monitor
        self.safety.start()

        # Keyboard
        self.keyboard.start()

        # Display window
        self.display.start()

        self.running = True
        prediction_interval = 1.0 / self.config.get('inference', {}).get('prediction_rate', 5)
        last_pred_time = 0.0
        last_status_time = 0.0
        status_interval = 1.0 / max(self.status_hz, 0.5)

        self._print_banner()

        try:
            while self.running:
                loop_start = time.time()

                # ---- Keyboard ----
                key = self.keyboard.poll()
                if key is not None:
                    if self._handle_key(key):
                        break

                # ---- State transitions ----
                self._maybe_resume_autonomous()
                self._maybe_return_home()
                self._check_home_arrival()
                self._check_home_pause_done()
                self._check_stuck()

                # ---- Compute command based on state ----
                linear = 0.0
                angular = 0.0
                is_turning = False

                if self.state == RobotState.MANUAL_OVERRIDE:
                    linear = self.manual_linear
                    angular = self.manual_angular
                    is_turning = abs(angular) > 0.1
                    # Grab frame for display (no vision inference)
                    if self.camera and loop_start - last_pred_time >= prediction_interval:
                        self.last_frame = self.camera.read()
                        last_pred_time = loop_start

                elif self.state == RobotState.AUTONOMOUS:
                    # Vision prediction (throttled)
                    if loop_start - last_pred_time >= prediction_interval:
                        frame = self.camera.read() if self.camera else None
                        self.last_frame = frame
                        action, confidence = self._predict_action(frame)
                        last_pred_time = loop_start

                        if self.evaluator and self.evaluator.should_execute(action, confidence):
                            self.last_action = action
                            self.last_confidence = confidence
                            self.safety.log_action(action)

                    # Convert last action to velocities
                    linear, angular = self.control.action_to_velocity(self.last_action)
                    is_turning = self.last_action in ('left', 'right')

                    # Exploration: steer toward unvisited frontier
                    ex, ey, eth = self.navigator.get_position()
                    self.explorer.update(ex, ey, eth)
                    angular = self.explorer.compute_angular_bias(
                        self.last_action, self.last_confidence, linear, angular)
                    is_turning = is_turning or abs(angular) > 0.3

                elif self.state == RobotState.RETURNING_HOME:
                    # Use navigator for heading toward home
                    nav_cmd = self.navigator.get_navigation_command()

                    # Also get vision for obstacle avoidance
                    if loop_start - last_pred_time >= prediction_interval:
                        frame = self.camera.read() if self.camera else None
                        self.last_frame = frame
                        action, confidence = self._predict_action(frame)
                        last_pred_time = loop_start

                    vision_cmd = VisionCommand(
                        action=self.last_action,
                        confidence=self.last_confidence,
                        obstacle_detected=(self.last_action in ('left', 'right', 'reverse')
                                           and self.last_confidence > 0.5),
                    )

                    x, y, theta = self.navigator.get_position()
                    fused = self.fusion.fuse_commands(nav_cmd, vision_cmd, theta)
                    linear = fused.linear_x
                    angular = fused.angular_z
                    is_turning = abs(angular) > 0.3

                    # Record for homing dead-reckoning
                    self.homing.record_command(linear, angular)

                elif self.state == RobotState.AT_HOME:
                    linear = 0.0
                    angular = 0.0

                # ---- Update heading lock ----
                if self.state in (RobotState.AUTONOMOUS, RobotState.RETURNING_HOME):
                    if not self.control.corrector.is_locked():
                        imu_yaw, _, _ = self.ros_bridge.get_imu_yaw_gyro()
                        self.control.corrector.lock_heading(imu_yaw)
                elif self.state == RobotState.MANUAL_OVERRIDE:
                    # Re-lock heading when driving straight
                    if abs(self.manual_linear) > 0.01 and abs(self.manual_angular) < 0.01:
                        if not self.control.corrector.is_locked():
                            imu_yaw, _, _ = self.ros_bridge.get_imu_yaw_gyro()
                            self.control.corrector.lock_heading(imu_yaw)
                    elif abs(self.manual_angular) > 0.01:
                        self.control.corrector.unlock_heading()

                # ---- Send command (with correction) ----
                if self.state != RobotState.AT_HOME:
                    self.control.send_corrected_command(linear, angular, is_turning=is_turning)

                # ---- Display + Status line ----
                if loop_start - last_status_time >= status_interval:
                    self._print_status()
                    self._update_display(linear, angular)
                    last_status_time = loop_start

                # Safety timeout
                if self.safety.check_timeout():
                    self.logger.warning("Safety timeout")
                    break

                # Loop pacing (~30 Hz)
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, (1.0 / 30.0) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

        return True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, signum=None, frame=None):
        if not self.running:
            return
        self.running = False
        print("\n\n  Shutting down ...")
        self.control.stop()
        self.keyboard.stop()
        self.display.stop()
        if self.camera:
            self.camera.stop()
        self.ros_bridge.stop()
        self.homing.stop_session()
        stats = self.safety.get_statistics()
        if stats:
            self.logger.info(f"Session stats: {stats}")
        self.logger.info("Shutdown complete")
        print("  Robot stopped.\n")

    def _signal_handler(self, signum, frame):
        self.shutdown()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self):
        ret = self.config.get('homing', {}).get('return_interval_seconds', 240)
        print(f"""
======================================================
       JetRacer Unified Controller
======================================================
  WASD  = manual override (auto-resumes after {self.manual_timeout:.0f}s)
  H     = set / reset home to current position
  SPACE = stop          +/- = adjust speed
  Q     = quit

  Returning home every {ret}s ({ret/60:.1f} min)
  IMU straight-line correction: active
  Exploration: frontier-based (visited-cell tracking)
  Camera display window: active
======================================================
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='JetRacer Unified Controller')
    parser.add_argument('--return-interval', type=int, default=None,
                        help='Return home interval in seconds (default: from config)')
    parser.add_argument('--no-check', action='store_true',
                        help='Skip Docker pre-flight check')
    args = parser.parse_args()

    controller = UnifiedController(
        return_interval=args.return_interval,
        no_check=args.no_check,
    )
    controller.run()


if __name__ == "__main__":
    main()
