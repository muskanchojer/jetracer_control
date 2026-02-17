#!/usr/bin/env python3
"""
JetRacer Control System - Main5
Yaw-Locked Forward Drive Mode
When the robot moves forward, it locks onto the IMU yaw heading
and uses PID correction to drive perfectly straight.
"""

import subprocess
import time
import sys
import os
import signal
import logging
from datetime import datetime


class JetRacerYawLock:
    def __init__(self):
        self.processes = []
        self.container = "jetracer_ros"
        self.running = True
        self.setup_logging()

    def setup_logging(self):
        """Configure logging to file and console"""
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"jetracer_yawlock_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("JetRacer Yaw-Lock Control System Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 60)

    def check_hardware(self):
        """Check if robot hardware is connected"""
        self.logger.info("Checking hardware...")

        if not os.path.exists("/dev/ttyACM0"):
            self.logger.error("Robot not connected at /dev/ttyACM0")
            print("❌ Robot not connected")
            return False
        self.logger.info("Robot connected at /dev/ttyACM0")
        print("✅ Robot connected")

        if os.path.exists("/dev/video0"):
            self.logger.info("USB Camera found at /dev/video0")
            print("✅ USB Camera found")
        else:
            self.logger.warning("No camera detected at /dev/video0")
            print("⚠️  No camera detected (optional)")

        print()
        return True

    def check_docker(self):
        """Check if Docker container is running"""
        self.logger.info(f"Checking Docker container: {self.container}")

        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={self.container}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )

        if self.container in result.stdout:
            self.logger.info(f"Docker container '{self.container}' is running")
            print(f"✅ Docker running\n")
            return True

        self.logger.error(f"Docker container '{self.container}' not found")
        print(f"❌ Start Docker with: ~/start_jetracer_docker.sh\n")
        return False

    def sync_scripts(self):
        """Copy custom scripts to Docker workspace"""
        self.logger.info("Syncing custom scripts to Docker...")

        script_dir = os.path.join(os.path.dirname(__file__), "scripts")
        if not os.path.exists(script_dir):
            self.logger.warning("No custom scripts directory found")
            return

        synced_count = 0
        for script in os.listdir(script_dir):
            if script.endswith(".py"):
                src = os.path.join(script_dir, script)
                dest_path = f"/root/catkin_ws/src/jetracer/scripts/{script}"

                result = subprocess.run(
                    ["docker", "cp", src, f"{self.container}:{dest_path}"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    subprocess.run(
                        ["docker", "exec", self.container, "chmod", "+x", dest_path],
                        capture_output=True
                    )
                    synced_count += 1
                    self.logger.info(f"Synced: {script}")
                else:
                    self.logger.error(f"Failed to sync {script}: {result.stderr}")

        self.logger.info(f"Synced {synced_count} scripts")
        print(f"✅ Synced {synced_count} scripts\n")

    def run_in_docker(self, cmd):
        """Run command inside Docker container (background, silent)"""
        self.logger.debug(f"Running in Docker: {cmd}")

        full_cmd = [
            "docker", "exec", "-i",
            self.container, "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {cmd}"
        ]

        return subprocess.Popen(
            full_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def cleanup(self, signum=None, frame=None):
        """Cleanup and shutdown all processes"""
        if not self.running:
            return

        self.running = False
        self.logger.info("Shutting down...")
        print("\n\n🛑 Shutting down...")

        # Stop robot movement
        self.logger.info("Stopping robot movement")
        try:
            subprocess.run([
                "docker", "exec", self.container, "bash", "-c",
                "source /root/catkin_ws/devel/setup.bash && "
                "rostopic pub /cmd_vel geometry_msgs/Twist "
                "'{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' -1"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        except Exception as e:
            self.logger.warning(f"Failed to stop robot: {e}")

        # Terminate all processes
        for proc in self.processes:
            try:
                proc.terminate()
                self.logger.debug(f"Terminated process {proc.pid}")
            except Exception as e:
                self.logger.warning(f"Failed to terminate process: {e}")

        self.logger.info("Shutdown complete")
        print("✅ Stopped\n")
        sys.exit(0)

    def start_roscore(self):
        """Start ROS core"""
        self.logger.info("Starting roscore...")
        print("  Starting ROS core...")
        self.processes.append(self.run_in_docker("roscore"))
        time.sleep(2)

    def start_robot_base(self):
        """Start robot base node"""
        self.logger.info("Starting robot base...")
        print("  Starting robot base...")
        self.processes.append(self.run_in_docker("rosrun jetracer jetracer"))
        time.sleep(2)

    def start_yaw_lock_control(self):
        """Start keyboard control with yaw-lock"""
        self.logger.info("Starting yaw-locked keyboard control")
        print("\n" + "=" * 55)
        print("  🧭 YAW-LOCK KEYBOARD CONTROL")
        print("  Forward (W) locks heading via IMU + PID")
        print("  Robot auto-corrects to drive perfectly straight")
        print("=" * 55 + "\n")

        full_cmd = [
            "docker", "exec", "-it",
            self.container, "bash", "-c",
            "source /root/catkin_ws/devel/setup.bash && "
            "python3 /root/catkin_ws/src/jetracer/scripts/keyboard_control_yaw_lock.py"
        ]

        proc = subprocess.Popen(
            full_cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        self.processes.append(proc)
        proc.wait()

        self.logger.info("Yaw-lock keyboard control ended")

    def run(self):
        """Main execution"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        print("\n" + "=" * 55)
        print("   🚗 JetRacer Control System v1.0 — YAW-LOCK MODE")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55 + "\n")

        # Pre-flight checks
        if not self.check_hardware():
            sys.exit(1)

        if not self.check_docker():
            sys.exit(1)

        # Sync scripts (including the new yaw-lock script)
        self.sync_scripts()

        # Start ROS components
        try:
            self.start_roscore()
            self.start_robot_base()

            print("⏳ Waiting for systems to initialize...")
            time.sleep(3)

            # Launch yaw-lock keyboard control
            self.start_yaw_lock_control()

        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.logger.exception(f"Error: {e}")
            print(f"\n❌ Error: {e}")
            self.cleanup()


if __name__ == "__main__":
    launcher = JetRacerYawLock()
    launcher.run()
