#!/usr/bin/env python3
"""
JetRacer Steering Calibration — Launcher
Starts Docker ROS environment and runs the steering calibration tool.
"""

import subprocess
import time
import sys
import os
import signal


CONTAINER = "jetracer_ros"
SCRIPT    = "steering_calibration.py"
DOCKER_SCRIPTS = "/root/catkin_ws/src/jetracer/scripts"


def check_hardware():
    if not os.path.exists("/dev/ttyACM0"):
        print("❌ Robot not connected (/dev/ttyACM0 missing)")
        return False
    print("✅ Robot connected")
    return True


def check_docker():
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if CONTAINER in result.stdout:
        print(f"✅ Docker container '{CONTAINER}' running")
        return True
    print(f"❌ Docker container '{CONTAINER}' not found")
    print("   Start it with:  ~/start_jetracer_docker.sh")
    return False


def sync_scripts():
    """Copy all .py scripts into the Docker container."""
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if not os.path.isdir(script_dir):
        print("⚠️  No scripts/ directory found")
        return 0

    count = 0
    for name in sorted(os.listdir(script_dir)):
        if not name.endswith(".py"):
            continue
        src  = os.path.join(script_dir, name)
        dest = f"{CONTAINER}:{DOCKER_SCRIPTS}/{name}"
        cp = subprocess.run(["docker", "cp", src, dest], capture_output=True)
        if cp.returncode == 0:
            subprocess.run(
                ["docker", "exec", CONTAINER, "chmod", "+x", f"{DOCKER_SCRIPTS}/{name}"],
                capture_output=True,
            )
            count += 1
    print(f"✅ Synced {count} script(s) into container")
    return count


def docker_exec(cmd, interactive=False):
    """Run a command inside the Docker container."""
    base = ["docker", "exec", "-it" if interactive else "-i", CONTAINER, "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {cmd}"]
    if interactive:
        return subprocess.Popen(base, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    return subprocess.Popen(base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Launcher:
    def __init__(self):
        self.procs = []

    def start(self, cmd, interactive=False):
        p = docker_exec(cmd, interactive)
        self.procs.append(p)
        return p

    def stop_robot(self):
        """Publish zero velocity so the robot doesn't keep moving."""
        try:
            subprocess.run(
                ["docker", "exec", CONTAINER, "bash", "-c",
                 "source /root/catkin_ws/devel/setup.bash && "
                 "rostopic pub /cmd_vel geometry_msgs/Twist "
                 "'{linear: {x: 0}, angular: {z: 0}}' -1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
            )
        except Exception:
            pass

    def cleanup(self, *_):
        print("\n🛑 Shutting down...")
        self.stop_robot()
        for p in self.procs:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("✅ Stopped\n")
        sys.exit(0)


def main():
    launcher = Launcher()
    signal.signal(signal.SIGINT,  launcher.cleanup)
    signal.signal(signal.SIGTERM, launcher.cleanup)

    print("\n" + "=" * 50)
    print("  🚗 JetRacer — Steering Calibration")
    print("=" * 50 + "\n")

    if not check_hardware() or not check_docker():
        sys.exit(1)

    sync_scripts()

    # --- roscore ---
    print("\n⏳ Starting roscore...")
    launcher.start("roscore")
    time.sleep(2)
    print("✅ roscore up")

    # --- robot base ---
    print("⏳ Starting robot base driver...")
    launcher.start("rosrun jetracer jetracer")
    time.sleep(2)
    print("✅ Robot base up")

    # --- calibration tool ---
    print("⏳ Launching steering calibration...\n")
    time.sleep(1)
    proc = launcher.start(f"python3 {DOCKER_SCRIPTS}/{SCRIPT}", interactive=True)
    proc.wait()

    launcher.cleanup()


if __name__ == "__main__":
    main()
