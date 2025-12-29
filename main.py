#!/usr/bin/env python3
#run main.py and start docker container in another terminal to control robot using keyboard only
"""
JetRacer Control System - Main Launcher
Manages all ROS nodes in a single interface
"""

import subprocess
import time
import sys
import signal
import os
from datetime import datetime

class JetRacerLauncher:
    def __init__(self):
        self.processes = []
        self.docker_container = "jetracer_ros"
        
    def check_hardware(self):
        """Check if robot hardware is connected"""
        print("🔍 Checking hardware connections...")
        
        devices = {
            "/dev/ttyACM0":  "Robot Base (Motors/IMU)",
            "/dev/ttyACM1": "LIDAR",
        }
        
        all_ok = True
        for device, name in devices.items():
            if os.path.exists(device):
                print(f"  ✅ {name}:  {device}")
            else:
                print(f"  ⚠️  {name}: NOT FOUND ({device})")
                if device == "/dev/ttyACM0":
                    all_ok = False
        
        if not all_ok:
            print("\n❌ Robot base not connected!  Cannot continue.")
            return False
        
        print("✅ Hardware check passed!\n")
        return True
    
    def check_docker(self):
        """Check if Docker container is running"""
        print("🐳 Checking Docker container...")
        
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={self.docker_container}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if self.docker_container in result.stdout:
            print(f"✅ Docker container '{self.docker_container}' is running\n")
            return True
        else:
            print(f"❌ Docker container '{self. docker_container}' not found!")
            print("   Please start it with: ~/start_jetracer_docker. sh\n")
            return False
        
    def sync_scripts(self):
        """Sync scripts from repo to Docker workspace to make the code portable everywhere"""
        print("📁 Syncing scripts to Docker workspace...")
        
        repo_scripts = os.path.join(os.path.dirname(__file__), "scripts")
        
        if not os.path.exists(repo_scripts):
            print("  ⚠️  No scripts folder found in repo")
            return
        
        # Copy all Python scripts to Docker workspace
        for script in os.listdir(repo_scripts):
            if script.endswith(". py"):
                src = os.path.join(repo_scripts, script)
                
                # Copy to Docker container
                subprocess.run([
                    "docker", "cp", src,
                    f"{self.docker_container}:/root/catkin_ws/src/jetracer/scripts/{script}"
                ], check=True)
                
                # Make executable
                subprocess.run([
                    "docker", "exec", self.docker_container,
                    "chmod", "+x", f"/root/catkin_ws/src/jetracer/scripts/{script}"
                ], check=True)               
                print(f"  ✅ Synced {script}")   
        print()

    def run_in_docker(self, command, name, show_output=False):
        """Run a command inside Docker container"""
        full_cmd = [
            "docker", "exec", "-i", self.docker_container,
            "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {command}"
        ]
        
        if show_output:
            # Interactive process (like keyboard control)
            process = subprocess.Popen(
                full_cmd,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        else:
            # Background process with log file
            log_file = open(f"logs/{name}.log", "w")
            process = subprocess. Popen(
                full_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        
        self.processes.append((name, process))
        return process
    
    def start_roscore(self):
        """Start ROS master"""
        print("🚀 Starting ROS Master (roscore)...")
        self.run_in_docker("roscore", "roscore", show_output=False)
        time.sleep(2)  # Wait for roscore to initialize
        print("✅ ROS Master started\n")
    
    def start_robot_base(self):
        """Start robot base controller"""
        print("🤖 Starting Robot Base Controller...")
        self.run_in_docker("rosrun jetracer jetracer", "jetracer", show_output=False)
        time.sleep(2)  # Wait for robot to initialize
        print("✅ Robot Base started\n")
    
    def start_keyboard_control(self):
        """Start keyboard control"""
        print("⌨️  Starting Keyboard Control...")
        print("  CONTROLS READY")
        print("="*50)
        print()
        #self.run_in_docker("rosrun jetracer keyboard_control. py", "keyboard", show_output=True)
            # Use python3 directly instead of rosrun
        full_cmd = [
            "docker", "exec", "-it", self.docker_container,
            "bash", "-c",
            "source /root/catkin_ws/devel/setup.bash && cd /root/catkin_ws/src/jetracer/scripts && python3 keyboard_control.py"
        ]
    
        process = subprocess.Popen(
            full_cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    
        self.processes.append(("keyboard", process))
     # Wait for keyboard process (it blocks here)
        process.wait()

    def cleanup(self, signum=None, frame=None):
        """Stop all processes"""
        print("\n\n🛑 Shutting down JetRacer Control System...")
        
        # Send stop command to robot
        print("  Stopping robot...")
        subprocess.run(
            ["docker", "exec", self.docker_container, "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && "
             "rostopic pub /cmd_vel geometry_msgs/Twist "
             "'{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' -1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
        
        # Kill all processes
        for name, process in self.processes:
            print(f"  Stopping {name}...")
            process. terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All systems stopped.  Goodbye!\n")
        sys.exit(0)
    
    def load_config(self):
        """Load configuration file"""
        import yaml
        config_file = os.path.join(os.path.dirname(__file__), "config", "robot_config. yaml")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            print("⚠️  Config file not found, using defaults")
            return {'features': {'keyboard_control': True}}
    
    def start_camera_monitor(self):
        """Start camera monitoring"""
        print("📷 Starting Camera Monitor...")
        self.run_in_docker("roslaunch jetracer usb_camera.launch", "camera_launch", show_output=False)
        time.sleep(2)
        self.run_in_docker("rosrun jetracer camera.py", "camera_monitor", show_output=False)
        print("✅ Camera Monitor started\n")
    
    def start_camera_autonomous(self):
        """Start autonomous driving with camera"""
        print("🤖 Starting Autonomous Camera Control...")
        self.run_in_docker("roslaunch jetracer usb_camera.launch", "camera_launch", show_output=False)
        time.sleep(2)
        self.run_in_docker("rosrun jetracer camera.py --autonomous", "camera_auto", show_output=True)
        print("✅ Autonomous mode started\n")
    
    def start_imu_monitor(self):
        """Start IMU monitoring"""
        print("📡 Starting IMU Monitor...")
        self.run_in_docker("rosrun jetracer imu_monitor. py", "imu_monitor", show_output=False)
        print("✅ IMU Monitor started\n")

    def run(self):
        """Main execution"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Print banner
        print("\n" + "="*50)
        print("   🚗 JetRacer Control System v1.0")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        # Load configuration
        config = self.load_config()
        features = config.get('features', {})

        # Pre-flight checks
        if not self.check_hardware():
            sys.exit(1)
        
        if not self.check_docker():
            sys.exit(1)
        
        # Sync scripts to Docker workspace
        self.sync_scripts()

        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Start components in order
        try:
            self.start_roscore()
            self.start_robot_base()
            
            # Start optional features based on config
            if features.get('imu_monitor', False):
                self.start_imu_monitor()
            
            if features.get('camera_monitor', False):
                self. start_camera_monitor()

            # Wait a bit to ensure everything is ready
            print("⏳ Waiting for all systems to initialize...")
            time.sleep(3)
            
            # Start primary control interface
            if features.get('camera_autonomous', False):
                self.start_camera_autonomous()
            elif features.get('keyboard_control', True):
                self.start_keyboard_control()
            else:
                print("⚠️  No control interface enabled!")
                print("   Keeping robot alive...  Press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            
        except Exception as e:
            print(f"\n❌ Error:  {e}")
            self.cleanup()

if __name__ == "__main__":
    launcher = JetRacerLauncher()
    launcher.run()