#!/usr/bin/env python3
"""
JetRacer Control System - Main4
Visual IMU Compass - Graphical or ASCII compass display
"""

import subprocess
import time
import sys
import os
import signal
import logging
from datetime import datetime

class JetRacerIMUVerbose:
    def __init__(self):
        self.processes = []
        self.container = "jetracer_ros"
        self.running = True
        self.x11_enabled = False
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to file and console"""
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"jetracer_imu_verbose_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("JetRacer Verbose IMU Monitor Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("="*70)
        
    def check_hardware(self):
        """Check if robot hardware is connected"""
        self.logger.info("Checking hardware...")
        
        if not os.path.exists("/dev/ttyACM0"):
            self.logger.error("Robot not connected at /dev/ttyACM0")
            print("❌ Robot not connected")
            return False
        self.logger.info("Robot connected at /dev/ttyACM0")
        print("✅ Robot connected")
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
    
    def enable_x11_for_docker(self):
        """Enable X11 forwarding for Docker to display GUI windows"""
        try:
            # Allow Docker to connect to X11 display
            result = subprocess.run(
                ["xhost", "+local:docker"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                self.logger.info("X11 access enabled for Docker")
                self.x11_enabled = True
                return True
            else:
                self.logger.warning(f"Failed to enable X11: {result.stderr}")
                return False
        except Exception as e:
            self.logger.warning(f"Could not enable X11 for Docker: {e}")
            return False
    
    def run_in_docker(self, cmd):
        """Run command inside Docker container"""
        self.logger.debug(f"Running in Docker: {cmd}")
        
        full_cmd = [
            "docker", "exec", "-i",
            self.container, "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {cmd}"
        ]
        
        return subprocess.Popen(full_cmd)
    
    def cleanup(self, signum=None, frame=None):
        """Cleanup and shutdown all processes"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Shutting down...")
        print("\n\n🛑 Shutting down...")
        
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
    def start_imu_compass_visual(self):
        """Start visual IMU compass - graphical display"""
        self.logger.info("Starting visual IMU compass...")
        print("  Starting visual IMU compass...")
        print("  (Opening compass window)\n")
        
        # Enable X11 access for Docker if not already done
        if not self.x11_enabled:
            print("  Setting up X11 display access for Docker...")
            self.enable_x11_for_docker()
        
        # Try to run visual compass inside Docker with X11 forwarding
        try:
            # Check if Docker container has OpenCV and X11 access
            check_cmd = [
                "docker", "exec", self.container, "bash", "-c",
                "python3 -c 'import cv2' 2>/dev/null && echo 'OK'"
            ]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=3)
            
            if "OK" in result.stdout:
                self.logger.info("OpenCV available in Docker, starting visual compass")
                
                # Run visual compass inside Docker with X11 forwarding
                # Docker needs access to host's X11 display
                compass_proc = self.run_in_docker(
                    "DISPLAY=:0 python3 /root/catkin_ws/src/jetracer/scripts/imu_compass_visual.py"
                )
                self.processes.append(compass_proc)
                
                # Wait a moment to see if it starts successfully
                time.sleep(1)
                if compass_proc.poll() is not None:
                    # Process died immediately, fall back
                    raise Exception("Visual compass failed to start (check X11 permissions)")
                
                return "visual"
            else:
                raise Exception("OpenCV not available in Docker")
                
        except Exception as e:
            self.logger.warning(f"Cannot run graphical compass: {e}")
            self.logger.info("Falling back to ASCII compass")
            print(f"  ⚠️  Visual compass unavailable: {e}")
            print(f"  ℹ️  Using ASCII compass instead\n")
            
            # Fallback to ASCII compass
            ascii_proc = self.run_in_docker(
                "python3 /root/catkin_ws/src/jetracer/scripts/imu_compass_ascii.py"
            )
            self.processes.append(ascii_proc)
            time.sleep(2)
            return "ascii"
    
    def run(self):
        """Main execution - verbose IMU monitoring"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Initial checks
        print("\n" + "="*70)
        print("  🚗 JetRacer Visual IMU Compass")
        print("  Graphical compass display with heading arrow")
        print("="*70 + "\n")
        
        if not self.check_hardware() or not self.check_docker():
            self.logger.error("Startup checks failed")
            sys.exit(1)
        
        # Sync scripts
        self.sync_scripts()
        
        try:
            # Start ROS infrastructure
            self.start_roscore()
            self.start_robot_base()
            
            print("\n" + "="*70)
            print("  ✅ COMPASS ACTIVE")
            print("="*70)
            
            # Start compass and check which mode
            compass_mode = self.start_imu_compass_visual()
            
            if compass_mode == "visual":
                print("  🧭 Graphical compass window opened showing:")
                print("     - Heading arrow (green) pointing your direction")
                print("     - Compass bearing in degrees (0-360°)")
                print("     - Cardinal direction (N, NE, E, SE, S, SW, W, NW)")
                print("     - Roll, Pitch, and acceleration data")
                print("  ⌨️  Press 'q' in compass window to stop")
            else:
                print("  🧭 ASCII compass running in terminal")
                print("     - Shows compass circle with N/S/E/W markers")
                print("     - Arrow points in your heading direction")
                print("     - Updates every few messages")
            
            print("\n  📌 ANSWER: YES, Yaw = Compass Heading!")
            print("     Yaw = rotation around vertical axis = where you're pointing")
            print("="*70 + "\n")
            
            # Wait for IMU process to finish
            for proc in self.processes:
                proc.wait()
            
            self.logger.info("IMU monitor ended")
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.logger.exception(f"Error in execution: {e}")
            print(f"\n❌ Error: {e}\n")
            self.cleanup()
        finally:
            self.cleanup()

if __name__ == "__main__":
    launcher = JetRacerIMUVerbose()
    launcher.run()
