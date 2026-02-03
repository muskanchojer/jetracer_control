#!/usr/bin/env python3
"""
JetRacer Control System - Complete Integration
Runs IMU logging, keyboard control, and native camera window together
"""

import subprocess
import time
import sys
import os
import signal
import logging
import threading
import cv2
from datetime import datetime

class JetRacerIntegrated:
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
        log_file = os.path.join(log_dir, f"jetracer_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*60)
        self.logger.info("JetRacer Integrated System Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("="*60)
        
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
    
    def run_in_docker(self, cmd, capture_output=False):
        """Run command inside Docker container
        
        Args:
            cmd: Command to run
            capture_output: If True, capture stdout/stderr for logging
        """
        self.logger.debug(f"Running in Docker: {cmd}")
        
        full_cmd = [
            "docker", "exec", "-i",
            self.container, "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {cmd}"
        ]
        
        if capture_output:
            # Capture output to PIPE for logging (redirect stderr to stdout for capture)
            return subprocess.Popen(
                full_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True, 
                bufsize=1
            )
        else:
            # No output capture - silent mode
            return subprocess.Popen(
                full_cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
    
    def log_process_output(self, process, prefix=""):
        """Monitor and log process output in real-time (file only, not console)"""
        def read_stdout():
            try:
                if process.stdout is None:
                    self.logger.warning(f"{prefix}Process has no stdout to capture")
                    return
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip()
                        # Log to file only (not console)
                        self.logger.info(f"{prefix}{line}")
            except Exception as e:
                self.logger.error(f"{prefix}Error reading stdout: {e}")
        
        def read_stderr():
            try:
                if process.stderr is None:
                    return
                
                for line in iter(process.stderr.readline, ''):
                    if line:
                        line = line.rstrip()
                        # Log to file only (not console)
                        self.logger.info(f"{prefix}{line}")
            except Exception as e:
                self.logger.error(f"{prefix}Error reading stderr: {e}")
        
        # Start both threads
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return (stdout_thread, stderr_thread)
    
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
    
    def start_imu_monitor(self):
        """Start IMU monitor with output logging"""
        self.logger.info("Starting IMU monitor with logging...")
        print("  Starting IMU monitor (logging to file)...")
        
        # Run with output capture enabled
        imu_proc = self.run_in_docker(
            "python3 /root/catkin_ws/src/jetracer/scripts/imu_monitor.py",
            capture_output=True
        )
        self.processes.append(imu_proc)
        
        # Start logging thread
        self.log_process_output(imu_proc, "[IMU] ")
        time.sleep(2)
    
    def run_keyboard_control(self):
        """Run keyboard control in interactive mode"""
        self.logger.info("Starting keyboard control...")
        print("\n" + "="*60)
        print("  ⌨️  KEYBOARD CONTROL ACTIVE")
        print("="*60 + "\n")
        
        # Interactive mode - attach to terminal
        full_cmd = [
            "docker", "exec", "-it",
            self.container, "bash", "-c",
            "source /root/catkin_ws/devel/setup.bash && "
            "python3 /root/catkin_ws/src/jetracer/scripts/keyboard_control.py"
        ]
        
        kb_proc = subprocess.Popen(
            full_cmd, 
            stdin=sys.stdin, 
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        self.processes.append(kb_proc)
        return kb_proc
    
    def run_camera_window(self):
        """Run native camera window in separate thread"""
        def camera_thread():
            self.logger.info("Starting native camera window...")
            
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    self.logger.error("Failed to open camera")
                    print("❌ Failed to open camera at /dev/video0")
                    return
                
                window_name = "JetRacer Camera"
                cv2.namedWindow(window_name)
                
                def nothing(x):
                    pass
                
                cv2.createTrackbar("Rotate", window_name, 0, 3, nothing)
                cv2.createTrackbar("Flip", window_name, 0, 1, nothing)
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                frame_count = 0
                self.logger.info("Camera window opened successfully")
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame")
                        break
                    
                    frame_count += 1
                    
                    rotate_val = cv2.getTrackbarPos("Rotate", window_name)
                    flip_val = cv2.getTrackbarPos("Flip", window_name)
                    
                    # Apply rotation
                    if rotate_val == 1:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotate_val == 2:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotate_val == 3:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Apply flip
                    if flip_val == 1:
                        frame = cv2.flip(frame, 1)
                    
                    # Add overlays
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' in terminal to quit", (10, frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, frame)
                    
                    # Log periodically
                    if frame_count % 300 == 0:
                        self.logger.info(f"Camera: {frame_count} frames captured")
                    
                    # Check for window close or 'q' key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                self.logger.info(f"Camera stopped. Total frames: {frame_count}")
                
            except Exception as e:
                self.logger.error(f"Camera error: {e}")
                print(f"❌ Camera error: {e}")
        
        # Start camera in separate daemon thread
        thread = threading.Thread(target=camera_thread, daemon=True)
        thread.start()
        self.logger.info("Camera thread started")
        return thread
    
    def start_keyboard_control(self):
        """Start keyboard control only (for main.py integration)"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        try:
            # Start ROS infrastructure
            self.start_roscore()
            self.start_robot_base()
            
            print("\n" + "="*60)
            print("  ✅ ROS SYSTEMS READY")
            print("="*60 + "\n")
            
            time.sleep(1)
            
            # Start keyboard control (blocking, interactive)
            kb_proc = self.run_keyboard_control()
            kb_proc.wait()
            
            self.logger.info("Keyboard control ended")
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.logger.exception(f"Error in execution: {e}")
            print(f"\n❌ Error: {e}\n")
            self.cleanup()
        finally:
            self.cleanup()
    
    def run(self):
        """Main execution - run all components together"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Initial checks
        print("\n" + "="*60)
        print("  🚗 JetRacer Integrated System")
        print("  IMU Logging + Keyboard Control + Camera Window")
        print("="*60 + "\n")
        
        if not self.check_hardware() or not self.check_docker():
            self.logger.error("Startup checks failed")
            sys.exit(1)
        
        # Sync scripts
        self.sync_scripts()
        
        try:
            # Start ROS infrastructure
            self.start_roscore()
            self.start_robot_base()
            
            # Start IMU monitor (logs to file)
            self.start_imu_monitor()
            
            # Start camera window (separate thread)
            camera_thread = self.run_camera_window()
            
            print("\n" + "="*60)
            print("  ✅ ALL SYSTEMS RUNNING")
            print("="*60)
            print("  📡 IMU: Logging to file (check logs directory)")
            print("  📷 Camera: Separate window opened")
            print("  ⌨️  Keyboard: Ready for control below")
            print("="*60 + "\n")
            
            time.sleep(1)
            
            # Start keyboard control (blocking, interactive)
            kb_proc = self.run_keyboard_control()
            kb_proc.wait()
            
            self.logger.info("Keyboard control ended")
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.logger.exception(f"Error in execution: {e}")
            print(f"\n❌ Error: {e}\n")
            self.cleanup()
        finally:
            self.cleanup()

if __name__ == "__main__":
    launcher = JetRacerIntegrated()
    launcher.run()
