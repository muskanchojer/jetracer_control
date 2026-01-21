#!/usr/bin/env python3
"""
JetRacer Control System - Main Launcher
Simplified launcher with proper logging
"""

import subprocess
import time
import sys
import os
import signal
import logging
from datetime import datetime

class JetRacerLauncher:
    def __init__(self):
        self.processes = []
        self.container = "jetracer_ros"
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to file and console"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"jetracer_{timestamp}.log")
        
        # Configure logging
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
        self.logger.info("JetRacer Control System Started")
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
                
                # Copy to container
                result = subprocess.run(
                    ["docker", "cp", src, f"{self.container}:{dest_path}"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    # Make executable
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
            # Capture output to PIPE for logging
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
    
    def log_process_output(self, process, prefix="", silent=False):
        """Monitor and log process output in real-time
        
        Args:
            process: Process with stdout to monitor
            prefix: Prefix for log entries
            silent: If True, only log to file (no console output)
        """
        import threading
        
        def read_stdout():
            try:
                if process.stdout is None:
                    self.logger.warning(f"{prefix}Process has no stdout to capture")
                    return
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip()
                        # Always log to file
                        self.logger.info(f"{prefix}{line}")
                        # Only print to console if not silent
                        if not silent:
                            print(f"{prefix}{line}")
            except Exception as e:
                self.logger.error(f"{prefix}Error reading stdout: {e}")
        
        def read_stderr():
            try:
                if process.stderr is None:
                    return
                
                for line in iter(process.stderr.readline, ''):
                    if line:
                        line = line.rstrip()
                        # Always log to file
                        self.logger.info(f"{prefix}{line}")
                        # Only print to console if not silent
                        if not silent:
                            print(f"{prefix}{line}")
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
    
    def show_menu(self):
        """Show control mode selection menu"""
        print("\n" + "="*50)
        print("   🚗 JetRacer Control System")
        print("="*50)
        print("\nSelect Control Mode:")
        print("  1) Keyboard Control (WASD)")
        print("  2) Camera Monitor (Live feed only)")
        print("  3) IMU Monitor + Keyboard Control")
        print("  0) Exit")
        print("="*50)
        
        choice = input("\nEnter choice [1-3, 0 to exit]: ").strip()
        self.logger.info(f"User selected menu option: {choice}")
        return choice
    
    def start_roscore(self):
        """Start ROS core"""
        self.logger.info("Starting roscore...")
        print("  Starting ROS...")
        self.processes.append(self.run_in_docker("roscore"))
        time.sleep(2)
    
    def start_robot_base(self):
        """Start robot base node"""
        self.logger.info("Starting robot base...")
        print("  Starting robot...")
        self.processes.append(self.run_in_docker("rosrun jetracer jetracer"))
        time.sleep(2)
    
    def start_usb_camera(self):
        """Start USB camera node"""
        self.logger.info("Starting USB camera...")
        print("  Starting USB camera...")
        self.processes.append(self.run_in_docker("roslaunch jetracer usb_camera.launch"))
        time.sleep(3)
    
    def mode_keyboard(self):
        """Keyboard control mode"""
        self.logger.info("Starting Keyboard Control Mode")
        print("\n🎮 Starting Keyboard Control Mode\n")
        
        self.start_roscore()
        self.start_robot_base()
        
        print("\n" + "="*50)
        print("  ⌨️  KEYBOARD CONTROL ACTIVE")
        print("="*50 + "\n")
        
        self.logger.info("Launching keyboard control (interactive)")
        # Run keyboard control interactively
        full_cmd = [
            "docker", "exec", "-it",
            self.container, "bash", "-c",
            "source /root/catkin_ws/devel/setup.bash && "
            "python3 /root/catkin_ws/src/jetracer/scripts/keyboard_control.py"
        ]
        proc = subprocess.Popen(full_cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        self.processes.append(proc)
        proc.wait()
        
        self.logger.info("Keyboard control mode ended")
    
    def mode_camera_monitor(self):
        """Camera monitoring mode - native viewer outside Docker"""
        import threading
        import cv2
        
        self.logger.info("Starting Native Camera Monitor Mode")
        print("\n📷 Starting Camera Monitor Mode (Native)\n")
        print("  Camera runs OUTSIDE Docker container")
        print("  No ROS needed for display\n")
        
        # Start camera in separate thread
        camera_thread = threading.Thread(target=self.run_native_camera, daemon=False)
        camera_thread.start()
        
        print("\n" + "="*50)
        print("  📷 NATIVE CAMERA VIEWER")
        print("  Press 'q' to stop")
        print("="*50 + "\n")
        
        self.logger.info("Camera viewer running in native thread")
        camera_thread.join()
        
        self.logger.info("Camera monitor mode ended")
    
    def run_native_camera(self):
        """Run camera viewer natively outside Docker"""
        import cv2
        import numpy as np

        self.logger.info("Opening native camera at /dev/video0")

        def nothing(x):
            pass

        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.logger.error("Failed to open camera")
                print("❌ Failed to open camera at /dev/video0")
                return
            
            # Use a variable for the window name to prevent mismatch errors
            window_name = "JetRacer Camera (Native)"
            cv2.namedWindow(window_name)

            cv2.createTrackbar("Rotate", window_name, 0, 3, nothing)
            cv2.createTrackbar("Flip", window_name, 0, 1, nothing)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # MOVE THIS OUTSIDE THE LOOP
            frame_count = 0 
            
            self.logger.info("Camera opened successfully, starting capture")
            print("📸 Camera opened successfully\n")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                # Increment counter properly
                frame_count += 1
                
                rotate_val = cv2.getTrackbarPos("Rotate", window_name)
                flip_val = cv2.getTrackbarPos("Flip", window_name)
                
                # Apply Rotation
                if rotate_val == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotate_val == 2:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotate_val == 3:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Apply Flip
                if flip_val == 1:
                    frame = cv2.flip(frame, 1)
                
                # Add overlays
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
                
                # Log periodically
                if frame_count % 60 == 0:
                    self.logger.info(f"Camera: {frame_count} frames captured")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User pressed 'q', stopping camera")
                    print("\n🛑 Stopping camera...")
                    break
                    
        except Exception as e:
            self.logger.error(f"Camera error: {e}")
            print(f"❌ Camera error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info(f"Camera stopped. Total frames: {frame_count}")
            print(f"✅ Camera stopped. Total frames: {frame_count}")

    def mode_imu(self):
        """IMU monitoring mode with keyboard control"""
        self.logger.info("Starting IMU Monitor Mode")
        print("\n📡 Starting IMU Monitor Mode\n")
        
        self.start_roscore()
        self.start_robot_base()
        
        # Start IMU monitor - log silently
        self.logger.info("Starting IMU monitor")
        print("  Starting IMU monitor...")
        imu_proc = self.run_in_docker( 
            "python3 /root/catkin_ws/src/jetracer/scripts/imu_monitor.py",
            capture_output=True
        )
        self.processes.append(imu_proc)
        self.log_process_output(imu_proc, "[IMU] ", silent=True)
        time.sleep(2)
        
        # Start keyboard control - interactive (show in console)
        print("\n" + "="*50)
        print("  📡 IMU MONITOR + KEYBOARD CONTROL")
        print("  IMU data: logged to file (silent)")
        print("  Keyboard: interactive control below")
        print("="*50 + "\n")
        
        self.logger.info("Launching keyboard control (interactive)")
        # Run keyboard control interactively
        full_cmd = [
            "docker", "exec", "-it",
            self.container, "bash", "-c",
            "source /root/catkin_ws/devel/setup.bash && "
            "python3 /root/catkin_ws/src/jetracer/scripts/keyboard_control.py"
        ]
        kb_proc = subprocess.Popen(full_cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        self.processes.append(kb_proc)
        kb_proc.wait()
        
        self.logger.info("IMU monitor mode ended")
    
    def run(self):
        """Main execution loop"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Initial checks
        if not self.check_hardware() or not self.check_docker():
            self.logger.error("Startup checks failed")
            sys.exit(1)
        
        # Sync scripts
        self.sync_scripts()
        
        # Main menu loop
        while True:
            choice = self.show_menu()
            
            try:
                if choice == '1':
                    self.mode_keyboard()
                elif choice == '2':
                    self.mode_camera_monitor()
                elif choice == '3':
                    self.mode_imu()
                elif choice == '0':
                    self.logger.info("User exited")
                    print("\n👋 Goodbye!\n")
                    sys.exit(0)
                else:
                    self.logger.warning(f"Invalid choice: {choice}")
                    print("\n❌ Invalid choice\n")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                self.cleanup()
            except Exception as e:
                self.logger.exception(f"Error in mode execution: {e}")
                print(f"\n❌ Error: {e}\n")
                self.cleanup()

if __name__ == "__main__":
    launcher = JetRacerLauncher()
    launcher.run()