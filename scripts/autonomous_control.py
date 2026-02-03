#!/usr/bin/env python3
"""
Autonomous Control Script - Main autonomous driving loop
Integrates vision, policy, and control for autonomous navigation
"""

import os
import sys
import time
import yaml
import logging
import signal
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auto.vision import VisionProcessor, CameraStream
from auto.policy import PolicyNetwork, PolicyEvaluator
from auto.control_adapter import ControlAdapter, SafetyMonitor


class AutonomousController:
    """Main autonomous driving controller"""
    
    def __init__(self, config_path=None):
        self.running = False
        self.setup_logging()
        
        # Load configurations
        self.load_config(config_path)
        
        # Initialize components
        use_tinygrad = self.config['inference']['use_tinygrad']
        
        self.logger.info("="*60)
        self.logger.info("Initializing Autonomous Controller")
        self.logger.info(f"Mode: {'tinygrad' if use_tinygrad else 'PyTorch (simple)'}")
        self.logger.info("="*60)
        
        self.vision = VisionProcessor(self.config, use_tinygrad=use_tinygrad)
        self.policy = PolicyNetwork(self.config, use_tinygrad=use_tinygrad)
        self.evaluator = PolicyEvaluator(self.config)
        self.control = ControlAdapter(self.config)
        self.safety = SafetyMonitor(self.config)
        
        # Camera stream
        self.camera = CameraStream(
            device=self.robot_config['hardware']['camera_device'],
            width=self.config['vision']['camera_width'],
            height=self.config['vision']['camera_height']
        )
        
        # Camera display window
        self.show_camera = True
        self.camera_window_name = "Autonomous Navigation"
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        self.logger.info("Autonomous controller initialized successfully")
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"autonomous_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Log file: {log_file}")
    
    def load_config(self, config_path):
        """Load configuration files"""
        config_dir = Path(__file__).parent.parent / "config"
        
        # Load robot config
        robot_config_path = config_dir / "robot_config.yaml"
        with open(robot_config_path, 'r') as f:
            self.robot_config = yaml.safe_load(f)
        
        # Load autonomous config
        if config_path is None:
            config_path = config_dir / "autonomous_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded config from: {config_path}")
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        self.logger.info("Checking prerequisites...")
        
        # Check Docker
        if not self.control.check_docker_running():
            self.logger.error("Docker container not running. Start with: ~/start_jetracer_docker.sh")
            return False
        
        print("✅ Docker container running")
        
        # Start ROS infrastructure if not already running
        self.logger.info("Ensuring ROS infrastructure is running...")
        self.start_ros_infrastructure()
        
        # Check camera
        camera_device = self.robot_config['hardware']['camera_device']
        if not os.path.exists(camera_device):
            self.logger.error(f"Camera not found at {camera_device}")
            print(f"❌ Camera not found at {camera_device}")
            return False
        
        print(f"✅ Camera found at {camera_device}")
        
        # Check hardware
        robot_port = self.robot_config['hardware']['robot_base_port']
        if not os.path.exists(robot_port):
            self.logger.error(f"Robot not connected at {robot_port}")
            print(f"❌ Robot not connected at {robot_port}")
            return False
        
        print(f"✅ Robot connected at {robot_port}")
        
        return True
    
    def start_ros_infrastructure(self):
        """Start roscore and robot_base if not already running"""
        import subprocess
        
        # Check if roscore is running
        try:
            result = subprocess.run(
                ["docker", "exec", "jetracer_ros", "bash", "-c",
                 "source /root/catkin_ws/devel/setup.bash && rostopic list"],
                capture_output=True,
                timeout=3.0
            )
            
            if result.returncode == 0:
                self.logger.info("ROS master already running")
                print("✅ ROS master running")
                return
        except:
            pass
        
        # Start roscore in detached mode
        self.logger.info("Starting roscore...")
        print("  Starting ROS core...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && roscore"],
            check=False
        )
        time.sleep(3)
        
        # Start robot base in detached mode
        self.logger.info("Starting robot base...")
        print("  Starting robot base...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && rosrun jetracer jetracer"],
            check=False
        )
        time.sleep(2)
        
        print("✅ ROS infrastructure started")
    
    def run(self):
        """Main autonomous driving loop"""
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed")
            return False
        
        # Start camera
        self.camera.start()
        
        # Setup camera display window
        if self.show_camera:
            import cv2
            cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.camera_window_name, 640, 480)
        
        self.running = True
        self.safety.start()
        
        prediction_rate = self.config['inference']['prediction_rate']
        
        self.logger.info("="*60)
        self.logger.info("🚗 AUTONOMOUS DRIVING STARTED")
        self.logger.info(f"Prediction Rate: {prediction_rate} Hz")
        self.logger.info(f"Press Ctrl+C to stop")
        self.logger.info("="*60)
        
        if self.show_camera:
            print("\n🤖 Autonomous mode active!")
            print("📷 Camera window opened")
            print("Press Ctrl+C or 'Q' in camera window to stop\n")
        else:
            print("\n🤖 Autonomous mode active!")
            print("Press Ctrl+C to stop\n")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            # Track last prediction time
            last_prediction_time = 0
            prediction_rate = self.config.get('inference', {}).get('prediction_rate', 1)
            prediction_interval = 1.0 / prediction_rate
            
            self.logger.info(f"Prediction rate: {prediction_rate} Hz (1 prediction every {prediction_interval:.1f}s)")
            
            while self.running:
                loop_start = time.time()
                
                # Check safety timeout
                if self.safety.check_timeout():
                    self.logger.warning("Safety timeout - stopping")
                    break
                
                # Check for 'q' key to quit (non-blocking)
                if self.show_camera:
                    import cv2
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("'Q' key pressed - stopping")
                        break
                
                # Capture frame for display
                frame = self.camera.read()
                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame and predict action at specified rate
                time_since_last = loop_start - last_prediction_time
                if time_since_last >= prediction_interval:
                    # Process vision
                    features = self.vision.process_frame(frame)
                    if features is None:
                        self.logger.warning("Failed to process frame")
                        continue
                    
                    # Get prediction
                    action, confidence, probs = self.policy.predict(features)
                    last_prediction_time = loop_start
                    
                    # Update camera display
                    if self.show_camera and frame is not None:
                        import cv2
                        display_frame = frame.copy()
                        
                        action_text = f"Action: {action} ({confidence:.2f})"
                        cv2.putText(display_frame, action_text, 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(display_frame, f"Prediction: {prediction_rate} Hz", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow(self.camera_window_name, display_frame)
                    
                    # Execute action if confidence is high enough (like keyboard sends command on key press)
                    if self.evaluator.should_execute(action, confidence):
                        success = self.control.execute_action(action, confidence)
                        self.safety.log_action(action)
                        frame_count += 1
                        status = "✓" if success else "✗"
                        print(f"[{frame_count:4d}] {status} {action:>8s} "
                              f"(conf: {confidence:.2f})")
                    else:
                        frame_count += 1
                        print(f"[{frame_count:4d}] ✗ {action:>8s} "
                              f"(conf: {confidence:.2f}) | Low confidence")
                
                # Small delay
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        except Exception as e:
            self.logger.error(f"Error in autonomous loop: {e}", exc_info=True)
        
        finally:
            # Close camera window
            if self.show_camera:
                import cv2
                cv2.destroyAllWindows()
            
            self.shutdown()
        
        return True
    
    def shutdown(self, signum=None, frame=None):
        """Clean shutdown"""
        if not self.running:
            return
        
        self.running = False
        
        self.logger.info("="*60)
        self.logger.info("Shutting down autonomous controller...")
        self.logger.info("="*60)
        
        # Stop robot
        self.control.stop()
        
        # Stop camera
        self.camera.stop()
        
        # Print statistics
        self.safety.print_statistics()
        
        self.logger.info("Shutdown complete")
        print("\n✅ Autonomous mode stopped\n")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Robot Controller')
    parser.add_argument('--config', type=str, help='Path to autonomous config file')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no hardware)')
    
    args = parser.parse_args()
    
    if args.test:
        print("Test mode not yet implemented")
        return
    
    controller = AutonomousController(config_path=args.config)
    controller.run()


if __name__ == "__main__":
    main()
