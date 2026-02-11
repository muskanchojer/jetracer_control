#!/usr/bin/env python3
"""
Homing Autonomous Control - Autonomous driving with return-to-start capability

This script extends autonomous control with:
1. Dead reckoning position tracking
2. Automatic return to starting point every 2 minutes (configurable)
3. Stuck detection with intelligent recovery
4. Command history for reverse playback

Usage:
    python scripts/homing_control.py [--return-interval 120] [--no-stuck-detection]
"""

import os
import sys
import time
import yaml
import logging
import signal
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auto.vision import VisionProcessor, CameraStream
from auto.policy import PolicyNetwork, PolicyEvaluator
from auto.control_adapter import ControlAdapter, SafetyMonitor
from auto.homing_controller import HomingController, HomingMode, ReversalPlanner


class HomingAutonomousController:
    """
    Autonomous controller with return-to-start and stuck recovery.
    
    This controller:
    1. Drives autonomously using vision
    2. Tracks position using dead reckoning
    3. Returns home every N minutes
    4. Detects when stuck and executes recovery maneuvers
    """
    
    def __init__(self, config_path=None, return_interval=120, enable_stuck_detection=True):
        self.running = False
        self.setup_logging()
        
        # Store command line overrides
        self.return_interval_override = return_interval
        self.stuck_detection_override = enable_stuck_detection
        
        # Load configurations
        self.load_config(config_path)
        
        # Initialize components
        use_tinygrad = self.config['inference']['use_tinygrad']
        
        self.logger.info("="*60)
        self.logger.info("Initializing Homing Autonomous Controller")
        self.logger.info(f"Mode: {'tinygrad' if use_tinygrad else 'PyTorch'}")
        self.logger.info(f"Return interval: {self.return_interval_override}s ({self.return_interval_override/60:.1f} min)")
        self.logger.info(f"Stuck detection: {'enabled' if self.stuck_detection_override else 'disabled'}")
        self.logger.info("="*60)
        
        # Core autonomous components
        self.vision = VisionProcessor(self.config, use_tinygrad=use_tinygrad)
        self.policy = PolicyNetwork(self.config, use_tinygrad=use_tinygrad)
        self.evaluator = PolicyEvaluator(self.config)
        self.control = ControlAdapter(self.config)
        self.safety = SafetyMonitor(self.config)
        
        # Homing components
        self._setup_homing_config()
        self.homing = HomingController(self.config)
        self.reversal_planner = ReversalPlanner(self.config)
        
        # Camera stream
        self.camera = CameraStream(
            device=self.robot_config['hardware']['camera_device'],
            width=self.config['vision']['camera_width'],
            height=self.config['vision']['camera_height']
        )
        
        # Display settings
        self.show_camera = True
        self.camera_window_name = "Homing Autonomous Navigation"
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'actions_executed': 0,
            'returns_completed': 0,
            'stuck_recoveries': 0,
            'total_distance': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        self.logger.info("Homing autonomous controller initialized")
    
    def _setup_homing_config(self):
        """Setup homing configuration, applying command line overrides"""
        if 'homing' not in self.config:
            self.config['homing'] = {}
        
        homing_cfg = self.config['homing']
        homing_cfg['return_interval_seconds'] = self.return_interval_override
        homing_cfg['home_tolerance'] = homing_cfg.get('home_tolerance', 0.3)
        homing_cfg['heading_tolerance'] = homing_cfg.get('heading_tolerance', 0.2)
        homing_cfg['return_speed'] = homing_cfg.get('return_speed', 0.3)
        homing_cfg['turn_speed'] = homing_cfg.get('turn_speed', 0.8)
        
        # Dead reckoning config
        if 'dead_reckoning' not in homing_cfg:
            homing_cfg['dead_reckoning'] = {}
        homing_cfg['dead_reckoning']['enabled'] = True
        homing_cfg['dead_reckoning']['wheel_base'] = 0.20
        
        # Stuck detection config
        if 'stuck_detection' not in homing_cfg:
            homing_cfg['stuck_detection'] = {}
        homing_cfg['stuck_detection']['enabled'] = self.stuck_detection_override
        homing_cfg['stuck_detection']['threshold_distance'] = 0.05
        homing_cfg['stuck_detection']['check_interval'] = 1.0
        homing_cfg['stuck_detection']['frames_threshold'] = 5
        homing_cfg['stuck_detection']['max_recovery_commands'] = 10
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"homing_autonomous_{timestamp}.log"
        
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
            self.logger.error("Docker container not running")
            return False
        print("✅ Docker container running")
        
        # Start ROS infrastructure
        self.start_ros_infrastructure()
        
        # Check camera
        camera_device = self.robot_config['hardware']['camera_device']
        if not os.path.exists(camera_device):
            self.logger.error(f"Camera not found at {camera_device}")
            return False
        print(f"✅ Camera found at {camera_device}")
        
        # Check robot
        robot_port = self.robot_config['hardware']['robot_base_port']
        if not os.path.exists(robot_port):
            self.logger.error(f"Robot not connected at {robot_port}")
            return False
        print(f"✅ Robot connected at {robot_port}")
        
        return True
    
    def start_ros_infrastructure(self):
        """Start roscore and robot_base if not running"""
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "exec", "jetracer_ros", "bash", "-c",
                 "source /root/catkin_ws/devel/setup.bash && rostopic list"],
                capture_output=True,
                timeout=3.0
            )
            if result.returncode == 0:
                print("✅ ROS master running")
                return
        except:
            pass
        
        # Start roscore
        print("  Starting ROS core...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && roscore"],
            check=False
        )
        time.sleep(3)
        
        # Start robot base
        print("  Starting robot base...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && rosrun jetracer jetracer"],
            check=False
        )
        time.sleep(2)
        print("✅ ROS infrastructure started")
    
    def execute_action_with_tracking(self, action: str, confidence: float) -> bool:
        """Execute action and record for dead reckoning"""
        linear, angular = self.control.action_to_velocity(action)
        
        # Record command in homing controller
        self.homing.record_command(linear, angular)
        
        # Execute the command
        success = self.control.execute_action(action, confidence)
        
        if success:
            self.stats['actions_executed'] += 1
        
        return success
    
    def execute_velocity_with_tracking(self, linear: float, angular: float, duration: float = None) -> bool:
        """Execute velocity command and record for dead reckoning"""
        # Record command in homing controller
        self.homing.record_command(linear, angular, duration)
        
        # Execute
        success = self.control.send_velocity_command(linear, angular)
        
        if success:
            self.stats['actions_executed'] += 1
        
        return success
    
    def handle_stuck_recovery(self) -> bool:
        """Handle stuck robot recovery"""
        self.logger.warning("🚨 Robot appears stuck! Starting recovery...")
        self.stats['stuck_recoveries'] += 1
        
        if not self.homing.start_stuck_recovery():
            # Not enough history, try simple reversal
            self.logger.info("Using simple reversal for recovery")
            # Back up
            self.control.send_velocity_command(-0.2, 0.0)
            time.sleep(1.0)
            # Turn
            self.control.send_velocity_command(0.1, 0.6)
            time.sleep(0.5)
            # Stop
            self.control.stop()
            return True
        
        # Execute recovery commands
        while self.running:
            cmd = self.homing.get_next_recovery_command()
            if cmd is None:
                break
            
            self.logger.info(f"Recovery: L={cmd.linear_velocity:.2f}, A={cmd.angular_velocity:.2f}, "
                           f"D={cmd.duration:.2f}s")
            
            self.control.send_velocity_command(cmd.linear_velocity, cmd.angular_velocity)
            time.sleep(cmd.duration)
        
        self.control.stop()
        return True
    
    def handle_return_home(self):
        """Handle return to home sequence"""
        self.logger.info("="*40)
        self.logger.info("🏠 RETURNING TO HOME POSITION")
        self.logger.info("="*40)
        
        self.homing.start_return_home()
        
        # Return home loop
        while self.running and self.homing.get_mode() == HomingMode.RETURNING_HOME:
            result = self.homing.get_return_home_command()
            if result is None:
                break
            
            linear, angular = result
            
            # Check if we've arrived
            if linear == 0.0 and angular == 0.0:
                self.logger.info("🏠 Arrived at home!")
                self.stats['returns_completed'] += 1
                break
            
            # Execute movement
            self.execute_velocity_with_tracking(linear, angular)
            
            # Check for stuck during return
            if self.homing.is_stuck():
                self.handle_stuck_recovery()
            
            time.sleep(0.1)
        
        # Stop at home
        self.control.stop()
        
        # Brief pause at home
        self.logger.info("Pausing at home for 3 seconds...")
        time.sleep(3.0)
        
        # Resume exploration
        self.homing.resume_exploration()
        self.logger.info("🚀 Resuming exploration")
    
    def draw_status_overlay(self, frame, action, confidence, mode):
        """Draw status information on frame"""
        import cv2
        
        display = frame.copy()
        
        # Action and confidence
        action_text = f"Action: {action} ({confidence:.2f})"
        cv2.putText(display, action_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode
        mode_colors = {
            HomingMode.EXPLORING: (0, 255, 0),      # Green
            HomingMode.RETURNING_HOME: (0, 165, 255),  # Orange
            HomingMode.AT_HOME: (255, 0, 0),        # Blue
            HomingMode.STUCK_RECOVERY: (0, 0, 255)  # Red
        }
        mode_color = mode_colors.get(mode, (255, 255, 255))
        cv2.putText(display, f"Mode: {mode.value}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Position
        pos = self.homing.get_position()
        cv2.putText(display, f"Pos: ({pos[0]:.2f}, {pos[1]:.2f})", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Distance from home
        dist = self.homing.get_distance_from_home()
        cv2.putText(display, f"Dist: {dist:.2f}m", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Time until return
        time_left = self.homing.get_time_until_return()
        mins, secs = divmod(int(time_left), 60)
        cv2.putText(display, f"Return in: {mins}:{secs:02d}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        return display
    
    def run(self):
        """Main autonomous driving loop with homing"""
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed")
            return False
        
        # Start camera
        self.camera.start()
        
        # Setup display
        if self.show_camera:
            import cv2
            cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.camera_window_name, 640, 480)
        
        self.running = True
        self.safety.start()
        self.homing.start_session()
        
        prediction_rate = self.config['inference']['prediction_rate']
        prediction_interval = 1.0 / prediction_rate
        
        self.logger.info("="*60)
        self.logger.info("🚗 HOMING AUTONOMOUS DRIVING STARTED")
        self.logger.info(f"Prediction Rate: {prediction_rate} Hz")
        self.logger.info(f"Return Interval: {self.return_interval_override}s")
        self.logger.info("Press Ctrl+C or 'Q' to stop")
        self.logger.info("="*60)
        
        print("\n🤖 Homing autonomous mode active!")
        print(f"🏠 Will return home every {self.return_interval_override/60:.1f} minutes")
        print("📷 Camera window opened")
        print("Press 'H' to force return home, 'Q' to quit\n")
        
        last_prediction_time = 0
        last_action = "straight"
        last_confidence = 0.0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Check safety timeout
                if self.safety.check_timeout():
                    self.logger.warning("Safety timeout - stopping")
                    break
                
                # Check keyboard input
                if self.show_camera:
                    import cv2
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("'Q' pressed - stopping")
                        break
                    elif key == ord('h'):
                        self.logger.info("'H' pressed - forcing return home")
                        self.handle_return_home()
                        continue
                
                # Get current mode
                mode = self.homing.get_mode()
                
                # Handle different modes
                if mode == HomingMode.STUCK_RECOVERY:
                    # Continue recovery
                    cmd = self.homing.get_next_recovery_command()
                    if cmd:
                        self.control.send_velocity_command(cmd.linear_velocity, cmd.angular_velocity)
                        time.sleep(cmd.duration)
                    continue
                
                elif mode == HomingMode.RETURNING_HOME:
                    self.handle_return_home()
                    continue
                
                elif mode == HomingMode.AT_HOME:
                    # Just arrived home, resume exploration
                    self.homing.resume_exploration()
                
                # Exploration mode - normal autonomous driving
                
                # Check if time to return home
                if self.homing.check_should_return_home():
                    self.handle_return_home()
                    continue
                
                # Check if stuck
                if self.homing.is_stuck():
                    self.handle_stuck_recovery()
                    continue
                
                # Capture frame
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process at prediction rate
                time_since_last = loop_start - last_prediction_time
                if time_since_last >= prediction_interval:
                    # Vision processing
                    features = self.vision.process_frame(frame)
                    if features is None:
                        continue
                    
                    # Get policy prediction
                    action, confidence, probs = self.policy.predict(features)
                    last_action = action
                    last_confidence = confidence
                    last_prediction_time = loop_start
                    self.stats['frames_processed'] += 1
                    
                    # Execute action if confidence sufficient
                    if self.evaluator.should_execute(action, confidence):
                        self.execute_action_with_tracking(action, confidence)
                        self.safety.log_action(action)
                        status = "✓"
                    else:
                        status = "✗"
                    
                    # Log
                    pos = self.homing.get_position()
                    time_left = self.homing.get_time_until_return()
                    print(f"[{self.stats['frames_processed']:4d}] {status} {action:>8s} "
                          f"({confidence:.2f}) | Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}) | "
                          f"Return: {int(time_left)}s")
                
                # Update display
                if self.show_camera and frame is not None:
                    import cv2
                    display = self.draw_status_overlay(frame, last_action, last_confidence, mode)
                    cv2.imshow(self.camera_window_name, display)
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")
        
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
        
        finally:
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
        self.logger.info("Shutting down homing autonomous controller...")
        self.logger.info("="*60)
        
        # Stop robot
        self.control.stop()
        
        # Stop homing session
        self.homing.stop_session()
        
        # Stop camera
        self.camera.stop()
        
        # Print statistics
        self.safety.print_statistics()
        
        # Print homing stats
        homing_status = self.homing.get_status()
        print("\n📊 Homing Statistics:")
        print(f"  Total distance: {homing_status['total_distance_traveled']:.2f} m")
        print(f"  Returns completed: {self.stats['returns_completed']}")
        print(f"  Stuck recoveries: {self.stats['stuck_recoveries']}")
        print(f"  Final position: ({homing_status['position']['x']:.2f}, "
              f"{homing_status['position']['y']:.2f})")
        
        self.logger.info("Shutdown complete")
        print("\n✅ Homing autonomous mode stopped\n")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Homing Autonomous Controller - Returns to start every N minutes'
    )
    parser.add_argument(
        '--config', type=str, 
        help='Path to autonomous config file'
    )
    parser.add_argument(
        '--return-interval', type=int, default=120,
        help='Seconds between return-to-home cycles (default: 120 = 2 min)'
    )
    parser.add_argument(
        '--no-stuck-detection', action='store_true',
        help='Disable stuck detection and recovery'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode (no hardware)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Test mode not yet implemented")
        return
    
    controller = HomingAutonomousController(
        config_path=args.config,
        return_interval=args.return_interval,
        enable_stuck_detection=not args.no_stuck_detection
    )
    controller.run()


if __name__ == "__main__":
    main()
