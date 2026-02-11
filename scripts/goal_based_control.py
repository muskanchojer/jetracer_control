#!/usr/bin/env python3
"""
Goal-Based Autonomous Control - Navigation + Vision fusion
Main entry point for goal-directed autonomous driving
Optimized for Jetson Orin Nano with decoupled vision and navigation loops
"""

import os
import sys
import time
import yaml
import logging
import signal
import math
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auto.vision import VisionProcessor, CameraStream, AsyncVisionProcessor
from auto.policy import PolicyNetwork, PolicyEvaluator
from auto.control_adapter import ControlAdapter, SafetyMonitor
from auto.navigation import WaypointNavigator
from auto.fusion_controller import FusionController, VisionCommand
from auto.lightweight_vision import LightweightVisionProcessor, AsyncLightweightVision


class GoalBasedController:
    """
    Goal-based autonomous controller with navigation + vision fusion
    Combines high-level waypoint navigation with low-level obstacle avoidance
    
    Optimized architecture:
    - High-frequency navigation loop (20-50 Hz) for responsive control
    - Async vision processing in separate thread
    - Non-blocking camera capture
    """
    
    def __init__(self, config_path=None):
        self.running = False
        self.setup_logging()
        
        # Load configurations
        self.load_config(config_path)
        
        # Initialize components
        use_tinygrad = self.config['inference']['use_tinygrad']
        vision_mode = self.config.get('vision', {}).get('mode', 'yolov8')
        
        self.logger.info("="*60)
        self.logger.info("Initializing Goal-Based Autonomous Controller")
        self.logger.info(f"Mode: Navigation + Vision Fusion (Optimized)")
        self.logger.info(f"Vision: {vision_mode} | Backend: {'tinygrad' if use_tinygrad else 'PyTorch'}")
        self.logger.info("="*60)
        
        # Vision system - choose based on config mode
        vision_skip_frames = self.config.get('inference', {}).get('vision_skip_frames', 3)
        
        if vision_mode == 'direct':
            # DirectPolicyNet - end-to-end, fastest option
            self.logger.info("🚀 Using DirectPolicyNet (fastest, end-to-end)")
            self.lightweight_vision = LightweightVisionProcessor(
                self.config, mode='direct', use_tinygrad=use_tinygrad
            )
            self.async_vision = AsyncLightweightVision(
                self.lightweight_vision, 
                skip_frames=vision_skip_frames
            )
            self.use_lightweight = True
        elif vision_mode == 'mobilenet':
            # MobileNetV3 + TinyNet - compatible with existing TinyNet weights
            self.logger.info("📱 Using MobileNetV3 + TinyNet (medium speed)")
            self.lightweight_vision = LightweightVisionProcessor(
                self.config, mode='mobilenet', use_tinygrad=False
            )
            self.policy = PolicyNetwork(self.config, use_tinygrad=use_tinygrad)
            self.async_vision = AsyncVisionProcessor(
                self.lightweight_vision.processor,
                self.policy,
                skip_frames=vision_skip_frames
            )
            self.use_lightweight = False
        else:
            # Original YOLOv8 + TinyNet
            self.logger.info("🔬 Using YOLOv8 + TinyNet (original)")
            self.vision = VisionProcessor(self.config, use_tinygrad=use_tinygrad)
            self.policy = PolicyNetwork(self.config, use_tinygrad=use_tinygrad)
            self.async_vision = AsyncVisionProcessor(
                self.vision, 
                self.policy, 
                skip_frames=vision_skip_frames
            )
            self.use_lightweight = False
        
        self.evaluator = PolicyEvaluator(self.config)
        
        # Navigation system for goal-directed movement
        self.navigator = WaypointNavigator(self.config)
        
        # Show initial position before setting goal
        time.sleep(1)  # Wait for odometry data
        x, y, theta = self.navigator.get_position()
        self.logger.info(f"📍 Starting position: ({x:.2f}, {y:.2f}), heading: {theta:.1f}°")
        
        # Fusion controller to blend navigation + vision
        self.fusion = FusionController(self.config)
        
        # Control adapter (ROS interface)
        self.control = ControlAdapter(self.config)
        self.safety = SafetyMonitor(self.config)
        
        # Camera stream (threaded capture)
        self.camera = CameraStream(
            device=self.robot_config['hardware']['camera_device'],
            width=self.config['vision']['camera_width'],
            height=self.config['vision']['camera_height']
        )
        
        # Display settings
        self.show_camera = True
        self.camera_window_name = "Goal-Based Navigation"
        
        # Control loop parameters
        nav_rate = self.config.get('navigation', {}).get('control_rate', 20)  # Hz
        self.nav_loop_delay = 1.0 / nav_rate
        
        # Latest vision result (thread-safe)
        self._vision_result_lock = threading.Lock()
        self._latest_vision_cmd = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        self.logger.info("Goal-based controller initialized successfully")
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"goal_based_{timestamp}.log"
        
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
        
        # Start ROS infrastructure
        self.logger.info("Ensuring ROS infrastructure is running...")
        self.start_ros_infrastructure()
        
        # Initialize ROS navigation interface
        self.logger.info("Initializing navigation ROS interface...")
        self.navigator.initialize_ros()
        
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
        
        # Start roscore
        self.logger.info("Starting roscore...")
        print("  Starting ROS core...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && roscore"],
            check=False
        )
        time.sleep(3)
        
        # Start robot base
        self.logger.info("Starting robot base...")
        print("  Starting robot base...")
        subprocess.run(
            ["docker", "exec", "-d", "jetracer_ros", "bash", "-c",
             "source /root/catkin_ws/devel/setup.bash && rosrun jetracer jetracer"],
            check=False
        )
        time.sleep(2)
        
        print("✅ ROS infrastructure started")
    
    def _draw_goal_indicator(self, frame, nav_cmd, current_heading):
        """
        Draw goal direction indicator on camera frame
        Shows an arrow pointing toward the goal
        """
        import cv2
        import numpy as np
        
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h - 80  # Bottom center
        
        # Calculate direction to goal relative to current heading
        heading_error = nav_cmd.target_heading - current_heading
        
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Draw compass circle
        radius = 50
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 2)
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Draw forward direction (up)
        cv2.line(frame, (center_x, center_y), (center_x, center_y - radius - 10),
                (255, 255, 255), 2)
        cv2.putText(frame, "FWD", (center_x - 20, center_y - radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw goal direction arrow
        arrow_length = radius - 5
        arrow_angle = -heading_error  # Negative because screen Y is inverted
        arrow_end_x = int(center_x + arrow_length * math.sin(arrow_angle))
        arrow_end_y = int(center_y - arrow_length * math.cos(arrow_angle))
        
        # Color based on distance
        if nav_cmd.distance_to_goal < 0.5:
            color = (0, 255, 0)  # Green - very close
        elif nav_cmd.distance_to_goal < 1.5:
            color = (0, 255, 255)  # Yellow - close
        else:
            color = (0, 165, 255)  # Orange - far
        
        # Draw thick arrow
        cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y),
                       color, 3, tipLength=0.3)
        
        # Distance text inside circle
        cv2.putText(frame, f"{nav_cmd.distance_to_goal:.1f}m",
                   (center_x - 25, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Angle text below circle
        angle_deg = math.degrees(heading_error)
        direction = "STRAIGHT" if abs(angle_deg) < 10 else ("LEFT" if angle_deg > 0 else "RIGHT")
        cv2.putText(frame, f"{direction} {abs(angle_deg):.0f}deg",
                   (center_x - 60, center_y + radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def set_goal(self, x: float, y: float):
        """Set navigation goal (x, y in meters from start)"""
        self.navigator.set_goal(x, y)
        self.logger.info(f"🎯 Goal set: ({x:.2f}, {y:.2f})")
        print(f"\n🎯 Goal set: ({x:.2f}m, {y:.2f}m)")
    
    def set_waypoints(self, waypoints):
        """Set multiple waypoints [(x1,y1), (x2,y2), ...]"""
        self.navigator.set_waypoints(waypoints)
        self.logger.info(f"🗺️ {len(waypoints)} waypoints set")
        print(f"\n🗺️ Loaded {len(waypoints)} waypoints")
    
    def run(self):
        """
        Main goal-based autonomous driving loop
        Optimized with decoupled high-frequency navigation and async vision
        """
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed")
            return False
        
        # Start camera (threaded capture)
        self.camera.start()
        
        # Start async vision processor
        self.async_vision.start()
        
        # Setup camera display
        if self.show_camera:
            import cv2
            cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.camera_window_name, 640, 480)
        
        self.running = True
        self.safety.start()
        
        # Get control rate from config (default 20 Hz for responsive control)
        nav_rate = self.config.get('navigation', {}).get('control_rate', 20)
        loop_delay = 1.0 / nav_rate
        
        # Display update rate (lower to reduce overhead)
        display_rate = self.config.get('inference', {}).get('display_rate', 10)
        display_interval = 1.0 / display_rate
        last_display_time = 0
        
        print("\n" + "="*60)
        print("🚀 Goal-Based Autonomous Navigation Active (Optimized)")
        print("="*60)
        print(f"\nControl rate: {nav_rate} Hz | Display rate: {display_rate} Hz")
        print("\nCommands:")
        print("  - Set goal during runtime: Not yet implemented (use code)")
        print("  - Press Ctrl+C to stop")
        print("\n")
        
        frame_count = 0
        last_status_time = time.time()
        last_frame = None
        last_vision_action = "straight"
        last_vision_confidence = 0.0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get camera frame (non-blocking, threaded capture)
                frame = self.camera.read()
                if frame is not None:
                    last_frame = frame
                    # Submit to async vision processor (non-blocking)
                    self.async_vision.submit_frame(frame)
                
                # Get latest vision result (non-blocking)
                vision_result = self.async_vision.get_result(max_age=0.5)
                vision_cmd = None
                
                if vision_result:
                    last_vision_action = vision_result['action']
                    last_vision_confidence = vision_result['confidence']
                    vision_cmd = self.fusion.vision_action_to_command(
                        vision_result['action'], 
                        vision_result['confidence']
                    )
                
                # Get navigation command (always runs at full rate)
                nav_cmd = self.navigator.get_navigation_command()
                
                # Get current heading
                _, _, current_heading = self.navigator.get_position()
                
                # Fuse commands (vision may be None - that's OK)
                fused_cmd = self.fusion.fuse_commands(nav_cmd, vision_cmd, current_heading)
                
                # Send to robot
                self.control.send_velocity_command(fused_cmd.linear_x, fused_cmd.angular_z)
                
                # Update simulated odometry if ROS not available
                if self.navigator.simulated_mode:
                    self.navigator.update_simulated_odometry(
                        fused_cmd.linear_x,
                        fused_cmd.angular_z,
                        loop_delay
                    )
                
                # Display update (at lower rate to reduce overhead)
                frame_count += 1
                now = time.time()
                if self.show_camera and last_frame is not None and (now - last_display_time) >= display_interval:
                    last_display_time = now
                    self._update_display(
                        last_frame, nav_cmd, fused_cmd, 
                        last_vision_action, last_vision_confidence, 
                        current_heading
                    )
                
                # Periodic status update (every 5 seconds)
                if now - last_status_time > 5.0:
                    status = self.navigator.get_status()
                    if status['has_goal']:
                        vision_stats = self.async_vision.get_stats()
                        self.logger.info(
                            f"Navigation: {status['distance_to_goal']:.2f}m to goal, "
                            f"{status['waypoints_remaining']} waypoints | "
                            f"Vision: {vision_stats['fps']:.1f} FPS"
                        )
                    last_status_time = now
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < loop_delay:
                    time.sleep(loop_delay - elapsed)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
            return False
        finally:
            self.cleanup()
        
        return True
    
    def _update_display(self, frame, nav_cmd, fused_cmd, vision_action, vision_confidence, current_heading):
        """Update the display window with current state (runs at reduced rate)"""
        import cv2
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw goal direction indicator
        if nav_cmd and not nav_cmd.at_goal:
            self._draw_goal_indicator(display_frame, nav_cmd, current_heading)
        
        # Vision info (top-left)
        cv2.putText(display_frame, f"Vision: {vision_action} ({vision_confidence:.2f})",
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Navigation info
        if nav_cmd and not nav_cmd.at_goal:
            cv2.putText(display_frame, f"Goal: {nav_cmd.distance_to_goal:.2f}m",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Show goal coordinates
            goal_status = self.navigator.get_status()
            if 'current_goal' in goal_status:
                gx, gy = goal_status['current_goal']
                cv2.putText(display_frame, f"Target: ({gx:.1f}, {gy:.1f})",
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        elif nav_cmd and nav_cmd.at_goal:
            cv2.putText(display_frame, "Goal: REACHED!",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Goal: None (Vision-only mode)",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Fusion mode
        mode_color = (255, 255, 0) if fused_cmd.mode == 'blended' else (0, 255, 255)
        cv2.putText(display_frame, f"Mode: {fused_cmd.mode}",
                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Speed and turn info
        cv2.putText(display_frame, f"Speed: {fused_cmd.linear_x:.2f} m/s | Turn: {fused_cmd.angular_z:.2f}",
                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Vision processing stats
        vision_stats = self.async_vision.get_stats()
        cv2.putText(display_frame, f"Vision FPS: {vision_stats['fps']:.1f}",
                  (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Position (bottom-left)
        x, y, theta = self.navigator.get_position()
        cv2.putText(display_frame, f"Pos: ({x:.2f}, {y:.2f}) theta: {math.degrees(theta):.0f}deg",
                  (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.camera_window_name, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.logger.info("User pressed 'q' to quit")
            self.running = False
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        self.running = False
        self.control.stop()
        
        # Stop async vision processor
        self.async_vision.stop()
        
        # Stop camera (threaded)
        self.camera.stop()
        
        if self.show_camera:
            import cv2
            cv2.destroyAllWindows()
        
        self.logger.info("Cleanup complete")
    
    def shutdown(self, signum, frame):
        """Signal handler for clean shutdown"""
        self.logger.info(f"Received signal {signum}")
        self.running = False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Goal-Based Autonomous Controller')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--goal', type=str, help='Goal coordinates as "x,y" in meters')
    parser.add_argument('--waypoints', type=str, help='Waypoints as "x1,y1;x2,y2;..." in meters')
    
    args = parser.parse_args()
    
    # Create controller
    controller = GoalBasedController(args.config)
    
    # Set goal or waypoints if provided
    if args.waypoints:
        waypoints = []
        for wp in args.waypoints.split(';'):
            x, y = map(float, wp.split(','))
            waypoints.append((x, y))
        controller.set_waypoints(waypoints)
    elif args.goal:
        x, y = map(float, args.goal.split(','))
        controller.set_goal(x, y)
    else:
        print("\n⚠️  No goal set! Robot will use vision-only mode.")
        print("Set a goal with: --goal x,y (e.g., --goal 2.0,1.5)")
        print("Or set waypoints with: --waypoints x1,y1;x2,y2 (e.g., --waypoints 1,0;2,1)")
        print("\nYou can also set goals programmatically in code.")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Run controller
    success = controller.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
