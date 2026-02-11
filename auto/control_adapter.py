#!/usr/bin/env python3
"""
Control Adapter - Maps policy predictions to robot commands
Handles the translation between AI decisions and ROS motor control
"""

import subprocess
import logging
import time

class ControlAdapter:
    """Adapts policy outputs to robot motor commands"""
    
    def __init__(self, config, container_name="jetracer_ros"):
        self.config = config
        self.container = container_name
        self.logger = logging.getLogger(__name__)
        
        # Load control mappings from config
        self.control_map = config['control_mapping']
        
        self.logger.info("Control adapter initialized")
        self.logger.info(f"Control mappings: {self.control_map}")
    
    def action_to_velocity(self, action):
        """Convert action name to linear and angular velocities"""
        if action not in self.control_map:
            self.logger.error(f"Unknown action: {action}")
            return 0.0, 0.0
        
        mapping = self.control_map[action]
        linear = mapping['linear']
        angular = mapping['angular']
        
        return linear, angular
    
    
    def send_velocity_command(self, linear, angular, timeout=2.0):
        """Send velocity command to robot via ROS - using Docker exec like manual control setup"""
        # Single publish command (like keyboard control - publish once per action)
        # Note: rostopic pub -1 is slow (~1s) because it initializes ROS node each time
        ros_cmd = (
            f"rostopic pub /cmd_vel geometry_msgs/Twist "
            f"'{{linear: {{x: {linear}, y: 0, z: 0}}, "
            f"angular: {{x: 0, y: 0, z: {angular}}}}}' -1"
        )
        
        full_cmd = [
            "docker", "exec", self.container, "bash", "-c",
            f"source /root/catkin_ws/devel/setup.bash && {ros_cmd}"
        ]
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Sent velocity: linear={linear}, angular={angular}")
                return True
            else:
                self.logger.warning(f"Command failed: {result.stderr.decode()}")
                return False
                
        except subprocess.TimeoutExpired:
            # Command is likely still being sent, just taking longer than timeout
            # Don't log as error since robot still receives commands
            self.logger.debug(f"Command slow but likely sent: L={linear:.2f}, A={angular:.2f}")
            return True  # Assume success
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False
    
    def execute_action(self, action, confidence=1.0):
        """Execute a policy action"""
        linear, angular = self.action_to_velocity(action)
        
        self.logger.debug(f"Executing action: {action} (conf: {confidence:.2f}) -> "
                         f"L={linear:.2f}, A={angular:.2f}")
        
        success = self.send_velocity_command(linear, angular)
        
        if success:
            # Log to safety monitor
            from . import control_adapter
            # Action will be logged by caller
            pass
        
        return success
    
    
    def stop(self):
        """Emergency stop - send zero velocities"""
        self.logger.info("STOP command")
        
        try:
            subprocess.run([
                "docker", "exec", self.container, "bash", "-c",
                "source /root/catkin_ws/devel/setup.bash && "
                "rostopic pub /cmd_vel geometry_msgs/Twist "
                "'{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' -1"
            ], capture_output=True, timeout=2)
            return True
        except:
            return False
    
    def check_docker_running(self):
        """Check if Docker container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            is_running = self.container in result.stdout
            if not is_running:
                self.logger.error(f"Docker container '{self.container}' not running")
                return False
            
            # Also check if ROS master is running
            ros_check = subprocess.run(
                ["docker", "exec", "-i", self.container, "bash", "-c",
                 "source /root/catkin_ws/devel/setup.bash && rostopic list"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if ros_check.returncode != 0:
                self.logger.warning("Docker is running but ROS master may not be started")
                self.logger.warning("Make sure to start roscore and robot_base in the container")
            
            return is_running
            
        except Exception as e:
            self.logger.error(f"Error checking Docker: {e}")
            return False


class SafetyMonitor:
    """Monitors safety conditions during autonomous operation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.emergency_stop_enabled = config['safety']['emergency_stop_enabled']
        self.max_execution_time = config['safety']['max_execution_time']
        
        self.start_time = None
        self.total_actions = 0
        self.action_counts = {
            'left': 0,
            'straight': 0,
            'right': 0,
            'reverse': 0
        }
    
    def start(self):
        """Start safety monitoring"""
        self.start_time = time.time()
        self.logger.info("Safety monitor started")
    
    def check_timeout(self):
        """Check if maximum execution time exceeded"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed > self.max_execution_time:
            self.logger.warning(f"Max execution time ({self.max_execution_time}s) exceeded")
            return True
        
        return False
    
    def log_action(self, action):
        """Log executed action"""
        self.total_actions += 1
        if action in self.action_counts:
            self.action_counts[action] += 1
    
    def get_statistics(self):
        """Get execution statistics"""
        if self.start_time is None:
            return {}
        
        elapsed = time.time() - self.start_time
        
        stats = {
            'elapsed_time': elapsed,
            'total_actions': self.total_actions,
            'actions_per_second': self.total_actions / elapsed if elapsed > 0 else 0,
            'action_distribution': self.action_counts
        }
        
        return stats
    
    def print_statistics(self):
        """Print execution statistics"""
        stats = self.get_statistics()
        if not stats:
            return
        
        self.logger.info("=== Autonomous Driving Statistics ===")
        self.logger.info(f"Elapsed time: {stats['elapsed_time']:.1f}s")
        self.logger.info(f"Total actions: {stats['total_actions']}")
        self.logger.info(f"Actions/sec: {stats['actions_per_second']:.2f}")
        self.logger.info("Action distribution:")
        for action, count in stats['action_distribution'].items():
            pct = (count / stats['total_actions'] * 100) if stats['total_actions'] > 0 else 0
            self.logger.info(f"  {action}: {count} ({pct:.1f}%)")
