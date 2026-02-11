#!/usr/bin/env python3
"""
Homing Controller - Autonomous return-to-start with dead reckoning and stuck recovery

Features:
1. Dead reckoning position tracking (independent of ROS odometry)
2. Command history for reverse playback when stuck
3. Automatic return-to-start every configurable interval (default: 2 minutes)
4. Stuck detection and intelligent recovery
"""

import math
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Deque
from enum import Enum


class HomingMode(Enum):
    """Operating modes for the homing controller"""
    EXPLORING = "exploring"      # Normal autonomous movement, tracking position
    RETURNING_HOME = "returning_home"  # Actively returning to start
    AT_HOME = "at_home"          # Reached starting position
    STUCK_RECOVERY = "stuck_recovery"  # Executing anti-stuck maneuvers


@dataclass
class MotionCommand:
    """Recorded motion command for dead reckoning and replay"""
    linear_velocity: float      # m/s
    angular_velocity: float     # rad/s
    duration: float             # seconds
    timestamp: float            # when command was issued
    
    def get_inverse(self) -> 'MotionCommand':
        """Get the inverse command (for reversing to escape stuck situations)"""
        return MotionCommand(
            linear_velocity=-self.linear_velocity,
            angular_velocity=-self.angular_velocity,
            duration=self.duration,
            timestamp=time.time()
        )


@dataclass
class DeadReckoningState:
    """Position state from dead reckoning"""
    x: float = 0.0              # meters from start
    y: float = 0.0              # meters from start
    theta: float = 0.0          # heading in radians
    total_distance: float = 0.0 # total distance traveled
    
    def copy(self) -> 'DeadReckoningState':
        return DeadReckoningState(
            x=self.x, y=self.y, theta=self.theta, 
            total_distance=self.total_distance
        )


@dataclass 
class StuckDetectorState:
    """State for stuck detection"""
    last_positions: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    last_check_time: float = 0.0
    stuck_count: int = 0
    consecutive_stuck_frames: int = 0


class HomingController:
    """
    Manages return-to-start functionality with dead reckoning and stuck recovery.
    
    This controller:
    1. Tracks robot position using dead reckoning (integrating commanded velocities)
    2. Records all motion commands for potential reverse playback
    3. Detects when robot is stuck and executes recovery maneuvers
    4. Returns robot to starting position every N minutes
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load homing configuration
        homing_config = config.get('homing', {})
        self.return_interval = homing_config.get('return_interval_seconds', 120)  # 2 minutes default
        self.home_tolerance = homing_config.get('home_tolerance', 0.3)  # meters
        self.heading_tolerance = homing_config.get('heading_tolerance', 0.2)  # radians
        
        # Dead reckoning parameters
        dr_config = homing_config.get('dead_reckoning', {})
        self.dr_enabled = dr_config.get('enabled', True)
        self.wheel_base = dr_config.get('wheel_base', 0.20)  # meters between wheels
        self.position_decay = dr_config.get('position_decay', 0.999)  # slight drift correction
        
        # Stuck detection parameters
        stuck_config = homing_config.get('stuck_detection', {})
        self.stuck_enabled = stuck_config.get('enabled', True)
        self.stuck_threshold_distance = stuck_config.get('threshold_distance', 0.05)  # meters
        self.stuck_check_interval = stuck_config.get('check_interval', 1.0)  # seconds
        self.stuck_frames_threshold = stuck_config.get('frames_threshold', 5)  # consecutive checks
        self.max_recovery_commands = stuck_config.get('max_recovery_commands', 10)
        
        # Speed parameters for return journey
        self.return_speed = homing_config.get('return_speed', 0.3)  # m/s
        self.turn_speed = homing_config.get('turn_speed', 0.8)  # rad/s
        
        # State
        self.mode = HomingMode.AT_HOME  # Start at home
        self.start_position = DeadReckoningState()  # Origin (0, 0, 0)
        self.current_position = DeadReckoningState()
        self.command_history: Deque[MotionCommand] = deque(maxlen=1000)  # Last ~1000 commands
        self.stuck_state = StuckDetectorState()
        
        # Timing
        self.session_start_time = time.time()
        self.last_exploration_start = time.time()
        self.last_return_home_time = time.time()
        self.last_command_time = time.time()
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        
        # Recovery state
        self.recovery_commands: List[MotionCommand] = []
        self.recovery_index = 0
        
        self.logger.info("HomingController initialized")
        self.logger.info(f"  Return interval: {self.return_interval}s ({self.return_interval/60:.1f} min)")
        self.logger.info(f"  Home tolerance: {self.home_tolerance}m")
        self.logger.info(f"  Stuck detection: {'enabled' if self.stuck_enabled else 'disabled'}")
    
    def start_session(self):
        """Start a new homing session - resets origin to current position"""
        with self.lock:
            self.mode = HomingMode.EXPLORING
            self.start_position = DeadReckoningState()
            self.current_position = DeadReckoningState()
            self.command_history.clear()
            self.session_start_time = time.time()
            self.last_exploration_start = time.time()
            self.last_return_home_time = time.time()
            self.stuck_state = StuckDetectorState()
            self.running = True
        
        self.logger.info("🏠 Homing session started - origin set to current position")
    
    def stop_session(self):
        """Stop the homing session"""
        with self.lock:
            self.running = False
            self.mode = HomingMode.AT_HOME
        self.logger.info("🏠 Homing session stopped")
    
    def record_command(self, linear_vel: float, angular_vel: float, duration: float = None):
        """
        Record a motion command and update dead reckoning position.
        
        Args:
            linear_vel: Linear velocity in m/s (positive = forward)
            angular_vel: Angular velocity in rad/s (positive = left/CCW)
            duration: Command duration in seconds (None = auto-calculate from timing)
        """
        current_time = time.time()
        
        with self.lock:
            if not self.running:
                return
            
            # Calculate duration if not provided
            if duration is None:
                duration = current_time - self.last_command_time
                # Cap duration to reasonable values (0.01s to 1s)
                duration = max(0.01, min(duration, 1.0))
            
            self.last_command_time = current_time
            
            # Create and store command
            cmd = MotionCommand(
                linear_velocity=linear_vel,
                angular_velocity=angular_vel,
                duration=duration,
                timestamp=current_time
            )
            self.command_history.append(cmd)
            
            # Update dead reckoning
            if self.dr_enabled:
                self._update_dead_reckoning(cmd)
            
            # Update stuck detection position history
            if self.stuck_enabled:
                self.stuck_state.last_positions.append(
                    (self.current_position.x, self.current_position.y, current_time)
                )
    
    def _update_dead_reckoning(self, cmd: MotionCommand):
        """Update position estimate using dead reckoning (assumes lock is held)"""
        dt = cmd.duration
        v = cmd.linear_velocity
        omega = cmd.angular_velocity
        
        if abs(omega) < 1e-6:
            # Straight line motion
            dx = v * math.cos(self.current_position.theta) * dt
            dy = v * math.sin(self.current_position.theta) * dt
            dtheta = 0.0
        else:
            # Arc motion (more accurate for turning)
            dtheta = omega * dt
            # Radius of curvature
            r = v / omega
            # Arc displacement
            dx = r * (math.sin(self.current_position.theta + dtheta) - 
                     math.sin(self.current_position.theta))
            dy = -r * (math.cos(self.current_position.theta + dtheta) - 
                      math.cos(self.current_position.theta))
        
        # Update position
        self.current_position.x += dx
        self.current_position.y += dy
        self.current_position.theta += dtheta
        
        # Normalize theta to [-pi, pi]
        self.current_position.theta = self._normalize_angle(self.current_position.theta)
        
        # Apply slight position decay to reduce drift accumulation
        self.current_position.x *= self.position_decay
        self.current_position.y *= self.position_decay
        
        # Track total distance
        self.current_position.total_distance += abs(v * dt)
    
    def check_should_return_home(self) -> bool:
        """Check if it's time to return home based on the interval"""
        with self.lock:
            if self.mode != HomingMode.EXPLORING:
                return False
            
            elapsed = time.time() - self.last_exploration_start
            return elapsed >= self.return_interval
    
    def is_stuck(self) -> bool:
        """Check if robot appears to be stuck"""
        if not self.stuck_enabled:
            return False
        
        with self.lock:
            current_time = time.time()
            
            # Only check periodically
            if current_time - self.stuck_state.last_check_time < self.stuck_check_interval:
                return self.stuck_state.consecutive_stuck_frames >= self.stuck_frames_threshold
            
            self.stuck_state.last_check_time = current_time
            
            # Need enough position history
            if len(self.stuck_state.last_positions) < 10:
                return False
            
            # Check if position has changed significantly
            positions = list(self.stuck_state.last_positions)
            oldest = positions[0]
            newest = positions[-1]
            
            # Calculate distance traveled in the recent window
            dx = newest[0] - oldest[0]
            dy = newest[1] - oldest[1]
            distance_moved = math.sqrt(dx*dx + dy*dy)
            time_elapsed = newest[2] - oldest[2]
            
            # If very little movement over significant time, might be stuck
            if time_elapsed > self.stuck_check_interval * 3:
                if distance_moved < self.stuck_threshold_distance:
                    self.stuck_state.consecutive_stuck_frames += 1
                    self.stuck_state.stuck_count += 1
                    self.logger.debug(
                        f"Stuck detection: moved {distance_moved:.3f}m in {time_elapsed:.1f}s "
                        f"(consecutive: {self.stuck_state.consecutive_stuck_frames})"
                    )
                else:
                    self.stuck_state.consecutive_stuck_frames = 0
            
            return self.stuck_state.consecutive_stuck_frames >= self.stuck_frames_threshold
    
    def start_stuck_recovery(self) -> bool:
        """
        Start stuck recovery mode - reverse recent commands.
        Returns True if recovery commands are prepared, False if not enough history.
        """
        with self.lock:
            if len(self.command_history) < 3:
                self.logger.warning("Not enough command history for stuck recovery")
                return False
            
            self.mode = HomingMode.STUCK_RECOVERY
            
            # Get recent commands in reverse order
            recent_commands = list(self.command_history)[-self.max_recovery_commands:]
            self.recovery_commands = [cmd.get_inverse() for cmd in reversed(recent_commands)]
            self.recovery_index = 0
            
            # Clear stuck state
            self.stuck_state.consecutive_stuck_frames = 0
            self.stuck_state.last_positions.clear()
            
            self.logger.info(f"🔧 Starting stuck recovery with {len(self.recovery_commands)} reverse commands")
            return True
    
    def get_next_recovery_command(self) -> Optional[MotionCommand]:
        """Get the next command in stuck recovery sequence"""
        with self.lock:
            if self.mode != HomingMode.STUCK_RECOVERY:
                return None
            
            if self.recovery_index >= len(self.recovery_commands):
                # Recovery complete
                self.mode = HomingMode.EXPLORING
                self.last_exploration_start = time.time()  # Reset timer
                self.logger.info("✅ Stuck recovery complete, resuming exploration")
                return None
            
            cmd = self.recovery_commands[self.recovery_index]
            self.recovery_index += 1
            return cmd
    
    def start_return_home(self):
        """Begin the return-to-home process"""
        with self.lock:
            if self.mode == HomingMode.RETURNING_HOME:
                return
            
            self.mode = HomingMode.RETURNING_HOME
            self.logger.info(
                f"🏠 Starting return home from ({self.current_position.x:.2f}, "
                f"{self.current_position.y:.2f})"
            )
    
    def get_return_home_command(self) -> Optional[Tuple[float, float]]:
        """
        Get velocity command to return home.
        
        Returns:
            Tuple of (linear_vel, angular_vel) or None if at home
        """
        with self.lock:
            if self.mode != HomingMode.RETURNING_HOME:
                return None
            
            # Calculate vector to home
            dx = self.start_position.x - self.current_position.x
            dy = self.start_position.y - self.current_position.y
            distance_to_home = math.sqrt(dx*dx + dy*dy)
            
            # Check if we're home
            if distance_to_home < self.home_tolerance:
                self.mode = HomingMode.AT_HOME
                self.last_return_home_time = time.time()
                self.logger.info("🏠 Arrived at home position!")
                return (0.0, 0.0)  # Stop
            
            # Calculate desired heading to home
            desired_heading = math.atan2(dy, dx)
            heading_error = self._normalize_angle(desired_heading - self.current_position.theta)
            
            # If heading error is large, rotate in place first
            if abs(heading_error) > self.heading_tolerance * 2:
                # Turn towards home
                angular_vel = self.turn_speed if heading_error > 0 else -self.turn_speed
                linear_vel = 0.1 * self.return_speed  # Very slow forward while turning
            else:
                # Move towards home with proportional heading correction
                linear_vel = self.return_speed
                # Proportional angular correction
                angular_vel = 2.0 * heading_error  # P-gain of 2.0
                angular_vel = max(-self.turn_speed, min(angular_vel, self.turn_speed))
            
            return (linear_vel, angular_vel)
    
    def mark_home_reached(self):
        """Manually mark that home has been reached (e.g., by external system)"""
        with self.lock:
            self.mode = HomingMode.AT_HOME
            self.current_position = DeadReckoningState()  # Reset position
            self.last_return_home_time = time.time()
            self.logger.info("🏠 Home position marked as reached")
    
    def resume_exploration(self):
        """Resume exploration mode after returning home"""
        with self.lock:
            if self.mode == HomingMode.AT_HOME:
                self.mode = HomingMode.EXPLORING
                self.last_exploration_start = time.time()
                self.command_history.clear()  # Fresh start
                self.logger.info("🚀 Resuming exploration from home")
    
    def get_mode(self) -> HomingMode:
        """Get current operating mode"""
        with self.lock:
            return self.mode
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current dead reckoning position (x, y, theta)"""
        with self.lock:
            return (
                self.current_position.x,
                self.current_position.y,
                self.current_position.theta
            )
    
    def get_distance_from_home(self) -> float:
        """Get distance from home position"""
        with self.lock:
            dx = self.current_position.x - self.start_position.x
            dy = self.current_position.y - self.start_position.y
            return math.sqrt(dx*dx + dy*dy)
    
    def get_time_until_return(self) -> float:
        """Get seconds until next scheduled return home"""
        with self.lock:
            if self.mode != HomingMode.EXPLORING:
                return 0.0
            elapsed = time.time() - self.last_exploration_start
            return max(0.0, self.return_interval - elapsed)
    
    def get_status(self) -> dict:
        """Get comprehensive status dict"""
        with self.lock:
            return {
                'mode': self.mode.value,
                'position': {
                    'x': self.current_position.x,
                    'y': self.current_position.y,
                    'theta': self.current_position.theta,
                    'theta_deg': math.degrees(self.current_position.theta)
                },
                'distance_from_home': self.get_distance_from_home(),
                'total_distance_traveled': self.current_position.total_distance,
                'command_history_length': len(self.command_history),
                'time_until_return': self.get_time_until_return(),
                'session_duration': time.time() - self.session_start_time,
                'stuck_count': self.stuck_state.stuck_count,
                'dead_reckoning_enabled': self.dr_enabled,
                'stuck_detection_enabled': self.stuck_enabled
            }
    
    def set_position_from_external(self, x: float, y: float, theta: float):
        """
        Update dead reckoning position from external source (e.g., ROS odometry).
        Useful for periodic correction/calibration.
        """
        with self.lock:
            self.current_position.x = x
            self.current_position.y = y
            self.current_position.theta = theta
            self.logger.debug(f"Position updated from external: ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
    
    def fuse_with_external_position(self, ext_x: float, ext_y: float, ext_theta: float, 
                                    weight: float = 0.3):
        """
        Fuse dead reckoning with external position estimate.
        
        Args:
            ext_x, ext_y, ext_theta: External position estimate
            weight: Weight for external estimate (0-1, higher = more trust in external)
        """
        with self.lock:
            self.current_position.x = (1 - weight) * self.current_position.x + weight * ext_x
            self.current_position.y = (1 - weight) * self.current_position.y + weight * ext_y
            
            # Special handling for angle wrapping
            theta_diff = self._normalize_angle(ext_theta - self.current_position.theta)
            self.current_position.theta += weight * theta_diff
            self.current_position.theta = self._normalize_angle(self.current_position.theta)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class ReversalPlanner:
    """
    Plans intelligent reversal maneuvers when simple command reversal isn't enough.
    Uses command history to find a good escape route.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        reversal_config = config.get('homing', {}).get('reversal', {})
        self.reverse_speed = reversal_config.get('reverse_speed', 0.2)  # m/s
        self.turn_speed = reversal_config.get('turn_speed', 0.6)  # rad/s
        self.reverse_duration = reversal_config.get('reverse_duration', 1.0)  # seconds
        self.turn_duration = reversal_config.get('turn_duration', 0.5)  # seconds
    
    def plan_escape_sequence(self, command_history: List[MotionCommand], 
                              current_position: DeadReckoningState) -> List[MotionCommand]:
        """
        Plan an escape sequence based on command history and current position.
        
        Strategy:
        1. If mostly going forward, reverse
        2. If stuck after reversal, try turning
        3. If still stuck, try the opposite direction from recent turn history
        """
        escape_commands = []
        
        if not command_history:
            # No history - just try backing up
            escape_commands.append(MotionCommand(
                linear_velocity=-self.reverse_speed,
                angular_velocity=0.0,
                duration=self.reverse_duration,
                timestamp=time.time()
            ))
            return escape_commands
        
        # Analyze recent command history
        recent = command_history[-10:] if len(command_history) >= 10 else command_history
        
        avg_linear = sum(c.linear_velocity for c in recent) / len(recent)
        avg_angular = sum(c.angular_velocity for c in recent) / len(recent)
        
        # Step 1: Reverse the average motion
        escape_commands.append(MotionCommand(
            linear_velocity=-avg_linear if abs(avg_linear) > 0.1 else -self.reverse_speed,
            angular_velocity=-avg_angular,
            duration=self.reverse_duration,
            timestamp=time.time()
        ))
        
        # Step 2: Turn in opposite direction of recent tendency
        turn_direction = 1.0 if avg_angular < 0 else -1.0
        escape_commands.append(MotionCommand(
            linear_velocity=0.1,  # Slight forward while turning
            angular_velocity=turn_direction * self.turn_speed,
            duration=self.turn_duration,
            timestamp=time.time()
        ))
        
        # Step 3: Try moving forward in new direction
        escape_commands.append(MotionCommand(
            linear_velocity=self.reverse_speed,
            angular_velocity=0.0,
            duration=self.reverse_duration,
            timestamp=time.time()
        ))
        
        self.logger.info(f"Planned escape sequence with {len(escape_commands)} maneuvers")
        return escape_commands
