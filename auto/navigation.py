#!/usr/bin/env python3
"""
Navigation Module - Goal-based waypoint following using ROS odometry
Subscribes to /odom and /imu topics from Docker ROS1 container
Provides high-level navigation commands to reach goal waypoints
"""

import os
import sys
import time
import math
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Waypoint:
    """Represents a 2D waypoint with x, y coordinates"""
    x: float
    y: float
    tolerance: float = 0.3  # meters - how close to consider "reached"
    
    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to a point"""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def bearing_to(self, x: float, y: float) -> float:
        """Calculate bearing angle to this waypoint from a point (radians)"""
        return math.atan2(self.y - y, self.x - x)


@dataclass
class NavigationState:
    """Current navigation state"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # yaw angle in radians
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0
    timestamp: float = 0.0
    # IMU-fused position data
    fused_x: float = 0.0
    fused_y: float = 0.0
    fused_theta: float = 0.0
    position_confidence: float = 0.0
    use_fused: bool = False  # Whether fused position is available


@dataclass
class NavigationCommand:
    """High-level navigation command"""
    target_speed: float  # m/s
    target_heading: float  # radians
    distance_to_goal: float  # meters
    at_goal: bool = False


class WaypointNavigator:
    """
    High-level waypoint navigation using odometry
    Provides goal-directed commands to the fusion controller
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Navigation parameters
        nav_config = config.get('navigation', {})
        self.max_speed = nav_config.get('max_speed', 0.5)  # m/s
        self.min_speed = nav_config.get('min_speed', 0.1)  # m/s
        self.max_angular_speed = nav_config.get('max_angular_speed', 1.5)  # rad/s
        self.waypoint_tolerance = nav_config.get('waypoint_tolerance', 0.3)  # meters
        self.heading_tolerance = nav_config.get('heading_tolerance', 0.2)  # radians (~11 degrees)
        self.lookahead_distance = nav_config.get('lookahead_distance', 0.5)  # meters
        
        # Slowdown parameters
        self.slowdown_distance = nav_config.get('slowdown_distance', 1.0)  # start slowing down at 1m
        
        # State
        self.current_state = NavigationState()
        self.waypoint_queue = deque()
        self.current_waypoint: Optional[Waypoint] = None
        
        # ROS interface (will be initialized when ROS is available)
        self.ros_available = False
        self.odom_sub = None
        self.imu_sub = None
        self.ros_bridge = None
        
        # Sensor fusion settings
        self.use_sensor_fusion = nav_config.get('use_sensor_fusion', True)
        self.odom_weight = nav_config.get('odom_weight', 0.6)
        self.imu_weight = nav_config.get('imu_weight', 0.4)
        self.min_confidence_for_waypoint = nav_config.get('min_confidence_for_waypoint', 0.5)
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        
        # Simulated odometry mode
        self.simulated_mode = False
        self.last_update_time = None
        
        self.logger.info("WaypointNavigator initialized (with IMU sensor fusion)")
    
    def initialize_ros(self):
        """Initialize ROS subscribers (call after roscore is running)"""
        try:
            # Use ROS topic bridge (docker exec method) - works from outside Docker!
            from auto.ros_bridge import ROSTopicBridge
            
            self.logger.info("Using ROS Topic Bridge (via Docker exec - streaming mode with IMU)")
            self.ros_bridge = ROSTopicBridge()
            self.ros_bridge.start()
            
            # Start a thread to update from bridge at high rate
            self._bridge_running = True
            
            def update_from_bridge():
                while self._bridge_running:
                    # Get raw odometry
                    x, y, theta, ts = self.ros_bridge.get_odometry()
                    
                    # Get fused position (odometry + IMU)
                    if self.use_sensor_fusion:
                        fused_x, fused_y, fused_theta, confidence, fused_ts = \
                            self.ros_bridge.get_fused_position(
                                odom_weight=self.odom_weight,
                                imu_weight=self.imu_weight
                            )
                    else:
                        fused_x, fused_y, fused_theta = x, y, theta
                        confidence = 1.0
                        fused_ts = ts
                    
                    if ts > 0 or fused_ts > 0:
                        with self.lock:
                            # Store raw odometry
                            self.current_state.x = x
                            self.current_state.y = y
                            self.current_state.theta = theta
                            self.current_state.timestamp = ts
                            
                            # Store fused position
                            self.current_state.fused_x = fused_x
                            self.current_state.fused_y = fused_y
                            self.current_state.fused_theta = fused_theta
                            self.current_state.position_confidence = confidence
                            self.current_state.use_fused = self.use_sensor_fusion
                    
                    time.sleep(0.02)  # 50 Hz update rate for responsive navigation
            
            self.bridge_thread = threading.Thread(target=update_from_bridge, daemon=True)
            self.bridge_thread.start()
            
            self.ros_available = True
            self.simulated_mode = False
            self.logger.info("✅ ROS navigation interface initialized (odometry + IMU fusion from Docker)")
            
        except Exception as e:
            self.logger.warning(f"⚠️ ROS bridge failed: {e}")
            self.logger.warning("   Navigation will use simulated odometry")
            self.ros_available = False
            self._enable_simulated_odometry()
    
    def _enable_simulated_odometry(self):
        """
        Enable simulated odometry when ROS is not available
        Estimates position based on commanded velocities (dead reckoning)
        """
        self.logger.info("📍 Enabled simulated odometry mode")
        self.simulated_mode = True
        self.last_update_time = time.time()
    
    def _odom_callback(self, msg):
        """Callback for odometry messages from /odom topic"""
        with self.lock:
            # Extract position
            self.current_state.x = msg.pose.pose.position.x
            self.current_state.y = msg.pose.pose.position.y
            
            # Extract orientation (quaternion to yaw)
            q = msg.pose.pose.orientation
            self.current_state.theta = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)
            
            # Extract velocities
            self.current_state.vx = msg.twist.twist.linear.x
            self.current_state.vy = msg.twist.twist.linear.y
            self.current_state.vtheta = msg.twist.twist.angular.z
            
            self.current_state.timestamp = time.time()
    
    def _imu_callback(self, msg):
        """Callback for IMU messages (backup orientation source)"""
        # IMU provides orientation as quaternion
        # Only update if odometry is stale
        with self.lock:
            if time.time() - self.current_state.timestamp > 0.5:
                q = msg.orientation
                self.current_state.theta = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)
    
    @staticmethod
    def _quaternion_to_yaw(x, y, z, w):
        """Convert quaternion to yaw angle (radians)"""
        # Roll, pitch, yaw from quaternion
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def set_goal(self, x: float, y: float, tolerance: float = None):
        """Set a single goal waypoint (clears existing waypoints)"""
        with self.lock:
            self.waypoint_queue.clear()
            tol = tolerance if tolerance is not None else self.waypoint_tolerance
            waypoint = Waypoint(x, y, tol)
            self.waypoint_queue.append(waypoint)
            self.current_waypoint = waypoint
            self.logger.info(f"🎯 New goal set: ({x:.2f}, {y:.2f})")
    
    def add_waypoint(self, x: float, y: float, tolerance: float = None):
        """Add a waypoint to the queue"""
        with self.lock:
            tol = tolerance if tolerance is not None else self.waypoint_tolerance
            waypoint = Waypoint(x, y, tol)
            self.waypoint_queue.append(waypoint)
            
            if self.current_waypoint is None:
                self.current_waypoint = waypoint
            
            self.logger.info(f"📍 Waypoint added: ({x:.2f}, {y:.2f}) [Queue: {len(self.waypoint_queue)}]")
    
    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set multiple waypoints at once"""
        with self.lock:
            self.waypoint_queue.clear()
            for x, y in waypoints:
                self.waypoint_queue.append(Waypoint(x, y, self.waypoint_tolerance))
            
            if self.waypoint_queue:
                self.current_waypoint = self.waypoint_queue[0]
                self.logger.info(f"🗺️ {len(waypoints)} waypoints loaded")
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        with self.lock:
            self.waypoint_queue.clear()
            self.current_waypoint = None
            self.logger.info("Waypoints cleared")
    
    def has_goal(self) -> bool:
        """Check if there's an active goal"""
        with self.lock:
            return self.current_waypoint is not None
    
    def get_navigation_command(self) -> Optional[NavigationCommand]:
        """
        Calculate navigation command to reach current waypoint
        Uses sensor-fused position (odometry + IMU) for accurate waypoint detection
        Returns NavigationCommand or None if no active waypoint
        """
        with self.lock:
            if self.current_waypoint is None:
                return None
            
            # Use fused position if available and sensor fusion is enabled
            if self.current_state.use_fused and self.current_state.position_confidence > 0:
                # Use IMU-fused position for more accurate waypoint detection
                x = self.current_state.fused_x
                y = self.current_state.fused_y
                theta = self.current_state.fused_theta
                confidence = self.current_state.position_confidence
                position_source = "fused (odom+IMU)"
            else:
                # Fall back to raw odometry
                x = self.current_state.x
                y = self.current_state.y
                theta = self.current_state.theta
                confidence = 1.0
                position_source = "odometry only"
            
            # Distance and bearing to goal using fused position
            distance = self.current_waypoint.distance_to(x, y)
            bearing = self.current_waypoint.bearing_to(x, y)
            
            # Also check distance from raw odometry for validation
            odom_distance = self.current_waypoint.distance_to(
                self.current_state.x, self.current_state.y
            )
            
            # Check if waypoint reached using fused position
            # Require minimum confidence for waypoint confirmation
            waypoint_reached = False
            if distance < self.current_waypoint.tolerance:
                if confidence >= self.min_confidence_for_waypoint:
                    waypoint_reached = True
                    self.logger.info(
                        f"✅ Waypoint reached ({position_source}): "
                        f"({self.current_waypoint.x:.2f}, {self.current_waypoint.y:.2f}) "
                        f"[confidence: {confidence:.2f}]"
                    )
                elif odom_distance < self.current_waypoint.tolerance:
                    # Low fusion confidence but odometry also says we're there
                    waypoint_reached = True
                    self.logger.info(
                        f"✅ Waypoint reached (odom confirmed): "
                        f"({self.current_waypoint.x:.2f}, {self.current_waypoint.y:.2f}) "
                        f"[confidence: {confidence:.2f}]"
                    )
                else:
                    # Fused says reached but low confidence and odometry disagrees
                    self.logger.debug(
                        f"⚠️ Waypoint proximity detected but low confidence ({confidence:.2f}), "
                        f"continuing navigation"
                    )
            
            if waypoint_reached:
                # Move to next waypoint
                if len(self.waypoint_queue) > 1:
                    self.waypoint_queue.popleft()
                    self.current_waypoint = self.waypoint_queue[0]
                    self.logger.info(f"🎯 Next waypoint: ({self.current_waypoint.x:.2f}, {self.current_waypoint.y:.2f})")
                    return self.get_navigation_command()  # Recursive call for new waypoint
                else:
                    # All waypoints reached
                    self.current_waypoint = None
                    self.waypoint_queue.clear()
                    return NavigationCommand(
                        target_speed=0.0,
                        target_heading=theta,
                        distance_to_goal=0.0,
                        at_goal=True
                    )
            
            # Calculate heading error
            heading_error = self._normalize_angle(bearing - theta)
            
            # Calculate target speed (slow down near goal)
            if distance < self.slowdown_distance:
                # Linear ramp down
                speed_factor = distance / self.slowdown_distance
                target_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            else:
                target_speed = self.max_speed
            
            # Reduce speed if large heading error
            if abs(heading_error) > self.heading_tolerance:
                # Slow down when turning
                turn_factor = 1.0 - min(abs(heading_error) / math.pi, 1.0)
                target_speed *= (0.3 + 0.7 * turn_factor)  # min 30% speed when turning
            
            # Ensure minimum speed
            target_speed = max(target_speed, self.min_speed)
            
            return NavigationCommand(
                target_speed=target_speed,
                target_heading=bearing,
                distance_to_goal=distance,
                at_goal=False
            )
    
    def update_simulated_odometry(self, linear_velocity, angular_velocity, dt):
        """
        Update simulated odometry based on commanded velocities
        Used when ROS odometry is not available
        
        Args:
            linear_velocity: Forward velocity (m/s)
            angular_velocity: Turning velocity (rad/s)
            dt: Time delta (seconds)
        """
        if not self.simulated_mode:
            return
        
        with self.lock:
            # Simple dead reckoning
            # dx = v * cos(theta) * dt
            # dy = v * sin(theta) * dt
            # dtheta = omega * dt
            
            dx = linear_velocity * math.cos(self.current_state.theta) * dt
            dy = linear_velocity * math.sin(self.current_state.theta) * dt
            dtheta = angular_velocity * dt
            
            self.current_state.x += dx
            self.current_state.y += dy
            self.current_state.theta += dtheta
            self.current_state.theta = self._normalize_angle(self.current_state.theta)
            
            self.current_state.vx = linear_velocity
            self.current_state.vtheta = angular_velocity
            self.current_state.timestamp = time.time()
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_state(self) -> NavigationState:
        """Get current navigation state (includes fused position data)"""
        with self.lock:
            return NavigationState(
                x=self.current_state.x,
                y=self.current_state.y,
                theta=self.current_state.theta,
                vx=self.current_state.vx,
                vy=self.current_state.vy,
                vtheta=self.current_state.vtheta,
                timestamp=self.current_state.timestamp,
                fused_x=self.current_state.fused_x,
                fused_y=self.current_state.fused_y,
                fused_theta=self.current_state.fused_theta,
                position_confidence=self.current_state.position_confidence,
                use_fused=self.current_state.use_fused
            )
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, theta) - uses fused position if available"""
        with self.lock:
            if self.current_state.use_fused and self.current_state.position_confidence > 0:
                return (self.current_state.fused_x, 
                        self.current_state.fused_y, 
                        self.current_state.fused_theta)
            return self.current_state.x, self.current_state.y, self.current_state.theta
    
    def get_raw_position(self) -> Tuple[float, float, float]:
        """Get raw odometry position (x, y, theta) without IMU fusion"""
        with self.lock:
            return self.current_state.x, self.current_state.y, self.current_state.theta
    
    def get_status(self) -> dict:
        """Get navigation status for logging/debugging (includes sensor fusion info)"""
        with self.lock:
            # Use fused position if available
            if self.current_state.use_fused and self.current_state.position_confidence > 0:
                pos_x, pos_y, pos_theta = (
                    self.current_state.fused_x,
                    self.current_state.fused_y, 
                    self.current_state.fused_theta
                )
                position_source = "fused"
            else:
                pos_x, pos_y, pos_theta = (
                    self.current_state.x,
                    self.current_state.y,
                    self.current_state.theta
                )
                position_source = "odometry"
            
            status = {
                'position': (pos_x, pos_y, pos_theta),
                'position_raw': (self.current_state.x, self.current_state.y, self.current_state.theta),
                'position_fused': (self.current_state.fused_x, self.current_state.fused_y, self.current_state.fused_theta),
                'position_confidence': self.current_state.position_confidence,
                'position_source': position_source,
                'sensor_fusion_enabled': self.use_sensor_fusion,
                'has_goal': self.current_waypoint is not None,
                'waypoints_remaining': len(self.waypoint_queue),
            }
            
            if self.current_waypoint:
                # Calculate distance using fused position
                distance = self.current_waypoint.distance_to(pos_x, pos_y)
                distance_odom = self.current_waypoint.distance_to(
                    self.current_state.x, 
                    self.current_state.y
                )
                status['current_goal'] = (self.current_waypoint.x, self.current_waypoint.y)
                status['distance_to_goal'] = distance
                status['distance_to_goal_odom'] = distance_odom
            
            return status
