#!/usr/bin/env python3
"""
Fusion Controller - Blends navigation commands with vision-based obstacle avoidance
Priority: Safety (Vision) > Navigation (Goals)
"""

import math
import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from auto.navigation import NavigationCommand


@dataclass
class VisionCommand:
    """Vision-based action from policy network"""
    action: str  # 'left', 'straight', 'right', 'reverse'
    confidence: float  # 0-1
    obstacle_detected: bool = False


@dataclass
class FusedCommand:
    """Final fused motor command"""
    linear_x: float  # Forward speed (m/s)
    angular_z: float  # Turning speed (rad/s)
    mode: str  # 'navigation', 'vision', 'blended'
    reason: str = ""


class FusionController:
    """
    Blends high-level navigation commands with low-level vision-based obstacle avoidance
    
    Strategy:
    1. If vision detects obstacles -> Vision takes priority (safety first)
    2. If clear path ahead -> Navigation directs, vision fine-tunes
    3. Smooth blending based on confidence levels
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Fusion parameters
        fusion_config = config.get('fusion', {})
        self.vision_weight = fusion_config.get('vision_weight', 0.7)  # Priority to vision
        self.navigation_weight = fusion_config.get('navigation_weight', 0.3)
        
        # Speed parameters
        self.max_linear_speed = fusion_config.get('max_linear_speed', 0.5)  # m/s
        self.max_angular_speed = fusion_config.get('max_angular_speed', 1.5)  # rad/s
        
        # Vision action mappings
        self.vision_actions = {
            'left': {'linear': 0.3, 'angular': 0.8},
            'right': {'linear': 0.3, 'angular': -0.8},
            'straight': {'linear': 0.5, 'angular': 0.0},
            'reverse': {'linear': -0.2, 'angular': 0.0}
        }
        
        # Obstacle detection thresholds
        self.obstacle_threshold = fusion_config.get('obstacle_confidence_threshold', 0.6)
        self.critical_obstacle_threshold = fusion_config.get('critical_obstacle_threshold', 0.8)
        
        self.logger.info("FusionController initialized")
        self.logger.info(f"  Vision weight: {self.vision_weight}, Navigation weight: {self.navigation_weight}")
    
    def fuse_commands(
        self, 
        nav_cmd: Optional[NavigationCommand],
        vision_cmd: Optional[VisionCommand],
        current_heading: float
    ) -> FusedCommand:
        """
        Fuse navigation and vision commands into final motor command
        
        Args:
            nav_cmd: High-level navigation command (can be None)
            vision_cmd: Vision-based action (can be None)
            current_heading: Current robot heading in radians
            
        Returns:
            FusedCommand with linear_x and angular_z velocities
        """
        
        # Case 1: No navigation goal - pure vision mode (obstacle avoidance only)
        if nav_cmd is None or nav_cmd.at_goal:
            if vision_cmd:
                return self._pure_vision_command(vision_cmd, "No active goal")
            else:
                return FusedCommand(0.0, 0.0, 'stopped', 'No goal and no vision')
        
        # Case 2: No vision input - pure navigation (not recommended, but fallback)
        if vision_cmd is None:
            return self._pure_navigation_command(nav_cmd, current_heading, "No vision input")
        
        # Case 3: Critical obstacle detected - vision overrides completely
        if vision_cmd.confidence > self.critical_obstacle_threshold and \
           vision_cmd.action in ['left', 'right', 'reverse']:
            return self._pure_vision_command(vision_cmd, "Critical obstacle detected")
        
        # Case 4: Obstacle detected - vision priority with navigation influence
        if vision_cmd.obstacle_detected or vision_cmd.confidence > self.obstacle_threshold:
            return self._vision_priority_blend(nav_cmd, vision_cmd, current_heading)
        
        # Case 5: Clear path - navigation priority with vision fine-tuning
        return self._navigation_priority_blend(nav_cmd, vision_cmd, current_heading)
    
    def _pure_vision_command(self, vision_cmd: VisionCommand, reason: str) -> FusedCommand:
        """Convert vision action to motor command"""
        action = vision_cmd.action
        params = self.vision_actions.get(action, self.vision_actions['straight'])
        
        linear = params['linear'] * self.max_linear_speed
        angular = params['angular'] * self.max_angular_speed
        
        return FusedCommand(
            linear_x=linear,
            angular_z=angular,
            mode='vision',
            reason=reason
        )
    
    def _pure_navigation_command(
        self, 
        nav_cmd: NavigationCommand, 
        current_heading: float,
        reason: str
    ) -> FusedCommand:
        """Convert navigation command to motor command"""
        
        # Calculate heading error
        heading_error = self._normalize_angle(nav_cmd.target_heading - current_heading)
        
        # Linear velocity from navigation
        linear = nav_cmd.target_speed
        
        # Angular velocity proportional to heading error
        # Use proportional control: angular = Kp * error
        Kp = 2.0  # Proportional gain
        angular = Kp * heading_error
        
        # Limit angular velocity
        angular = np.clip(angular, -self.max_angular_speed, self.max_angular_speed)
        
        return FusedCommand(
            linear_x=linear,
            angular_z=angular,
            mode='navigation',
            reason=reason
        )
    
    def _vision_priority_blend(
        self,
        nav_cmd: NavigationCommand,
        vision_cmd: VisionCommand,
        current_heading: float
    ) -> FusedCommand:
        """
        Blend with vision priority (obstacle avoidance mode)
        Vision gets 70-90% weight, navigation provides directional hint
        """
        
        # Get base commands
        vision_params = self.vision_actions.get(
            vision_cmd.action, 
            self.vision_actions['straight']
        )
        
        # Vision command
        vision_linear = vision_params['linear'] * self.max_linear_speed
        vision_angular = vision_params['angular'] * self.max_angular_speed
        
        # Navigation influence (reduced)
        heading_error = self._normalize_angle(nav_cmd.target_heading - current_heading)
        nav_angular = 2.0 * heading_error  # Proportional control
        nav_angular = np.clip(nav_angular, -self.max_angular_speed, self.max_angular_speed)
        
        # Blend: Heavy vision weight
        weight_vision = 0.8
        weight_nav = 0.2
        
        linear = vision_linear  # Keep vision speed for safety
        angular = weight_vision * vision_angular + weight_nav * nav_angular
        
        # Limit final command
        angular = np.clip(angular, -self.max_angular_speed, self.max_angular_speed)
        
        return FusedCommand(
            linear_x=linear,
            angular_z=angular,
            mode='blended',
            reason=f'Obstacle avoidance + navigation (vision: {vision_cmd.action})'
        )
    
    def _navigation_priority_blend(
        self,
        nav_cmd: NavigationCommand,
        vision_cmd: VisionCommand,
        current_heading: float
    ) -> FusedCommand:
        """
        Blend with navigation priority (clear path mode)
        Navigation directs, vision provides fine corrections
        """
        
        # Navigation command
        heading_error = self._normalize_angle(nav_cmd.target_heading - current_heading)
        nav_linear = nav_cmd.target_speed
        nav_angular = 2.0 * heading_error  # Proportional control
        nav_angular = np.clip(nav_angular, -self.max_angular_speed, self.max_angular_speed)
        
        # Vision correction
        vision_params = self.vision_actions.get(
            vision_cmd.action,
            self.vision_actions['straight']
        )
        vision_angular = vision_params['angular'] * self.max_angular_speed
        
        # Blend: Navigation priority
        weight_nav = 0.7
        weight_vision = 0.3
        
        linear = nav_linear
        angular = weight_nav * nav_angular + weight_vision * vision_angular
        
        # If vision says reverse but navigation says forward, trust vision
        if vision_cmd.action == 'reverse' and vision_cmd.confidence > 0.5:
            linear = min(linear, 0.1)  # Slow down significantly
        
        # Limit final command
        angular = np.clip(angular, -self.max_angular_speed, self.max_angular_speed)
        
        return FusedCommand(
            linear_x=linear,
            angular_z=angular,
            mode='blended',
            reason=f'Navigation to goal + vision correction'
        )
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def vision_action_to_command(self, action: str, confidence: float) -> VisionCommand:
        """
        Convert policy network output to VisionCommand
        
        Args:
            action: One of ['left', 'straight', 'right', 'reverse']
            confidence: Confidence score 0-1
            
        Returns:
            VisionCommand object
        """
        # Detect obstacles based on action
        obstacle_detected = action in ['left', 'right', 'reverse']
        
        return VisionCommand(
            action=action,
            confidence=confidence,
            obstacle_detected=obstacle_detected
        )
    
    def get_status(self) -> dict:
        """Get controller status"""
        return {
            'vision_weight': self.vision_weight,
            'navigation_weight': self.navigation_weight,
            'max_linear_speed': self.max_linear_speed,
            'max_angular_speed': self.max_angular_speed
        }
