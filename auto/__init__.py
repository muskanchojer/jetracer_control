"""
Autonomous control modules for JetRacer
Integrates dollop AI navigation with JetRacer hardware
"""

from .vision import VisionProcessor, CameraStream
from .policy import PolicyNetwork, PolicyEvaluator
from .control_adapter import ControlAdapter, SafetyMonitor
from .homing_controller import HomingController, HomingMode, ReversalPlanner

__all__ = [
    'VisionProcessor',
    'CameraStream',
    'PolicyNetwork',
    'PolicyEvaluator',
    'ControlAdapter',
    'SafetyMonitor',
    'HomingController',
    'HomingMode',
    'ReversalPlanner',
]

__version__ = '1.1.0'
