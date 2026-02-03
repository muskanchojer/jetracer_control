"""
Autonomous control modules for JetRacer
Integrates dollop AI navigation with JetRacer hardware
"""

from .vision import VisionProcessor, CameraStream
from .policy import PolicyNetwork, PolicyEvaluator
from .control_adapter import ControlAdapter, SafetyMonitor

__all__ = [
    'VisionProcessor',
    'CameraStream',
    'PolicyNetwork',
    'PolicyEvaluator',
    'ControlAdapter',
    'SafetyMonitor',
]

__version__ = '1.0.0'
