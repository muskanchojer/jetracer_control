#!/usr/bin/env python3
"""Quick script to show robot's current position from odometry"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto.ros_bridge import ROSTopicBridge
import time

bridge = ROSTopicBridge(container_name="jetracer_ros")
time.sleep(2)  # Wait for data

x, y, theta, ts = bridge.get_odometry()
print(f"\n🤖 Robot Current Position:")
print(f"   X: {x:.3f}m")
print(f"   Y: {y:.3f}m")
print(f"   Heading: {theta:.1f}°")
print(f"\n🎯 Goal: (2.0, 1.5)")
print(f"   Distance from current position: {((2.0-x)**2 + (1.5-y)**2)**0.5:.2f}m\n")

bridge.stop()
