#!/usr/bin/env python3
"""
Map-Based Navigation with SLAM (Optional Advanced Feature)
Integrates with ROS SLAM for map building and navigation
Requires: gmapping or cartographer to be running in Docker
"""

import sys
import time
import logging
from pathlib import Path

try:
    import rospy
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import PoseStamped
except ImportError:
    print("❌ rospy not available")
    sys.exit(1)


class SLAMNavigator:
    """
    Map-based navigation using SLAM
    This is an ADVANCED feature that requires SLAM to be running
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Subscribe to SLAM map
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self._map_callback)
        
        # Current map
        self.map = None
        self.map_resolution = 0.05  # meters per cell
        self.map_origin = None
        
        self.logger.info("SLAM Navigator initialized")
        self.logger.info("Waiting for /map topic from SLAM...")
    
    def _map_callback(self, msg):
        """Callback for SLAM map updates"""
        self.map = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        
        self.logger.info(f"Received map: {msg.info.width}x{msg.info.height} @ {self.map_resolution}m/cell")
    
    def has_map(self):
        """Check if map is available"""
        return self.map is not None
    
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path using A* on the occupancy grid
        
        Returns:
            List of (x, y) waypoints or None if no path found
        """
        if not self.has_map():
            self.logger.warning("No map available for path planning")
            return None
        
        # TODO: Implement A* path planning on occupancy grid
        # This is a placeholder for future implementation
        
        self.logger.info(f"Path planning from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
        
        # For now, return straight line waypoints
        num_waypoints = 5
        waypoints = []
        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints
            wx = start_x + t * (goal_x - start_x)
            wy = start_y + t * (goal_y - start_y)
            waypoints.append((wx, wy))
        
        return waypoints


def main():
    """
    Example: Using SLAM for navigation
    
    Prerequisites:
    1. Run SLAM in Docker:
       docker exec -it jetracer_ros bash
       source /root/catkin_ws/devel/setup.bash
       roslaunch jetracer gmapping.launch  # or cartographer.launch
    
    2. Build a map by driving robot around
    
    3. Use this script to navigate with the map
    """
    rospy.init_node('slam_navigator_example', anonymous=True)
    
    navigator = SLAMNavigator()
    
    # Wait for map
    print("Waiting for SLAM map...")
    timeout = 10
    start_time = time.time()
    
    while not navigator.has_map() and (time.time() - start_time) < timeout:
        time.sleep(0.5)
    
    if navigator.has_map():
        print("✅ Map received!")
        
        # Example: Plan path to goal
        waypoints = navigator.plan_path(0, 0, 2, 2)
        
        if waypoints:
            print(f"✅ Path found with {len(waypoints)} waypoints:")
            for i, (x, y) in enumerate(waypoints):
                print(f"  {i+1}. ({x:.2f}, {y:.2f})")
    else:
        print("❌ No map received. Is SLAM running?")
        print("\nTo enable SLAM mapping:")
        print("1. docker exec -it jetracer_ros bash")
        print("2. source /root/catkin_ws/devel/setup.bash")
        print("3. Check if gmapping or cartographer launch files exist:")
        print("   roslaunch jetracer gmapping.launch")


if __name__ == '__main__':
    main()
