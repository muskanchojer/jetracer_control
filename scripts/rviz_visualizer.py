#!/usr/bin/env python3
"""
RViz Visualization Node for Goal-Based Navigation
Publishes goal markers and path to RViz for visualization
Run this alongside goal_based_control.py
"""

import sys
import time
from pathlib import Path

# ROS imports
try:
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Path
except ImportError:
    print("❌ rospy not available. Install ROS Python bindings.")
    sys.exit(1)


class NavigationVisualizer:
    """Visualize navigation goals and robot path in RViz"""
    
    def __init__(self):
        rospy.init_node('navigation_visualizer', anonymous=True)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/navigation/goal_markers', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('/navigation/path', Path, queue_size=10)
        self.goal_pub = rospy.Publisher('/navigation/current_goal', PoseStamped, queue_size=10)
        
        # Current goals
        self.goals = []
        self.path_history = []
        
        rospy.loginfo("Navigation visualizer initialized")
        rospy.loginfo("Open RViz and add these topics:")
        rospy.loginfo("  - /navigation/goal_markers (MarkerArray)")
        rospy.loginfo("  - /navigation/path (Path)")
        rospy.loginfo("  - /navigation/current_goal (PoseStamped)")
    
    def set_goals(self, goals):
        """
        Set goals to visualize
        Args:
            goals: List of (x, y) tuples
        """
        self.goals = goals
        rospy.loginfo(f"Set {len(goals)} goals for visualization")
    
    def publish_goal_markers(self):
        """Publish goal markers for RViz"""
        marker_array = MarkerArray()
        
        for i, (x, y) in enumerate(self.goals):
            # Goal sphere
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "goals"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.2
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Color: green for current goal, blue for future goals
            if i == 0:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
            # Text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "goal_labels"
            text_marker.id = 100 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.5
            
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"Goal {i+1}\n({x:.1f}, {y:.1f})"
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_current_goal(self, x, y):
        """Publish current active goal"""
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
    
    def run(self):
        """Main loop - publish markers periodically"""
        rate = rospy.Rate(2)  # 2 Hz
        
        while not rospy.is_shutdown():
            if self.goals:
                self.publish_goal_markers()
                
                # Publish first goal as current
                if len(self.goals) > 0:
                    self.publish_current_goal(self.goals[0][0], self.goals[0][1])
            
            rate.sleep()


def main():
    """Example usage"""
    visualizer = NavigationVisualizer()
    
    # Example: Set some goals
    example_goals = [
        (1.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (0.0, 2.0)
    ]
    
    visualizer.set_goals(example_goals)
    
    try:
        visualizer.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
