#!/usr/bin/env python3
"""
Verbose IMU Monitor for JetRacer
Displays ALL orientation, acceleration, and compass heading data
"""

import rospy
from sensor_msgs.msg import Imu
import math
import sys

class IMUMonitorVerbose: 
    def __init__(self):
        rospy.init_node('imu_monitor_verbose')
        
        # Subscribe to IMU topic
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        self.msg_count = 0
        
        print("="*70)
        print("📡 VERBOSE IMU Monitor Started")
        print("="*70)
        print("   Printing ALL IMU messages with compass heading")
        print("   Press Ctrl+C to stop")
        print("="*70 + "\n")
        sys.stdout.flush()
        
    def yaw_to_compass(self, yaw_deg):
        """Convert yaw angle to compass direction
        
        Args:
            yaw_deg: Yaw angle in degrees (-180 to 180)
            
        Returns:
            tuple: (compass_heading, direction_string)
        """
        # Normalize yaw to 0-360 range (0 = North, 90 = East, 180 = South, 270 = West)
        compass_heading = (yaw_deg + 360) % 360
        
        # Determine cardinal direction
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((compass_heading + 22.5) / 45) % 8
        direction = directions[index]
        
        return compass_heading, direction
        
    def imu_callback(self, msg):
        """Process and print ALL IMU data"""
        self.msg_count += 1
        
        # Extract orientation (quaternion)
        orientation = msg.orientation
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        
        # Convert quaternion to roll, pitch, yaw (in radians)
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(2*(w*y - z*x))
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        # Convert to degrees
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        # Convert yaw to compass heading
        compass_heading, compass_dir = self.yaw_to_compass(yaw_deg)
        
        # Extract linear acceleration
        accel = msg.linear_acceleration
        
        # Extract angular velocity
        gyro = msg.angular_velocity
        
        # Print everything
        output = (
            f"MSG #{self.msg_count:05d} | "
            f"🧭 Compass: {compass_heading:6.2f}° {compass_dir:2s} | "
            f"Roll: {roll_deg:7.2f}° | Pitch: {pitch_deg:7.2f}° | Yaw: {yaw_deg:7.2f}° | "
            f"Accel: X={accel.x:6.2f} Y={accel.y:6.2f} Z={accel.z:6.2f} | "
            f"Gyro: X={gyro.x:6.2f} Y={gyro.y:6.2f} Z={gyro.z:6.2f}"
        )
        print(output)
        sys.stdout.flush()
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping verbose IMU monitor...")
            print(f"Total messages received: {self.msg_count}")
            sys.stdout.flush()

if __name__ == '__main__':
    try:
        monitor = IMUMonitorVerbose()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
