#!/usr/bin/env python3
"""
IMU Monitor for JetRacer
Displays orientation and acceleration data
"""

import rospy
from sensor_msgs.msg import Imu
import math
import sys

class IMUMonitor: 
    def __init__(self, log_interval=10):
        rospy.init_node('imu_monitor')
        
        # Subscribe to IMU topic
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        self.msg_count = 0
        self.log_interval = log_interval  # Log every N messages
        
        print("="*50)
        print("📡 IMU Monitor Started")
        print("="*50)
        print("   Monitoring orientation and acceleration")
        print("   Press Ctrl+C to stop")
        print("="*50 + "\n")
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
        """Process IMU data"""
        self.msg_count += 1
        
        # Extract orientation (quaternion to euler)
        orientation = msg.orientation
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        
        # Convert to roll, pitch, yaw
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
        
        # Log every N messages with newline (for proper file logging)
        if self.msg_count % self.log_interval == 0:
            output = f"🧭 Compass: {compass_heading:6.2f}° {compass_dir:2s} | Roll: {roll_deg:6.2f}° | Pitch: {pitch_deg:6.2f}° | Yaw: {yaw_deg:6.2f}° | Accel: X={accel.x:5.2f} Y={accel.y:5.2f} Z={accel.z:5.2f} | Msgs: {self.msg_count}"
            print(output)
            sys.stdout.flush()
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping IMU monitor...")
            sys.stdout.flush()

if __name__ == '__main__':
    try:
        monitor = IMUMonitor(log_interval=10)  # Log every 10 IMU messages
        monitor.run()
    except rospy.ROSInterruptException:
        pass