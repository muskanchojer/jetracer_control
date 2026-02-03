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
        
        # Extract linear acceleration
        accel = msg.linear_acceleration
        
        # Log every N messages with newline (for proper file logging)
        if self.msg_count % self.log_interval == 0:
            output = f"🧭 Roll: {roll_deg:6.2f}° | Pitch: {pitch_deg:6.2f}° | Yaw: {yaw_deg:6.2f}° | Accel: X={accel.x:5.2f} Y={accel.y:5.2f} Z={accel.z:5.2f} | Msgs: {self.msg_count}"
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