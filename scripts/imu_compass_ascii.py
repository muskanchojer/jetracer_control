#!/usr/bin/env python3
"""
Simple ASCII Compass for Terminal
Shows compass direction without needing OpenCV
"""

import rospy
from sensor_msgs.msg import Imu
import math
import sys
import os

class ASCIICompass:
    def __init__(self):
        rospy.init_node('imu_compass_ascii')
        
        # Subscribe to IMU topic
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        self.compass_heading = 0.0
        self.compass_dir = "N"
        self.yaw = 0.0
        self.msg_count = 0
        
        print("🧭 ASCII Compass Started - Press Ctrl+C to stop\n")
        
    def yaw_to_compass(self, yaw_deg):
        """Convert yaw angle to compass direction"""
        compass_heading = (yaw_deg + 360) % 360
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((compass_heading + 22.5) / 45) % 8
        direction = directions[index]
        return compass_heading, direction
    
    def draw_ascii_compass(self):
        """Draw ASCII compass in terminal"""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Calculate arrow position
        angle = self.compass_heading
        
        # Create compass grid (21x21)
        size = 21
        center = size // 2
        grid = [[' ' for _ in range(size)] for _ in range(size)]
        
        # Draw circle
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                if 8.5 < dist < 9.5:
                    grid[i][j] = '·'
        
        # Draw cardinal directions
        grid[1][center] = 'N'
        grid[size-2][center] = 'S'
        grid[center][1] = 'W'
        grid[center][size-2] = 'E'
        
        # Draw heading arrow
        rad = math.radians(angle - 90)  # -90 to make 0° point up
        for length in range(1, 9):
            x = int(center + length * math.cos(rad))
            y = int(center + length * math.sin(rad))
            if 0 <= x < size and 0 <= y < size:
                if length == 8:
                    grid[y][x] = '→'  # Arrow head
                else:
                    grid[y][x] = '─'  # Arrow line
        
        # Print compass
        print("="*60)
        print(f"  🧭 IMU COMPASS - Message #{self.msg_count}")
        print("="*60)
        for row in grid:
            print("  " + ''.join(row))
        
        print("\n" + "="*60)
        print(f"  Heading: {self.compass_heading:6.2f}° {self.compass_dir:2s}")
        print(f"  Yaw:     {self.yaw:6.2f}° (same as heading!)")
        print("="*60)
        print("\n  YES! Yaw = Compass Heading")
        print("  Yaw is rotation around vertical axis = where you're pointing")
        print("\n  Press Ctrl+C to stop")
    
    def imu_callback(self, msg):
        """Process IMU data"""
        self.msg_count += 1
        
        # Extract orientation
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        # Convert to yaw
        self.yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        
        # Convert to compass
        self.compass_heading, self.compass_dir = self.yaw_to_compass(self.yaw)
        
        # Update display every 5 messages (to avoid flicker)
        if self.msg_count % 5 == 0:
            self.draw_ascii_compass()
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\n🛑 Compass stopped")

if __name__ == '__main__':
    try:
        compass = ASCIICompass()
        compass.run()
    except rospy.ROSInterruptException:
        pass
