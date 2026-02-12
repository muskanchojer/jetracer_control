#!/usr/bin/env python3
"""
Visual IMU Compass for JetRacer
Displays a graphical compass with heading arrow
"""

import rospy
from sensor_msgs.msg import Imu
import math
import cv2
import numpy as np

class VisualCompass:
    def __init__(self):
        rospy.init_node('imu_compass_visual')
        
        # Subscribe to IMU topic
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        # Compass display parameters
        self.width = 600
        self.height = 600
        self.center = (self.width // 2, self.height // 2)
        self.radius = 220
        
        # Current heading data
        self.compass_heading = 0.0
        self.compass_dir = "N"
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.accel = [0.0, 0.0, 0.0]
        self.msg_count = 0
        
        # Create window
        cv2.namedWindow('IMU Compass', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('IMU Compass', self.width, self.height)
        
        print("="*50)
        print("🧭 Visual IMU Compass Started")
        print("="*50)
        print("   Displaying graphical compass")
        print("   Press 'q' to quit")
        print("="*50 + "\n")
        
    def yaw_to_compass(self, yaw_deg):
        """Convert yaw angle to compass direction"""
        compass_heading = (yaw_deg + 360) % 360
        
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((compass_heading + 22.5) / 45) % 8
        direction = directions[index]
        
        return compass_heading, direction
    
    def imu_callback(self, msg):
        """Process IMU data"""
        self.msg_count += 1
        
        # Extract orientation (quaternion to euler)
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        # Convert to roll, pitch, yaw
        self.roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        self.pitch = math.degrees(math.asin(2*(w*y - z*x)))
        self.yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        
        # Convert yaw to compass heading
        self.compass_heading, self.compass_dir = self.yaw_to_compass(self.yaw)
        
        # Extract linear acceleration
        self.accel = [msg.linear_acceleration.x, 
                     msg.linear_acceleration.y, 
                     msg.linear_acceleration.z]
    
    def draw_compass(self):
        """Draw the compass visualization"""
        # Create blank image
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Draw outer circle
        cv2.circle(img, self.center, self.radius, (100, 100, 100), 3)
        
        # Draw cardinal directions
        cardinal_points = [
            (0, "N", (0, 0, 255)),      # North - Red
            (90, "E", (200, 200, 200)), # East - Gray
            (180, "S", (200, 200, 200)), # South - Gray
            (270, "W", (200, 200, 200))  # West - Gray
        ]
        
        for angle, label, color in cardinal_points:
            # Calculate position for label
            rad = math.radians(angle - 90)  # -90 to make 0° point up
            x = int(self.center[0] + (self.radius + 40) * math.cos(rad))
            y = int(self.center[1] + (self.radius + 40) * math.sin(rad))
            
            # Draw tick mark
            x_tick_start = int(self.center[0] + (self.radius - 15) * math.cos(rad))
            y_tick_start = int(self.center[1] + (self.radius - 15) * math.sin(rad))
            x_tick_end = int(self.center[0] + (self.radius + 5) * math.cos(rad))
            y_tick_end = int(self.center[1] + (self.radius + 5) * math.sin(rad))
            cv2.line(img, (x_tick_start, y_tick_start), (x_tick_end, y_tick_end), color, 3)
            
            # Draw label
            font_scale = 1.5 if label == "N" else 1.2
            thickness = 3 if label == "N" else 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_BOLD, 
                       font_scale, color, thickness)
        
        # Draw degree markers every 30°
        for angle in range(0, 360, 30):
            if angle % 90 != 0:  # Skip cardinal directions
                rad = math.radians(angle - 90)
                x_start = int(self.center[0] + (self.radius - 10) * math.cos(rad))
                y_start = int(self.center[1] + (self.radius - 10) * math.sin(rad))
                x_end = int(self.center[0] + self.radius * math.cos(rad))
                y_end = int(self.center[1] + self.radius * math.sin(rad))
                cv2.line(img, (x_start, y_start), (x_end, y_end), (150, 150, 150), 1)
        
        # Draw heading arrow
        arrow_length = self.radius - 30
        rad = math.radians(self.compass_heading - 90)  # -90 to make 0° point up
        arrow_x = int(self.center[0] + arrow_length * math.cos(rad))
        arrow_y = int(self.center[1] + arrow_length * math.sin(rad))
        
        # Draw arrow line (thicker)
        cv2.line(img, self.center, (arrow_x, arrow_y), (0, 150, 0), 5)
        
        # Draw arrowhead
        cv2.arrowedLine(img, self.center, (arrow_x, arrow_y), (0, 200, 0), 8, 
                       tipLength=0.2)
        
        # Draw center circle
        cv2.circle(img, self.center, 10, (0, 100, 0), -1)
        
        # Display heading information
        heading_text = f"Heading: {self.compass_heading:.1f}deg {self.compass_dir}"
        cv2.putText(img, heading_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 0), 2)
        
        # Display yaw (for reference)
        yaw_text = f"Yaw: {self.yaw:.1f}deg"
        cv2.putText(img, yaw_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (100, 100, 100), 2)
        
        # Display roll and pitch
        rp_text = f"Roll: {self.roll:.1f}deg  Pitch: {self.pitch:.1f}deg"
        cv2.putText(img, rp_text, (20, self.height - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 100, 100), 1)
        
        # Display acceleration
        accel_text = f"Accel: X={self.accel[0]:.2f} Y={self.accel[1]:.2f} Z={self.accel[2]:.2f}"
        cv2.putText(img, accel_text, (20, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 100, 100), 1)
        
        # Display message count
        msg_text = f"Messages: {self.msg_count}"
        cv2.putText(img, msg_text, (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 100, 100), 1)
        
        return img
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(30)  # 30 Hz update rate
        
        try:
            while not rospy.is_shutdown():
                # Draw compass
                img = self.draw_compass()
                
                # Display
                cv2.imshow('IMU Compass', img)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('IMU Compass', cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                rate.sleep()
                
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            print("\n🛑 Compass stopped")

if __name__ == '__main__':
    try:
        compass = VisualCompass()
        compass.run()
    except rospy.ROSInterruptException:
        pass
