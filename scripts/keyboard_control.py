#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import sys
import termios
import tty

def getKey():
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    
    rospy.init_node("simple_keyboard_teleop")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    
    twist = Twist()
    
    linear_speed = 0.3
    angular_speed = 0.8
    
    current_linear = 0.0
    current_angular = 0.0
    
    print("""
========================================
    JetRacer Keyboard Control
========================================
  W - Forward      S - Backward
  A - Steer Left   D - Steer Right
  X - Center       SPACE - Stop All
  
  + - Faster       - - Slower
  Q - Quit
========================================
Current Speed: Linear=%.2f Angular=%.2f
""" % (linear_speed, angular_speed))
    
    rate = rospy.Rate(20)
    
    try:
        while not rospy.is_shutdown():
            key = getKey()
            
            if key == "w" or key == "W": 
                current_linear = linear_speed
                print("Forward (steering:  %.2f)" % current_angular)
                
            elif key == "s" or key == "S":
                current_linear = -linear_speed
                print("Backward (steering: %.2f)" % current_angular)
            
            elif key == "a" or key == "A":
                current_angular = angular_speed
                print("Steer Left (speed: %.2f)" % current_linear)
                
            elif key == "d" or key == "D":
                current_angular = -angular_speed
                print("Steer Right (speed: %.2f)" % current_linear)
            
            elif key == "x" or key == "X":
                current_angular = 0.0
                print("Steering Centered")
            
            elif key == " ":
                current_linear = 0.0
                current_angular = 0.0
                print("FULL STOP")
            
            elif key == "+" or key == "=":
                linear_speed = min(linear_speed + 0.1, 1.0)
                angular_speed = min(angular_speed + 0.2, 2.0)
                print("Speed UP: Linear=%.2f Angular=%.2f" % (linear_speed, angular_speed))
                
            elif key == "-" or key == "_":
                linear_speed = max(linear_speed - 0.1, 0.1)
                angular_speed = max(angular_speed - 0.2, 0.2)
                print("Speed DOWN:  Linear=%.2f Angular=%.2f" % (linear_speed, angular_speed))
            
            elif key == "q" or key == "Q":
                print("Quitting...")
                break
                
            elif key == "\x03": 
                break
            
            twist.linear.x = current_linear
            twist.angular.z = current_angular
            pub.publish(twist)
            
            rate.sleep()
            
    except Exception as e:
        print(e)
        
    finally:
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        termios.tcsetattr(sys. stdin, termios.TCSADRAIN, settings)
        print("Robot stopped")
