#doesnt work - libbiff.so.1: cannot open shared object file: No such file or directory error issue 
#ros-noetic with ubuntu 22.04 issue
#cv2.imshow doesn't work in Docker)

import rospy
from sensor_msgs.msg import Image

class CameraViewer:
    def __init__(self):
        rospy.init_node('camera_viewer')
        
        self.frame_count = 0
        self.last_print_time = rospy.Time.now()
        
        # Subscribe to USB camera
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        
        print("="*50)
        print("📷 Camera Monitor Started")
        print("="*50)
        print("   Monitoring camera feed from /usb_cam/image_raw")
        print("")
        print("   To VIEW the camera feed, open a NEW terminal and run:")
        print("   docker exec -it jetracer_ros bash -c \\")
        print("     'source /root/catkin_ws/devel/setup.bash && \\")
        print("     rosrun image_view image_view image:=/usb_cam/image_raw'")
        print("")
        print("   Press Ctrl+C to stop")
        print("="*50 + "\n")
        
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            self.frame_count += 1
            
            # Print status every second
            current_time = rospy.Time.now()
            if (current_time - self.last_print_time).to_sec() >= 1.0:
                print(f"\r📸 Receiving frames: {self.frame_count} total | "
                      f"Size: {msg.width}x{msg.height} | "
                      f"Encoding: {msg.encoding}", end='', flush=True)
                self.last_print_time = current_time
                
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
    
    def run(self):
        """Main loop"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping camera monitor...")
            print(f"   Total frames received: {self.frame_count}")

if __name__ == '__main__':
    viewer = CameraViewer()
    viewer.run()