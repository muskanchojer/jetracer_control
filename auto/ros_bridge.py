#!/usr/bin/env python3
"""
ROS Topic Bridge - Subscribe to ROS topics from Docker
Uses streaming rostopic commands via docker exec for low-latency odometry data
Optimized for Jetson Orin Nano - uses continuous streaming instead of polling
"""

import subprocess
import json
import re
import time
import threading
import logging
from collections import deque


class ROSTopicBridge:
    """
    Bridge to access ROS topics from outside Docker
    Uses docker exec with continuous streaming for low-latency updates
    Optimized for high-frequency odometry updates (50+ Hz)
    Now includes IMU data for sensor fusion and accurate waypoint detection
    """
    
    def __init__(self, container_name='jetracer_ros'):
        self.container_name = container_name
        self.logger = logging.getLogger(__name__)
        
        # Odometry data with thread-safe access
        self._lock = threading.Lock()
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.odom_timestamp = 0.0
        
        # Velocity data (often available in odom)
        self.vx = 0.0
        self.vy = 0.0
        self.vtheta = 0.0
        
        # IMU data for sensor fusion
        self._imu_lock = threading.Lock()
        self.imu_accel_x = 0.0
        self.imu_accel_y = 0.0
        self.imu_accel_z = 0.0
        self.imu_gyro_x = 0.0
        self.imu_gyro_y = 0.0
        self.imu_gyro_z = 0.0
        self.imu_orientation_x = 0.0
        self.imu_orientation_y = 0.0
        self.imu_orientation_z = 0.0
        self.imu_orientation_w = 1.0
        self.imu_timestamp = 0.0
        
        # IMU-integrated position (dead reckoning from IMU)
        self.imu_integrated_x = 0.0
        self.imu_integrated_y = 0.0
        self.imu_velocity_x = 0.0
        self.imu_velocity_y = 0.0
        self.last_imu_time = 0.0
        
        # Thread control
        self.running = False
        self.thread = None
        self.imu_thread = None
        self.process = None
        self.imu_process = None
        
        # Performance tracking
        self.update_count = 0
        self.last_update_rate_time = time.time()
        self.update_rate = 0.0
        self.imu_update_count = 0
        self.imu_update_rate = 0.0
        
        # Message buffer for parsing
        self._msg_buffer = ""
        self._imu_msg_buffer = ""
        
        self.logger.info(f"ROS Topic Bridge initialized for container: {container_name}")
    
    def start(self):
        """Start listening to ROS topics with continuous streaming"""
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        # Start IMU streaming in parallel
        self.imu_thread = threading.Thread(target=self._imu_stream_loop, daemon=True)
        self.imu_thread.start()
        self.logger.info("Started ROS topic listener (streaming mode with IMU)")
    
    def stop(self):
        """Stop listening"""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                self.process.kill()
        if self.imu_process:
            try:
                self.imu_process.terminate()
                self.imu_process.wait(timeout=1.0)
            except:
                self.imu_process.kill()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.imu_thread:
            self.imu_thread.join(timeout=2.0)
        self.logger.info("Stopped ROS topic listener")
    
    def _stream_loop(self):
        """
        Continuous streaming loop - much faster than polling with echo -n 1
        Uses rostopic echo without -n flag for continuous output
        """
        while self.running:
            try:
                # Start continuous streaming process
                cmd = [
                    "docker", "exec", self.container_name,
                    "bash", "-c",
                    "source /root/catkin_ws/devel/setup.bash && rostopic echo /odom"
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                self._msg_buffer = ""
                
                # Read lines continuously
                while self.running and self.process.poll() is None:
                    line = self.process.stdout.readline()
                    if line:
                        self._process_line(line)
                
            except Exception as e:
                self.logger.error(f"Stream error: {e}")
                time.sleep(0.5)  # Brief pause before reconnecting
            
            if self.running:
                self.logger.debug("Reconnecting to /odom stream...")
    
    def _imu_stream_loop(self):
        """
        Continuous streaming loop for IMU data
        Uses rostopic echo on /imu topic for continuous output
        """
        while self.running:
            try:
                # Start continuous streaming process for IMU
                cmd = [
                    "docker", "exec", self.container_name,
                    "bash", "-c",
                    "source /root/catkin_ws/devel/setup.bash && rostopic echo /imu"
                ]
                
                self.imu_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                self._imu_msg_buffer = ""
                
                # Read lines continuously
                while self.running and self.imu_process.poll() is None:
                    line = self.imu_process.stdout.readline()
                    if line:
                        self._process_imu_line(line)
                
            except Exception as e:
                self.logger.error(f"IMU stream error: {e}")
                time.sleep(0.5)  # Brief pause before reconnecting
            
            if self.running:
                self.logger.debug("Reconnecting to /imu stream...")
    
    def _process_imu_line(self, line):
        """Process a single line from IMU rostopic echo output"""
        self._imu_msg_buffer += line
        
        # Check if we have a complete message (ends with ---)
        if line.strip() == "---":
            self._parse_imu_message(self._imu_msg_buffer)
            self._imu_msg_buffer = ""
            
            # Update rate tracking
            self.imu_update_count += 1
            now = time.time()
            elapsed = now - self.last_update_rate_time
            if elapsed >= 1.0:
                self.imu_update_rate = self.imu_update_count / elapsed
                self.imu_update_count = 0
    
    def _parse_imu_message(self, msg_text):
        """Parse rostopic echo output for IMU data and integrate position"""
        import math
        try:
            lines = msg_text.split('\n')
            
            # IMU data
            accel_x = accel_y = accel_z = 0.0
            gyro_x = gyro_y = gyro_z = 0.0
            orient_x = orient_y = orient_z = 0.0
            orient_w = 1.0
            
            in_orientation = False
            in_angular_velocity = False
            in_linear_acceleration = False
            
            for line in lines:
                stripped = line.strip()
                
                # Orientation parsing
                if 'orientation:' in line:
                    in_orientation = True
                    continue
                elif in_orientation:
                    if stripped.startswith('x:'):
                        orient_x = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        orient_y = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        orient_z = float(stripped.split(':')[1])
                    elif stripped.startswith('w:'):
                        orient_w = float(stripped.split(':')[1])
                        in_orientation = False
                
                # Angular velocity parsing
                if 'angular_velocity:' in line:
                    in_angular_velocity = True
                    continue
                elif in_angular_velocity:
                    if stripped.startswith('x:'):
                        gyro_x = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        gyro_y = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        gyro_z = float(stripped.split(':')[1])
                        in_angular_velocity = False
                
                # Linear acceleration parsing
                if 'linear_acceleration:' in line:
                    in_linear_acceleration = True
                    continue
                elif in_linear_acceleration:
                    if stripped.startswith('x:'):
                        accel_x = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        accel_y = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        accel_z = float(stripped.split(':')[1])
                        in_linear_acceleration = False
            
            current_time = time.time()
            
            with self._imu_lock:
                # Store raw IMU data
                self.imu_accel_x = accel_x
                self.imu_accel_y = accel_y
                self.imu_accel_z = accel_z
                self.imu_gyro_x = gyro_x
                self.imu_gyro_y = gyro_y
                self.imu_gyro_z = gyro_z
                self.imu_orientation_x = orient_x
                self.imu_orientation_y = orient_y
                self.imu_orientation_z = orient_z
                self.imu_orientation_w = orient_w
                
                # Integrate IMU for position estimation (dead reckoning)
                if self.last_imu_time > 0:
                    dt = current_time - self.last_imu_time
                    if dt > 0 and dt < 0.5:  # Sanity check on dt
                        # Get orientation (yaw) from quaternion
                        siny_cosp = 2.0 * (orient_w * orient_z + orient_x * orient_y)
                        cosy_cosp = 1.0 - 2.0 * (orient_y * orient_y + orient_z * orient_z)
                        yaw = math.atan2(siny_cosp, cosy_cosp)
                        
                        # Remove gravity from acceleration (assuming z is up)
                        # and transform to world frame
                        # Simple approach: use only x,y acceleration in body frame
                        # Transform body-frame accel to world frame using yaw
                        accel_world_x = accel_x * math.cos(yaw) - accel_y * math.sin(yaw)
                        accel_world_y = accel_x * math.sin(yaw) + accel_y * math.cos(yaw)
                        
                        # Apply low-pass filter to reduce noise (simple exponential filter)
                        alpha = 0.3  # Filter coefficient
                        
                        # Integrate acceleration to get velocity
                        self.imu_velocity_x += accel_world_x * dt * alpha
                        self.imu_velocity_y += accel_world_y * dt * alpha
                        
                        # Apply velocity decay to reduce drift
                        decay = 0.98
                        self.imu_velocity_x *= decay
                        self.imu_velocity_y *= decay
                        
                        # Integrate velocity to get position
                        self.imu_integrated_x += self.imu_velocity_x * dt
                        self.imu_integrated_y += self.imu_velocity_y * dt
                
                self.last_imu_time = current_time
                self.imu_timestamp = current_time
                
        except Exception as e:
            self.logger.debug(f"Error parsing IMU message: {e}")

    def _process_line(self, line):
        """Process a single line from rostopic echo output"""
        self._msg_buffer += line
        
        # Check if we have a complete message (ends with ---)
        if line.strip() == "---":
            self._parse_odom_message(self._msg_buffer)
            self._msg_buffer = ""
            
            # Update rate tracking
            self.update_count += 1
            now = time.time()
            elapsed = now - self.last_update_rate_time
            if elapsed >= 1.0:
                self.update_rate = self.update_count / elapsed
                self.update_count = 0
                self.last_update_rate_time = now
    
    def _parse_odom_message(self, msg_text):
        """Parse rostopic echo output for odometry - optimized parsing"""
        try:
            # Extract position x, y using faster parsing
            lines = msg_text.split('\n')
            
            x = y = z = None
            qx = qy = qz = qw = 0.0
            vx = vy = vtheta = 0.0
            
            in_position = False
            in_orientation = False
            in_twist_linear = False
            in_twist_angular = False
            position_done = False
            
            for line in lines:
                stripped = line.strip()
                
                # Position parsing
                if 'position:' in line and not position_done:
                    in_position = True
                    continue
                elif in_position:
                    if stripped.startswith('x:'):
                        x = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        y = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        z = float(stripped.split(':')[1])
                        in_position = False
                        position_done = True
                
                # Orientation parsing
                if 'orientation:' in line:
                    in_orientation = True
                    continue
                elif in_orientation:
                    if stripped.startswith('x:'):
                        qx = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        qy = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        qz = float(stripped.split(':')[1])
                    elif stripped.startswith('w:'):
                        qw = float(stripped.split(':')[1])
                        in_orientation = False
                
                # Twist linear parsing (velocity)
                if 'linear:' in line and 'twist' not in line.lower():
                    in_twist_linear = True
                    continue
                elif in_twist_linear:
                    if stripped.startswith('x:'):
                        vx = float(stripped.split(':')[1])
                    elif stripped.startswith('y:'):
                        vy = float(stripped.split(':')[1])
                    elif stripped.startswith('z:'):
                        in_twist_linear = False
                
                # Twist angular parsing
                if 'angular:' in line:
                    in_twist_angular = True
                    continue
                elif in_twist_angular:
                    if stripped.startswith('z:'):
                        vtheta = float(stripped.split(':')[1])
                        in_twist_angular = False
            
            if x is not None and y is not None:
                # Convert quaternion to yaw
                import math
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                theta = math.atan2(siny_cosp, cosy_cosp)
                
                with self._lock:
                    self.odom_x = x
                    self.odom_y = y
                    self.odom_theta = theta
                    self.vx = vx
                    self.vy = vy
                    self.vtheta = vtheta
                    self.odom_timestamp = time.time()
                
        except Exception as e:
            self.logger.debug(f"Error parsing odom message: {e}")
    
    def get_odometry(self):
        """
        Get current odometry (thread-safe)
        Returns: (x, y, theta, timestamp)
        """
        with self._lock:
            return self.odom_x, self.odom_y, self.odom_theta, self.odom_timestamp
    
    def get_velocity(self):
        """
        Get current velocity (thread-safe)
        Returns: (vx, vy, vtheta)
        """
        with self._lock:
            return self.vx, self.vy, self.vtheta
    
    def get_update_rate(self):
        """Get current odometry update rate in Hz"""
        return self.update_rate

    def get_imu_data(self):
        """
        Get current IMU data (thread-safe)
        Returns: dict with accel, gyro, orientation, and integrated position
        """
        with self._imu_lock:
            return {
                'accel': (self.imu_accel_x, self.imu_accel_y, self.imu_accel_z),
                'gyro': (self.imu_gyro_x, self.imu_gyro_y, self.imu_gyro_z),
                'orientation': (self.imu_orientation_x, self.imu_orientation_y, 
                               self.imu_orientation_z, self.imu_orientation_w),
                'integrated_position': (self.imu_integrated_x, self.imu_integrated_y),
                'integrated_velocity': (self.imu_velocity_x, self.imu_velocity_y),
                'timestamp': self.imu_timestamp
            }
    
    def get_imu_integrated_position(self):
        """
        Get IMU-integrated position estimate (dead reckoning)
        Returns: (x, y, timestamp)
        """
        with self._imu_lock:
            return self.imu_integrated_x, self.imu_integrated_y, self.imu_timestamp
    
    def reset_imu_integration(self, x=0.0, y=0.0):
        """Reset IMU integration to a known position (e.g., sync with odometry)"""
        with self._imu_lock:
            self.imu_integrated_x = x
            self.imu_integrated_y = y
            self.imu_velocity_x = 0.0
            self.imu_velocity_y = 0.0
    
    def get_fused_position(self, odom_weight=0.7, imu_weight=0.3):
        """
        Get sensor-fused position combining odometry and IMU
        Uses weighted average for robust position estimation
        
        Args:
            odom_weight: Weight for odometry (default 0.7)
            imu_weight: Weight for IMU integration (default 0.3)
        
        Returns: (x, y, theta, confidence, timestamp)
        """
        import math
        
        with self._lock:
            odom_x, odom_y, odom_theta = self.odom_x, self.odom_y, self.odom_theta
            odom_ts = self.odom_timestamp
        
        with self._imu_lock:
            imu_x, imu_y = self.imu_integrated_x, self.imu_integrated_y
            imu_ts = self.imu_timestamp
            # Get yaw from IMU orientation
            orient_x, orient_y = self.imu_orientation_x, self.imu_orientation_y
            orient_z, orient_w = self.imu_orientation_z, self.imu_orientation_w
        
        # Calculate IMU yaw
        siny_cosp = 2.0 * (orient_w * orient_z + orient_x * orient_y)
        cosy_cosp = 1.0 - 2.0 * (orient_y * orient_y + orient_z * orient_z)
        imu_theta = math.atan2(siny_cosp, cosy_cosp)
        
        # Check data freshness
        now = time.time()
        odom_fresh = (now - odom_ts) < 0.5 if odom_ts > 0 else False
        imu_fresh = (now - imu_ts) < 0.2 if imu_ts > 0 else False
        
        # Adjust weights based on data freshness
        if not odom_fresh and imu_fresh:
            # Rely more on IMU if odometry is stale
            odom_weight, imu_weight = 0.2, 0.8
        elif not imu_fresh and odom_fresh:
            # Rely more on odometry if IMU is stale
            odom_weight, imu_weight = 0.9, 0.1
        elif not odom_fresh and not imu_fresh:
            # Both stale - low confidence
            confidence = 0.1
            return odom_x, odom_y, odom_theta, confidence, max(odom_ts, imu_ts)
        
        # Normalize weights
        total_weight = odom_weight + imu_weight
        odom_weight /= total_weight
        imu_weight /= total_weight
        
        # Fused position
        fused_x = odom_x * odom_weight + imu_x * imu_weight
        fused_y = odom_y * odom_weight + imu_y * imu_weight
        
        # For theta, prefer IMU as it's typically more accurate for orientation
        # Use circular mean for angle averaging
        fused_theta = math.atan2(
            odom_weight * math.sin(odom_theta) + imu_weight * math.sin(imu_theta),
            odom_weight * math.cos(odom_theta) + imu_weight * math.cos(imu_theta)
        )
        
        # Calculate confidence based on sensor agreement
        position_diff = math.sqrt((odom_x - imu_x)**2 + (odom_y - imu_y)**2)
        # High confidence if sensors agree, lower if they diverge
        confidence = max(0.1, 1.0 - position_diff / 2.0)  # Drops to 0.1 at 1.8m disagreement
        
        return fused_x, fused_y, fused_theta, confidence, max(odom_ts, imu_ts)


if __name__ == '__main__':
    # Test the bridge
    import math
    logging.basicConfig(level=logging.INFO)
    
    bridge = ROSTopicBridge()
    bridge.start()
    
    print("Listening to /odom topic from Docker (streaming mode)...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            x, y, theta, ts = bridge.get_odometry()
            vx, vy, vtheta = bridge.get_velocity()
            rate = bridge.get_update_rate()
            if ts > 0:
                print(f"Pos: x={x:.3f}m, y={y:.3f}m, θ={math.degrees(theta):.1f}° | "
                      f"Vel: {vx:.2f}m/s, ω={vtheta:.2f}rad/s | Rate: {rate:.1f}Hz")
            else:
                print("Waiting for odometry data...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        bridge.stop()
