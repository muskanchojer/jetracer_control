# Docker setup

Every version of this project talks to the robot through ROS 1 Noetic running in a
Docker container. The Jetson Orin Nano runs Ubuntu 22.04, where Noetic doesn't install
natively, so the container is what gives access to the Waveshare hardware.

`start_jetracer_docker.sh` (in this folder) starts that container. It is reconstructed
from my own notes, so check it against your setup before relying on it.

## What the container needs

**1. The image, `jetracer-slam:latest`**
Built on `ros:noetic-robot` (arm64). The original Dockerfile isn't preserved, but the
image held Ubuntu 20.04 + ROS Noetic plus: gmapping, hector-slam, cartographer,
move_base, navigation, rplidar-ros, amcl, rviz, teleop-twist-keyboard, python3-pip.
The control code in this repo only needs ROS itself and the driver below; the mapping
packages were for the LiDAR experiments.

**2. A catkin workspace at `~/jetracer_ws` on the host**
Mounted into the container as `/root/catkin_ws`. Its `src/` holds:

- `jetracer/` - Waveshare's driver package, from
  [waveshare/jetracer_ros](https://github.com/waveshare/jetracer_ros). It contains the
  C++ node that bridges `/cmd_vel` to the RP2040 motor board over serial, and
  publishes `/imu`, `/odom`, `/motor/lvel` and `/motor/rvel`.
- `usb_cam/` - the standard ROS USB camera node.

The host has no ROS, so build the workspace with `catkin_make` from inside the
container.

**3. Devices**
`/dev/ttyACM0` is the motor board and `/dev/video0` is the USB camera. `/dev/ttyACM1`
was the LiDAR in the later experiments.

## Quick check

With the container running, in another terminal:

```bash
docker exec jetracer_ros bash -c "source /root/catkin_ws/devel/setup.bash && rostopic list"
```

You should see `/cmd_vel` and `/imu`. To nudge the car forward (wheels off the ground
first):

```bash
docker exec jetracer_ros bash -c "source /root/catkin_ws/devel/setup.bash && \
  rostopic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1}}' -1"
```
