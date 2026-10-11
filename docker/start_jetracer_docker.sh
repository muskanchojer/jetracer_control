#!/bin/bash
# Starts the ROS Noetic container used by every version of this project.
#
# Reconstructed from my own setup notes - the original lived in ~/ on the Jetson.
# Adjust the devices and the workspace path to match your setup.
#
# Host ~/jetracer_ws is mounted as the container's /root/catkin_ws, so the
# Waveshare driver package and usb_cam are built once and reused across runs.
# The container is --rm: it is destroyed on exit and recreated from the image.

docker run -it --rm \
  --privileged \
  --network host \
  --name jetracer_ros \
  --runtime nvidia \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -v $HOME/jetracer_ws:/root/catkin_ws \
  -v /etc/localtime:/etc/localtime:ro \
  --device=/dev/ttyACM0 \
  --device=/dev/ttyACM1 \
  --device=/dev/video0 \
  jetracer-slam:latest \
  /bin/bash
