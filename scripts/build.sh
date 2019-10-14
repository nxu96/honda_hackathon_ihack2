#!/bin/bash
cd ~/ros_wrapper_dev
#catkin_make -DPYTHON_VERSION=3.6 -DPYTHON_EXECUTABLE=/usr/bin/python3.6
catkin_make
source devel/setup.bash
