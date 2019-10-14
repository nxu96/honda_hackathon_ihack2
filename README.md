# Foresight Prototype

This repository contains the code files and supporting documentation/presentation/demos of the Foresight prototype. Foresight is a system for the Honda MobilityHacks events (October 2019). It's aim is to provide enchanced perception capabilities for both human and autonomous vehicles alike by broadcasting perception information between connected vehicles, providing advanced notice of road information otherwise not in the line of sight.

# Setup

The ros_wrapper package runs on both ROS Kinetic and ROS Melodic. `gps_common` is a package dependency needed, as well as several sensor messages you'll get errors about if they're not installed.

On the Python end, most everything is pip-installable and will likewise give a useful error message.

# Running

The current model is not real-time, so in order to accomodate that, the pipeline was split in 2 so things could be written and read from file, as otherwise the ZMQ buffers would overflow and crash the HUD.

A prerequiste is to build the package with `./scripts/build.sh`

First, in `heads_up_display.py`, comment out lines in `Display`'s `run(self)` function not relevant to reading in ZMQ data and writing to a pickle file. Then, in 3 terminals, perform the following:
- `roslaunch launch/ros_wrapper.launch`
- `./src/ros_wrapper/scripts/ros_wrapper.py`
- `python3 heads_up_display.py`

Following that, swap comments in `run(self)` to read from a pickle file instead, as well as running the main logic, then execute:
- `python3 heads_up_display.py`

The output videos will be written as `back_video.avi` and `side_video.avi`

