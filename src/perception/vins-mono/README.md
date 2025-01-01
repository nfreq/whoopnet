## Build VINS-Mono on ROS2
Clone the repository and colcon build:

```
    cd ~/vins_mono_ws/src
    git clone https://gitee.com/walleve/vins-mono-ros2.git
    cd ..
    colcon build
    source install/setup.bash
```
## Run VINS-Mono on ROS2
Run:

```
    ros2 launch vins_estimator euroc.launch 
    ros2 launch vins_estimator vins_rviz.launch
    ros2 bag play YOUR_PATH_TO_DATASET/MH_01_easy.db3
```
Note:

To convert ROS1 bags to ROS2 bags, you will need to install the rosbags package
> pip install rosbags && echo "export PATH=\"~/.local/bin:\$PATH\"" >> ~/.bashrc && source ~/.bashrc

and run the following command
> rosbags-convert foo.bag