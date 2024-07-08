cmake_minimum_required(VERSION 3.0.2)
project(ip_camera_processing_single)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/ip_camera_processor_single.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
