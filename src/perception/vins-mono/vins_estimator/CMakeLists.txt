cmake_minimum_required(VERSION 3.5)
project(vins_estimator)

# Default to C11
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )

target_link_libraries(vins_estimator 
  ${rclcpp_TARGETS} 
  ${cv_bridge_TARGETS} 
  ${std_msgs_TARGETS} 
  ${nav_msgs_TARGETS} 
  ${visualization_msgs_TARGETS} 
  ${geometry_msgs_TARGETS} 
  ${tf2_ros_TARGETS} 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES}
)

install(DIRECTORY 
  launch 
  config
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()