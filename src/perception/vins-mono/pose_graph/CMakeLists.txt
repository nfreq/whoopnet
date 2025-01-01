cmake_minimum_required(VERSION 3.5)
project(pose_graph)

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
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(camera_model REQUIRED)
include_directories(${camera_model_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(pose_graph
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

target_link_libraries(pose_graph 
  ${rclcpp_TARGETS} 
  ${nav_msgs_TARGETS} 
  ${std_msgs_TARGETS} 
  ${sensor_msgs_TARGETS} 
  ${cv_bridge_TARGETS} 
  ${visualization_msgs_TARGETS} 
  ${camera_model_LIBRARIES}
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES}
) 

install(DIRECTORY 
  support_files 
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()