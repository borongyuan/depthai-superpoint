cmake_minimum_required(VERSION 3.4)

project(depthai_superpoint)

find_package(OpenCV REQUIRED)
find_package(depthai CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/depthai_superpoint.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE depthai::opencv ${OpenCV_LIBS})