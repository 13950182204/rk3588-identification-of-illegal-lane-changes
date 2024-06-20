cmake_minimum_required(VERSION 3.4...3.18)
project(gratuate_project)
set(project "gratuate_project")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-g -Wall")

#---------------------------------------------------------------------

#---------------------------------------------------------------------

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#---------------------------------------------------------------------

#---------------------------------------------------------------------


add_subdirectory(src/capture)
add_subdirectory(src/display)
add_subdirectory(src/detection)
add_subdirectory(src/gpu_postprocess)
#include_directories(./lib)
# 添加库文件搜索路径
link_directories(/home/cat/Documents/gratuate_project/lib)

add_executable(${project} main.cpp)

target_link_libraries(${project} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${project} PRIVATE Display)
target_link_libraries(${project} PRIVATE Capture)
target_link_libraries(${project} PRIVATE pthread)
target_link_libraries(${project} PRIVATE Detection)

target_link_libraries(${project} PUBLIC rknnrt)
target_link_libraries(${project} PUBLIC rknn_api)
target_link_libraries(${project} PUBLIC rga)
