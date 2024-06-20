
add_library(Display STATIC display.cpp)
target_include_directories(Display PUBLIC ../common)
target_include_directories(Display PUBLIC ./)

# 设置 "Display" 静态库的输出路径为 build 目录下的 lib 子目录
set_target_properties(Display PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY /home/cat/Documents/gratuate_project/lib
)