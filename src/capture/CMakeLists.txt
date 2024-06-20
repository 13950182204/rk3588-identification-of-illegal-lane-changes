add_library(Capture STATIC capture.cpp)
target_include_directories(Capture PUBLIC ../common)
target_include_directories(Capture PUBLIC ./)

set_target_properties(Capture PROPERTIES ARCHIVE_OUTPUT_DIRECTORY /home/cat/Documents/gratuate_project/lib)