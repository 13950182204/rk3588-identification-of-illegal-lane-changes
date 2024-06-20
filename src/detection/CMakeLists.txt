aux_source_directory(./ SOURCE)
add_library(Detection STATIC ${SOURCE})

link_directories(/home/cat/Documents/gratuate_project/lib)
target_link_libraries(Detection  PUBLIC gpu_postprocess)


target_include_directories(Detection PUBLIC ./)
target_include_directories(Detection PUBLIC ../common)
target_include_directories(Detection PUBLIC /home/cat/Documents/gratuate_project/3rdparty/librga/include)
target_include_directories(Detection PUBLIC ../gpu_postprocess)

# 设置 "Detection" 静态库的输出路径为 build 目录下的 lib 子目录
set_target_properties(Detection PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY /home/cat/Documents/gratuate_project/lib
)