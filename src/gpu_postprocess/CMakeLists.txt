## gpu_postprocess
####################
### opencl wrapper
set(OPENCL_PATH  /home/cat/Documents/gratuate_project/3rdparty/opencl/libopencl-stub)

set(libopencl_SRCS
    ${OPENCL_PATH}/src/libopencl.cc
)
add_definitions(-fPIC)
add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

add_library(OpenCL SHARED ${libopencl_SRCS})
include_directories(${OPENCL_PATH}/include)

set(gpu_postprocessSRC  gpu_compose_impl.cc
                    workflow.cc
                    opencl_workspace.cc
                    opencl_memory.cc
                    opencl_wrapper.cc)
                
add_library(gpu_postprocess STATIC ${gpu_postprocessSRC})

# /home/cat/Documents/gratuate_project/lib/libOpenCL.so
target_link_libraries(gpu_postprocess  /home/cat/Documents/gratuate_project/src/gpu_postprocess/libOpenCL.so)


target_include_directories(gpu_postprocess PUBLIC ${OPENCL_PATH}/include)

target_include_directories(gpu_postprocess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(gpu_postprocess PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY /home/cat/Documents/gratuate_project/lib
)





