cmake_minimum_required(VERSION 3.13.4)
project(Angular_Correlation LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_subdirectory(src)

# target_compile_options(filereader PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        -ccbin clang-8
#                        >)


# add_library(cuda STATIC [CUDAFILES])

# target_compile_features(cuda PUBLIC cxx_std_11)

# set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(angcorr main.cu)


# target_link_libraries(angcorr PRIVATE cuda)