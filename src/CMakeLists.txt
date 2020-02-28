project(Angular_Correlation LANGUAGES CXX CUDA)

###

add_library(freader STATIC filereader.cpp anglepair.h)

target_compile_features(freader PUBLIC cxx_std_11)

# set_target_properties(freader PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

###

add_library(cuda STATIC sim.cu)

target_compile_features(cuda PUBLIC cxx_std_11)

# set_target_properties(cuda PROPERTIES LINKER_LANGUAGE CUDA)

set_target_properties(cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

###

find_package (Threads)

add_executable(angcorr main.cpp)

target_compile_options(angcorr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
--verbose
--O3
>)


set_property(TARGET angcorr
PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(angcorr PRIVATE ${CMAKE_THREAD_LIBS_INIT} freader cuda)