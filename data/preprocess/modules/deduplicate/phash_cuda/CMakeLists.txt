cmake_minimum_required(VERSION 3.18)
project(phash_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES
    src/phash_cuda.cu
    src/phash_cuda_pybind.cpp
)

# Python binding
pybind11_add_module(_phash_cuda_py ${SOURCES})

target_link_libraries(_phash_cuda_py
    PRIVATE
        CUDA::cudart
        CUDA::nvjpeg
        Threads::Threads
)

target_include_directories(_phash_cuda_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(_phash_cuda_py PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

install(TARGETS _phash_cuda_py DESTINATION phash_cuda)
