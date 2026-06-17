cmake_minimum_required(VERSION 3.16)
project(centerpoint_onnx_mnn_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# Build options
# ------------------------------------------------------------------
option(USE_ONNXRUNTIME "Build the ONNX Runtime backend" ON)
option(USE_MNN         "Build the MNN backend"          OFF)
option(USE_OPENCV      "Enable OpenCV BEV visualisation" ON)

# ------------------------------------------------------------------
# Core library: shared pre/post-processing + runtime-engine interface.
# Backend sources are added conditionally below.
# ------------------------------------------------------------------
add_library(centerpoint_core
    src/centerpoint_config.cpp
    src/centerpoint_detector.cpp
    src/visualization.cpp
)
target_include_directories(centerpoint_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(centerpoint_core PRIVATE -Wall -Wextra)

# ------------------------------------------------------------------
# yaml-cpp (required — config loader)
# ------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)
target_link_libraries(centerpoint_core PUBLIC yaml-cpp)

# ------------------------------------------------------------------
# ONNX Runtime backend
# ------------------------------------------------------------------
if(USE_ONNXRUNTIME)
    set(ONNXRUNTIME_ROOT "$ENV{ORT_ROOT}" CACHE PATH "Path to ONNX Runtime install")
    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}/include
              /usr/local/include/onnxruntime
              /usr/include/onnxruntime
              /usr/local/include
        REQUIRED)
    find_library(ORT_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/lib /usr/local/lib /usr/lib
        REQUIRED)
    message(STATUS "ONNX Runtime: ${ORT_LIB}")
    target_sources(centerpoint_core PRIVATE src/onnx_engine.cpp)
    target_include_directories(centerpoint_core PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(centerpoint_core PRIVATE ${ORT_LIB})
    target_compile_definitions(centerpoint_core PUBLIC HAVE_ONNXRUNTIME=1)
endif()

# ------------------------------------------------------------------
# MNN backend
# ------------------------------------------------------------------
if(USE_MNN)

    ## 已编译好的mnn库
    # set(MNN_ROOT "$ENV{MNN_ROOT}" CACHE PATH "Path to MNN install")
    # find_path(MNN_INCLUDE_DIR
    #     NAMES MNN/Interpreter.hpp
    #     PATHS ${MNN_ROOT}/include /usr/local/include /usr/include
    #     REQUIRED)
    # find_library(MNN_LIB
    #     NAMES MNN
    #     PATHS ${MNN_ROOT}/lib ${MNN_ROOT}/build /usr/local/lib /usr/lib
    #     REQUIRED)

    ## 源码编译的mnn库
    set(DEFAULT_MNN_ROOT "/home/panguofeng/project/MNN")
    set(MNN_ROOT "${DEFAULT_MNN_ROOT}" CACHE PATH "Path to MNN")
    message(STATUS "MNN root: ${MNN_ROOT}")
    find_path(MNN_INCLUDE_DIR
        NAMES MNN/Interpreter.hpp
        PATHS ${MNN_ROOT}/include
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(MNN_LIB
        NAMES MNN
        PATHS ${MNN_ROOT}/lib ${MNN_ROOT}/build
        NO_DEFAULT_PATH
        REQUIRED)
    message(STATUS "MNN: ${MNN_LIB}")
    target_sources(centerpoint_core PRIVATE src/mnn_engine.cpp)
    target_include_directories(centerpoint_core PRIVATE ${MNN_INCLUDE_DIR})
    target_link_libraries(centerpoint_core PRIVATE ${MNN_LIB})
    target_compile_definitions(centerpoint_core PUBLIC HAVE_MNN=1)
endif()

if(NOT USE_ONNXRUNTIME AND NOT USE_MNN)
    message(FATAL_ERROR "Enable at least one backend: -DUSE_ONNXRUNTIME=ON or -DUSE_MNN=ON")
endif()

# ------------------------------------------------------------------
# OpenCV visualisation (optional)
# ------------------------------------------------------------------
if(USE_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV ${OpenCV_VERSION} found — BEV visualisation enabled")
        target_include_directories(centerpoint_core PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(centerpoint_core PRIVATE ${OpenCV_LIBS})
        target_compile_definitions(centerpoint_core PUBLIC HAVE_OPENCV=1)
    else()
        message(WARNING "OpenCV not found — building without BEV visualisation")
    endif()
endif()

# ------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------
add_executable(centerpoint src/main.cpp)
target_link_libraries(centerpoint PRIVATE centerpoint_core)
