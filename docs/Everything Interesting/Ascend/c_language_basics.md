# 语言基础

还想额外问一句：大家都是怎么学`C/C++`的呢？

大多数教科书的逻辑是这样的：

1. `printf("Hello, World!")`、编译的流程
2. 单变量定义、基本运算、输入输出
3. 函数的概念
4. 数组
5. 指针的原理、以指针为参数的函数、数组名作指针
6. 动态内存：`malloc`与`free`
7. 标准库的文件操作
8. 标准库中其他内容
9. 数据结构与算法入门

但是工程上并不是这么做的。

## `CMake`

> CMake is an open-source, cross-platform family of tools designed to build, test and package software. CMake is used to **control the software compilation process** using **simple platform** and **compiler independent configuration files**, and generate native makefiles and workspaces that can be used in the compiler environment of your choice. The suite of CMake tools were created by Kitware in response to the need for a powerful, cross-platform build environment for open-source projects such as ITK and VTK. https://cmake.org/

### 教程

https://cmake.org/cmake/help/book/mastering-cmake/

### 原理

CMakeLists –> `CMake Software` –> configuration file for different toolchains

~~用`CMake`写一个…`Hello World!`?~~

### 语法

`CMake`的语法来源于~~我们数据结构课上都学过也都不知道有什么用的一种数据结构~~广义表。

### 推荐工程规范

这里以``samples/cplusplus/level2_simple_inference/2_object_detection/YOLOV3_coco_detection_picture`工程为例。

#### 目录结构

```
samples/cplusplus/level2_simple_inference/2_object_detection/YOLOV3_coco_detection_picture
├── CMakeLists.txt                    # CMake 工程
├── README.md
├── README_CN.md
├── YOLOV3_coco_detection_picture.iml # IDEA 的工程配置文件（可忽略）
├── data                              # 模型demo使用的数据
├── inc                               # 头文件
│   ├── dvpp_jpegd.h
│   ├── dvpp_process.h
│   ├── dvpp_resize.h
│   ├── model_process.h
│   ├── object_detect.h
│   └── utils.h
├── model                             # 模型文件
├── scripts                           # 测试/运行所用脚本文件
│   ├── sample_build.sh
│   └── sample_run.sh
└── src                               # 源代码
    ├── CMakeLists.txt
    ├── acl.json
    ├── dvpp_jpegd.cpp
    ├── dvpp_process.cpp
    ├── dvpp_resize.cpp
    ├── main.cpp
    ├── model_process.cpp
    ├── object_detect.cpp
    └── utils.cpp
```

### 顶层`CMakeLists.txt`

```cmake
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(classification)

add_subdirectory("./src")
```

### `src`目录下的 CMakeLists.txt

```cmake
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(classification)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)

if (NOT DEFINED ENV{INSTALL_DIR})
    MESSAGE(FATAL_ERROR "Not Defined INSTALL_DIR")
endif()

if (NOT DEFINED ENV{THIRDPART_PATH})
    MESSAGE(FATAL_ERROR "Not Defined THIRDPART_PATH")
endif()

if (NOT DEFINED ENV{CPU_ARCH})
    MESSAGE(FATAL_ERROR "Not Defined CPU_ARCH")
endif()

add_definitions(-DENABLE_DVPP_INTERFACE)
list(APPEND COMMON_DEPEND_LIB avcodec avformat avdevice avutil swresample avfilter swscale)
if ($ENV{CPU_ARCH} MATCHES "aarch64")
    if(EXISTS "$ENV{INSTALL_DIR}/driver/libmedia_mini.so")
        list(APPEND COMMON_DEPEND_LIB media_mini ascend_hal c_sec mmpa)
        add_definitions(-DENABLE_BOARD_CAMARE)
        message(STATUS "arch: arm")
    endif()
endif()

# Header path
include_directories(
    $ENV{INSTALL_DIR}/acllib/include/
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    $ENV{INSTALL_DIR}/runtime/lib64/stub
)

add_executable(main
        utils.cpp
        model_process.cpp
        object_detect.cpp
        dvpp_process.cpp
        dvpp_resize.cpp
        dvpp_jpegd.cpp
        main.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl acl_dvpp stdc++ opencv_core opencv_imgproc opencv_imgcodecs dl rt)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
```

https://cmake.org/cmake/help/latest/command/target_link_directories.html
