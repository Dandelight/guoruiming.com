# 从源码开始编译`MindSpore`

## 获取源码

`MindSpore`的源代码托管在`Gitee`上，按照以下命令获取

```bash
git clone https://gitee.com/mindspore/mindspore
# git clone git@gitee.com:mindspore/mindspore # 据说ssh会快一些？
```

## 配置环境

首先要装好`miniconda`、`git`。

### python

为了简化配置我们选用[Anaconda](https://anaconda.com/)

```bash
conda create -n ms-build python=3.9.0
conda activate ms-build
pip install -U pip
```

### `cmake`

`MindSpore`及其众多依赖以[`CMake`](https://cmake.org/)为构建工具。此处使用[pip](https://github.com/pypa/pip)进行安装

```bash
pip install cmake
```

### CUDA

如果你拥有很不错的 NVIDIA GPU，需要添加 CUDA 支持，安装方法见：https://developer.nvidia.com/cuda-downloads，此处选用x86_64 Ubuntu 18.04 网络安装（deb[network]），好处是可以一个`apt update && apt upgrade -y`把 CUDA 升级到最新

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

### CUDNN

顾名思义，cuDNN 是一个用于加速深度神经网络训练的 CUDA 库。MindSpore 依赖这个库。下载方式见https://developer.nvidia.cn/rdp/cudnn-download，可能需要注册NVIDIA开发者账号，可以微信登录。**注意cuDNN版本一定要和CUDA版本对应！**此处安装cuDNN v8.3.1 (November 22nd, 2021), for CUDA 10.2

```bash
sudo apt install libcudnn-dev # libcudnn-samples # 示例程序，可以用于学习
```

### tclsh

tcl 的解释器，依赖

```bash
sudo apt install tclsh
```

### libnuma

CPU 核心绑定，依赖

```bash
sudo apt install libnuma-dev
```

### 环境变量

```bash
export CC=`which gcc`
export CXX=`which g++`
```

这两个环境变量会被`flatbuffer`调用，参考：https://gitee.com/mindspore/mindspore/issues/I4ODSV

## 编译配置

打开`scripts/build`文件夹，编辑`default_options.sh`。在该版本还没有找到文档，只能看名字了。

```bash
#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# shellcheck disable=SC2034

set -e

init_default_options()
{
  # Init default values of build options
  export THREAD_NUM=12                 # 视计算机核心数
  export DEBUG_MODE="off"
  VERBOSE=""
  export ENABLE_SECURITY="off"
  export ENABLE_COVERAGE="off"
  export RUN_TESTCASES="on"            # 运行测试样例
  export RUN_CPP_ST_TESTS="on"
  export ENABLE_BACKEND=""
  export TRAIN_MODE="INFER"
  export ENABLE_ASAN="off"
  export ENABLE_PROFILE="off"
  export INC_BUILD="off"              # 不知道为啥开了这个会报错，什么cache之类的
  export ENABLE_TIMELINE="off"
  export ENABLE_DUMP2PROTO="on"
  export ENABLE_DUMP_IR="on"
  export COMPILE_MINDDATA="on"
  export COMPILE_MINDDATA_LITE="lite_cv"
  export ENABLE_MPI="on"
  export CUDA_VERSION="11.5"
  export COMPILE_LITE="off"
  export LITE_PLATFORM=""
  export LITE_ENABLE_AAR="off"
  export USE_GLOG="on"
  export ENABLE_AKG="off"
  export ENABLE_ACL="off"
  export ENABLE_D="off"
  export ENABLE_DEBUGGER="on"
  export ENABLE_IBVERBS="off"
  export ENABLE_PYTHON="on"
  export ENABLE_GPU="on"               # GPU支持
  export ENABLE_VERBOSE="off"
  export ENABLE_GITEE="on"             # 国内github不稳定
  export ENABLE_MAKE_CLEAN="on"        # make clean
  export X86_64_SIMD="on"              # 打开SIMD
  export ARM_SIMD="off"
  export DEVICE_VERSION=""
  export DEVICE=""
  export ENABLE_HIDDEN="on"
  export TENSORRT_HOME=""
  export USER_ENABLE_DUMP_IR=false
  export USER_ENABLE_DEBUGGER=false
  export ENABLE_SYM_FILE="off"
  export ENABLE_FAST_HASH_TABLE="on"
}

```

## 开始编译

```bash
bash build.sh
```

一般没问题，中间可能因为网络问题有个`'https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip' failed`错误。

整个编译过程视 CPU 性能需要大概半个小时。

先这样，期末复习了，原意是想搞明白`dataset`的原理，因为一个不知原因的报错：https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=175672

无奈，`MindSpore`都能把 C++的报错展示给用户，我也只能去看源代码了。

看到`MindSpore`的代码仓库，心想，单这些就够我学一辈子的了。下学期好好学编译原理。

特别感谢 SCU Maker 提供的算力支持
