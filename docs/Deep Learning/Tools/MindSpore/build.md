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

顾名思义，cuDNN 是一个用于加速深度神经网络训练的 CUDA 库。MindSpore 依赖这个库。下载方式见https://developer.nvidia.cn/rdp/cudnn-download，可能需要注册NVIDIA开发者账号，可以微信登录。**注意cuDNN版本一定要和CUDA版本对应！**此处安装cuDNN v8.3.1 (November 22nd, 2021), for CUDA 11.5

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

这两个环境变量会被`flatbuffers`调用，参考：https://gitee.com/mindspore/mindspore/issues/I4ODSV

## 编译配置···

~~打开`scripts/build`文件夹，编辑`default_options.sh`。在该版本还没有找到文档，只能看名字了。~~

上一次编译本人采用了魔改默认配置的方法，直接造成连接`ut_tests`失败，详见[issue #I4OH8W](https://gitee.com/mindspore/mindspore/issues/I4OH8W)。

这次我终于学会了活下去的方法：看文档！

```bash
$ bash build.sh -?
---------------- MindSpore: build start ----------------
build.sh: illegal option -- ?
Unknown option ?!
Usage:
bash build.sh [-d] [-r] [-v] [-c on|off] [-t ut|st] [-g on|off] [-h] [-b ge] [-m infer|train] \
              [-a on|off] [-p on|off] [-i] [-R] [-D on|off] [-j[n]] [-e gpu|ascend|cpu] \
              [-P on|off] [-z [on|off]] [-M on|off] [-V 10.1|11.1|310|910] [-I arm64|arm32|x86_64] [-K on|off] \
              [-B on|off] [-E] [-l on|off] [-n full|lite|off] [-H on|off] \
              [-A on|off] [-S on|off] [-k on|off] [-W sse|neon|avx|avx512|off] \
              [-L Tensor-RT path] [-y on|off] [-F on|off] \

Options:
    -d Debug mode
    -r Release mode, default mode
    -v Display build command
    -c Enable code coverage, default off
    -t Run testcases, default off
    -g Use glog to output log, default on
    -h Print usage
    -b Select other backend, available: \
           ge:graph engine
    -m Select graph engine backend mode, available: infer, train, default is infer
    -a Enable ASAN, default off
    -p Enable pipeline profile, print to stdout, default off
    -R Enable pipeline profile, record to json, default off
    -i Enable increment building, default off
    -j[n] Set the threads when building (Default: -j8)
    -e Use cpu, gpu or ascend
    -s Enable security, default off
    -P Enable dump anf graph to file in ProtoBuffer format, default on
    -D Enable dumping of function graph ir, default on
    -z Compile dataset & mindrecord, default on
    -n Compile minddata with mindspore lite, available: off, lite, full, lite_cv, full mode in lite train and lite_cv, wrapper mode in lite predict
    -M Enable MPI and NCCL for GPU training, gpu default on
    -V Specify the device version, if -e gpu, default CUDA 10.1, if -e ascend, default Ascend 910
    -I Enable compiling mindspore lite for arm64, arm32 or x86_64, default disable mindspore lite compilation
    -A Enable compiling mindspore lite aar package, option: on/off, default: off
    -K Compile with AKG, default on if -e gpu or -e ascend, else default off
    -B Enable debugger, default on
    -E Enable IBVERBS for parameter server, default off
    -l Compile with python dependency, default on
    -S Enable enable download cmake compile dependency from gitee , default off
    -k Enable make clean, clean up compilation generated cache
    -W Enable SIMD instruction set, use [sse|neon|avx|avx512|off], default avx for cloud CPU backend
    -H Enable hidden
    -L Link and specify Tensor-RT library path, default disable Tensor-RT lib linking
    -y Compile the symbol table switch and save the symbol table to the directory output
    -F Use fast hash table in mindspore compiler, default on
```

您看这文档真的很详细。

## 开始编译

```bash
bash build.sh -e gpu -S on -V 11.1 -v -k -j12 | tee build.log
```

![success](media/build/success.png)

整个编译过程视 CPU 性能需要大概一两个小时。

先这样，期末复习了，原意是想搞明白`dataset`的原理，因为一个不知原因的报错：https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=175672

编译成功会是这个结果：

```
--------------- MindSpore: build end ---------------
```

如果在`output/`文件夹中找到`mindspore_gpu-1.6.0-cp39-cp39-linux_x86_64.whl `就是成功了。

然后我们把它装上

```bash
pip install wheel
pip install mindspore_gpu-1.6.0-cp39-cp39-linux_x86_64.whl
```

测试一下：

![image-20211230204353743](media/build/image-20211230204353743.png)

虽然有两个警告，但是因为我们是从源码编译，已经自动链接`CUDA 11.5`了，通过`nvidia-smi`命令监控`GPU`利用率也是正常的。完工！

**下一步就是读懂`MindSpore`的源码，然后贡献代码啦！**

看到`MindSpore`的代码仓库，心想，单这些就够我学一辈子的了。下学期好好学编译原理。

特别感谢 SCU Maker 提供的算力支持
