# REIGNITE：重学 PyTorch

## 安装

首先在安装好 `CUDA 11.x` 的 Linux 电脑上安装 [`docker`](https://docker.com) 和 [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) 和 [`docker-compose v2`](https://github.com/docker/compose)。

使用 `docker` 的好处是可以连 `CUDA` 版本一起放进虚拟环境。

然后随便在哪新建一个文件夹，随便起名，比如 `torch_env`

然后 `cd torch_env`，将以下内容写进 `Dockerfile`

```dockerfile
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# apt 换源
USER root
RUN sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list &&\
        sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
# 接收 cudatools 的公钥，信任 CUDA 更新包，如果出现因为 cuda 更新的错误就将这句反注释掉
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# 安装 openssh
RUN apt-get update && apt-get install -y openssh-server --fix-missing
# pip 换源
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
# 添加 SSH 公钥
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config &&\
        echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config &&\
        echo "AuthorizedKeysFile  .ssh/authorized_keys" >> /etc/ssh/sshd_config &&\
        /etc/init.d/ssh restart &&\
        mkdir -p ~/.ssh &&\
        echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMYZoO4AcJo32+7D8n7/JRWOMEU0KB/J8w4HuJ01GVSD min@DESKTOP-7BOCHVM" > ~/.ssh/authorized_keys
# >>> 在这里可以进行 pip install 等软件包安装
# 开启容器内 SSH 访问
ENTRYPOINT ["/usr/sbin/sshd", "-D"]
```

再将以下内容写进 `docker-compose.yml`

```yaml
version: "3"
services:
  jupyter:
    restart: always
    # image: ufoym/deepo:all-jupyter-py36-cu111
    build: "."
    container_name: jupyter-all
    ports:
      - "8822:22"
    shm_size: "32gb" # PyTorch 多线程加载数据
    volumes:
      - "$HOME:$HOME"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"] # NVIDIA GPU支持
```

最后

```bash
docker compose up -d
```

## 定义

`array_like` 指 list, tuple, NumPy `ndarray`, 标量等数据类型

1. a tensor
2. a NumPy array
3. a DLPack capsule
4. an object that implements Python’s buffer protocol
5. a scalar
6. a sequence of scalars

## 重新认识 Tensor

### Tensor 是什么

### 创建一个 Tensor

- 使用 [`torch.tensor()`](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) 从已有数据创建 Tensor
- 使用 `torch.*` Tensor Creation Ops 创建特定形状、初始内容和数据类型的 Tensor (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).
- 使用 `torch.*_like` 创建**形状等同**于当前某个 Tensor 的具有特定内容和数据类型的 Tensor (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).
- 使用 `tensor.new_*` 创建与当前某 Tensor 具有相同**数据类型**的 Tensor

#### `torch.tensor`

创建一个没有 Autograd 记录的 leaf tensor，不会与原对象共享内存。

接收一个 `data: array_like` 参数，作为原始数据。如果没有附加参数，从 `data` 中推断类型信息。

但从 `tensor` 创建 `tensor` 应该使用 `Tensor.clone()`、`Tensor.detach()`（返回一个从当前计算图中脱离的 `tensor`，但返回的 `tensor` 与之共享 `storage`，所以不应该使用 inplace 操作进行改变）、`Tensor.requires_grad_()`。

#### `torch.asarray`

与 `torch.tensor` 不同的是 `asarray` ，如果原对象是 `ndarray`、`DLPack capsule`、实现了 `Python buffer protocol` 的对象，则新 `Tensor` 会与原对象共享内存。

#### `torch.as_tensor`

与上述不同的是，如果原对象是 `tensor`，则会最大程度保留 autograd 历史。

#### `torch.from_numpy`

一定会与原对象共享内存，不支持 `resize` 操作

其他不太常用的还有 `from_dlpack`、`frombuffer` 等。

#### 统一初始化

- `zeros`
- `ones`
- `empty`

以上三者 API 原型相同

```python
a = torch.zeros(1, 2, 3, 4, 5)
a.shape # torch.Size([1, 2, 3, 4, 5])
```

- `full`

```python
def torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor: ...
```

第一个参数是一个 shape_like 对象，第二个是统一的初始值。

```
a = torch((1, 2, 3), 42)
```

- `eye`

单位矩阵，谐音 Identity 里的 I。

```python
def torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

~~看原型应该能看懂是啥意思~~

```python
a = torch.eye(3)
b = torch.eye(3, 4)
```

#### 创建类似现有 Tensor 的 Tensor

- `zeros_like`
- `ones_like`
- `empty_like`
- `full_like`
- `heaviside`

~~前四个怎么用猜都猜得出来~~

第五个，

$$
\text { heaviside }(\text { input, values })= \begin{cases}0, & \text { if input }<0 \\ values, & \text { if input }=0 \\ 1, & \text { if input }>0\end{cases}
$$

#### 范围创建

- ~~`range`：创建有 $\left\lfloor \frac{\text { end-start }}{\text { step }}\right\rfloor+1$ 个元素的一维 Tensor，返回 Tensor 的 dtype 为 `float32`~~ 在 PyTorch 1.11 被 depreciate 了，不要用了
- `arange`：创建有 $\left\lceil\frac{\text { end-start }}{\text { step }}\right\rceil$ 个元素的一维 Tensor，返回 Tensor 的 dtype 为 `int64`
- `linspace`：创建有 $\mathrm{step}$ 个元素的 Tensor，和 MATLAB 中同名函数同义
- `logspace`：$(\mathrm{base} ^{\text {start }}, \mathrm{base} ^{\left(\text {start+ } \frac{\text { end -start }}{\text { steps-1 }}\right)}, \ldots, \mathrm{base} ^{\left(\text {start+(steps-2)* } \frac{\text { end }-\text { start }}{\text { steps }^{-1}}\right)}, \mathrm{base} \left.^{\text {end }}\right)$

#### 特殊的 Tensor

- `meta tensor` 只有形状而没有内容，用于？？？
- sparce tensor 基于稀疏矩阵，类似 `scipy` 的 `sparce_matrix`，可用于图神经网络
- quantized tensor 将 float32 量化为 uint8 而提高推理性能
- complex tensor 复数 Tensor，提供了 `real` 和 `image` 两个属性
- named tensor 带有命名的 Tensor，设计的初衷是用“命名”来跟踪**维度**，在建图时发现维度不匹配问题，而不是像现在一样 `RuntimeException`。

#### 用 Tensor 的 `new_*` 方法创建 Tensor

- `new_tensor`
- `new_full`
- `new_empty`
- `new_ones`
- `new_zeros`

`new` 出的 Tensor 与原 Tensor 有相同的 `dtype` 和 `device`。

```python
tensor = torch.ones((2,), dtype=torch.float64)
tensor.new_full((3, 4), 3.141592)
```

~~一生二二生三三生万物~~

### 随机初始化

- [`torch.rand()`](https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand) 值初始化为 $[0, 1)$ 之间均匀分布
- [`torch.rand_like()`](https://pytorch.org/docs/stable/generated/torch.rand_like.html#torch.rand_like)
- [`torch.randn()`](https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn) 值初始化为服从 $\mathcal{N}(0, 1)$ 的正态分布
- [`torch.randn_like()`](https://pytorch.org/docs/stable/generated/torch.randn_like.html#torch.randn_like)
- [`torch.randint()`](https://pytorch.org/docs/stable/generated/torch.randint.html#torch.randint) 值为 `[low, high)` 之间的随机 `torch.int64`
- [`torch.randint_like()`](https://pytorch.org/docs/stable/generated/torch.randint_like.html#torch.randint_like)
- [`torch.randperm()`](https://pytorch.org/docs/stable/generated/torch.randperm.html#torch.randperm) 返回一个 $1\ldots n$ 的随机排列的一维 Tensor，`dtype` 默认 `torch.int64`

### Tensor 的基本属性（从实现的角度）

Tensor 是线性计算中最常见的**数据结构**。

- 维数：`ndim`：返回一个 `Python` `int` 对象，表示维度
- 形状：`shape`：返回一个 `torch.Shape` 对象，表示每个维度的大小
- 访问步长：`stride()`：返回一个 `tuple`，为维度的 `stride`。
- storage：`storage()`：返回一个 `torch.Storage` 的子类，为 `torch` 的底层一维存储
- 数据类型：`dtype`：返回一个 `torch.dtype` 对象，表示这个 `tensor` 中数据的类型
- 数据内容：直接访问或者 `b.data`，可以像数组一样操作，具体访问方式在后文介绍。
- 设备：`device`：返回一个 `torch.device` 对象
- 梯度：`grad`：如果这个 Tensor 有 grad 返回之；如果没有返回 None，具体会在 Autograd 一节中详细介绍。
- `layout`（beta）对于 dense tensor 等价于 `stride`，对于 `sparse COO tensor` 处于测试阶段
- `memory_format`：规定数据的存储和访问方式，主要有 `contiguous_format` 和 `channel_last` 两种。
  - `torch.contiguous_format`：默认表现，Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.
  - `torch.channels_last`： Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in `strides[0] > strides[2] > strides[3] > strides[1] == 1` aka NHWC order.
  - `torch.preserve_format`：Used in functions like `clone` to preserve the memory format of the input tensor. If input tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied from the input. Otherwise output strides will follow `torch.contiguous_format`

目前能想到的就是这些。

### Tensor 的数据类型

直接摘抄 `PyTorch` 官网的解释如下：~~难得认真地分个类画个表~~

| Data type                                                                                | dtype                                 | CPU tensor             | GPU tensor                  |
| ---------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------- | --------------------------- |
| 32-bit floating point                                                                    | `torch.float32` or `torch.float`      | `torch.FloatTensor`    | `torch.cuda.FloatTensor`    |
| 64-bit floating point                                                                    | `torch.float64` or `torch.double`     | `torch.DoubleTensor`   | `torch.cuda.DoubleTensor`   |
| 16-bit floating point [1](https://pytorch.org/docs/stable/tensors.html#id4)              | `torch.float16` or `torch.half`       | `torch.HalfTensor`     | `torch.cuda.HalfTensor`     |
| 16-bit floating point [2](https://pytorch.org/docs/stable/tensors.html#id5)              | `torch.bfloat16`                      | `torch.BFloat16Tensor` | `torch.cuda.BFloat16Tensor` |
| 32-bit complex                                                                           | `torch.complex32` or `torch.chalf`    |                        |                             |
| 64-bit complex                                                                           | `torch.complex64` or `torch.cfloat`   |                        |                             |
| 128-bit complex                                                                          | `torch.complex128` or `torch.cdouble` |                        |                             |
| 8-bit integer (unsigned)                                                                 | `torch.uint8`                         | `torch.ByteTensor`     | `torch.cuda.ByteTensor`     |
| 8-bit integer (signed)                                                                   | `torch.int8`                          | `torch.CharTensor`     | `torch.cuda.CharTensor`     |
| 16-bit integer (signed)                                                                  | `torch.int16` or `torch.short`        | `torch.ShortTensor`    | `torch.cuda.ShortTensor`    |
| 32-bit integer (signed)                                                                  | `torch.int32` or `torch.int`          | `torch.IntTensor`      | `torch.cuda.IntTensor`      |
| 64-bit integer (signed)                                                                  | `torch.int64` or `torch.long`         | `torch.LongTensor`     | `torch.cuda.LongTensor`     |
| Boolean                                                                                  | `torch.bool`                          | `torch.BoolTensor`     | `torch.cuda.BoolTensor`     |
| quantized 8-bit integer (unsigned)                                                       | `torch.quint8`                        | `torch.ByteTensor`     | /                           |
| quantized 8-bit integer (signed)                                                         | `torch.qint8`                         | `torch.CharTensor`     | /                           |
| quantized 32-bit integer (signed)                                                        | `torch.qint32`                        | `torch.IntTensor`      | /                           |
| quantized 4-bit integer (unsigned) [3](https://pytorch.org/docs/stable/tensors.html#id6) | `torch.quint4x2`                      | `torch.ByteTensor`     | /                           |

相同设备上，不同数据类型的实数 Tensor 进行计算时遵循类似 C 语言的变量类型提升规则，而不同设备上的 Tensor 不能直接计算。

### Tensor 的操作

半天才到这一步，好戏还在后头呢。
