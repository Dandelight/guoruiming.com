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
# 接收 cudatools 的公钥，信任 CUDA 更新包
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys  A4B469963BF863CC
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

### Tensor 的基本属性（从实现的角度）

Tensor 是线性计算中最常见的**数据结构**。

- 维数：`ndim`
- 形状：`shape`
- 访问步长：`stride`
- storage：`storage`
- 数据类型：`dtype`
- 数据内容
- 设备：`device`
- 梯度：`grad`

### 特殊的 Tensor

- `meta tensor` 只有形状而没有内容，用于？？？
- sparce tensor 基于稀疏矩阵，类似 `scipy` 的 `sparce_matrix`，可用于图神经网络
- quantized tensor 将 float32 量化为 uint8 而提高推理性能
- complex tensor 复数 Tensor，提供了 `real` 和 `image` 两个属性

### 创建一个 Tensor

- 使用 [`torch.tensor()`](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) 从已有数据创建 Tensor
- 使用 `torch.*` Tensor Creation Ops 创建特定形状、初始内容和数据类型的 Tensor (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).
- 使用 `torch.*_like` 创建**形状等同**于当前某个 Tensor 的具有特定内容和数据类型的 Tensor (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).
- 使用 `tensor.new_*` 创建与当前某 Tensor 具有相同**数据类型**的 Tensor

#### `torch.tensor`

创建一个没有 Autograd 记录的 leaf tensor，不会与原对象共享内存。

接收一个 `data: array_like` 参数，作为原始数据。如果没有附加参数，从 `data` 中推断类型信息。

但从 `tensor` 创建 `tensor` 应该使用 `Tensor.clone()`、`Tensor.detach()`（返回一个从当前计算图中脱离的 `tensor`，但返回的 `tensor` 与之共享 `storage`，所以不应该使用 inplace 操作进行改变）、`Tensor.requires_grad_()`。

### `torch.asarray`

与 `torch.tensor` 不同的是 `asarray` ，如果原对象是 `ndarray`、`DLPack capsule`、实现了 `Python buffer protocol` 的对象，则新 `Tensor` 会与原对象共享内存。

### `torch.as_tensor`

与上述不同的是，如果原对象是 `tensor`，则会最大程度保留 autograd 历史。

### `torch.from_numpy`

一定会与原对象共享内存，不支持 `resize` 操作

其他不太常用的还有 `from_dlpack`、`frombuffer` 等。

### 统一初始化

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

### 创建类似现有 Tensor 的 Tensor

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

### 范围创建

- `range`：创建有 $\left\lfloor \frac{\text { end-start }}{\text { step }}\right\rfloor+1$ 个元素的一维 Tensor
- `arange`：创建有 $\left\lceil\frac{\text { end-start }}{\text { step }}\right\rceil$ 个元素的一维 Tensor
- `linspace`：创建有 $\mathrm{step}$ 个元素的 Tensor，和 MATLAB 中同名函数同义
- `logspace`：$(\mathrm{base} ^{\text {start }}, \mathrm{base} ^{\left(\text {start+ } \frac{\text { end -start }}{\text { steps-1 }}\right)}, \ldots, \mathrm{base} ^{\left(\text {start+(steps-2)* } \frac{\text { end }-\text { start }}{\text { steps }^{-1}}\right)}, \mathrm{base} \left.^{\text {end }}\right)$

## 使用这个 Tensor
