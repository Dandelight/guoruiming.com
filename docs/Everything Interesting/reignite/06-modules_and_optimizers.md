# Module 和 Optimizer

首先说明：在神经网络构建方面，`PyTorch` 也有面向对象编程和函数式编程两种形式，分别在 `torch.nn` 和 `torch.nn.functional` 模块下面。面向对象式编程中，状态（模型参数）保存在对象中；函数式编程中，函数没有状态，状态由用户维护；对于无状态的层（比如激活、归一化和池化）两者差异甚微。应用较广泛的是面向对象式编程，本文也重点介绍前者。

## Module 是什么

- 有状态的神经网络计算模块
- 与 PyTorch 的 autograd 系统紧密结合，与 Parameter、Optimizer 共同组成易于训练学习的神经网络模型
- 方便存储、转移。`torch` 内置序列化与反序列化方法、在 `cpu`、`gpu`、`tpu` 等设备之间无缝切换、进行模型剪枝、量化、JIT 等

关键点就两个：**状态**和**计算**~~（这怎么听起来这么像动态规划）~~

`torch.nn.Module` 是 `torch` 中所有神经网络层的基类。

`Module` 的组成：

- `submodule`：子模块
- `Parameter`：参数，可跟踪梯度，可被更新
- `Buffer`：Buffer 是状态的一部分，但是不是参数，就比如说 `BatchNorm` 的 `running_mean` 不是参数，但是是模型的状态。

公有属性

- `training`：标识是在 `train` 模式还是在 `eval` 模式，可通过 `Module.train()` 和 `Module.eval()` 切换（或者可以直接赋值）

公有方法

注意，**`Module` 的方法如果不加说明都是 in-place 操作**

### cast 类

- `bfloat16()`
- `half()`
- `float()`
- `double()`
- `type(dst_type)`
- `to(dtype)`

### 设备间搬运类

- `cpu()`
- `cuda(device=None)`
- `to(device)`

### 前向计算

- `forward`
- `zero_grad`

### 增加子模块

- `add_module(module, name)`：挂一个命名为 `name` 的子模块上去
- `register_module(module, name)`：同上
- `register_buffer(name, tensor, persistent=True)`：挂 `buffer`

### 挂 hook

PyTorch 1.11 一共有五个，常用来打印个向量的类型，或者提示进度之类，或者丢进 `TensorBoard` 之类的

- `register_backward_hook(hook)`
- `register_forward_hook(hook)`
- `register_forward_pre_hook(hook)`
- `register_full_backward_hook(hook)`
- `register_load_state_dict_post_hook(hook)`

### 迭代（只读）

- `modules()`：返回一个**递归**迭代过所有 `module` 的迭代器。相同 `module` 只会被迭代过一次。
- `children()`：Returns an iterator over **immediate** children modules. 与 `modules()` 的区别是本函数只迭代直接子节点，不会递归。
- `parameters(recurse=True)`：Returns an iterator over module parameters. This is typically passed to an optimizer. 若 `recurse=True` 会递归，否则只迭代直接子节点。
- `named_children()`：上述函数的命名版本。
- `named_modules()`
- `named_parameters()`

### 迭代并修改

- `apply(fn)`：对 `self` 自己和所有子模块（`.children()`）递归应用 `fn`，常用于初始化权值

### 序列化

- `load_state_dict(state_dict, strict=True)`：将一个 `state_dict` 载入自己的参数
- `state_dict`：返回自己的 `state_dict`，可以使用 `torch.save` 保存之

### 数据格式

- `to(memory_format)`：`memory_format` 在 `Tensor` 一节有写。

### 杂项

- `extra_repr`
- `train`
- `eval`
- `get_submodule`
- `get_submodule`
- `set_extra_state`
- `shared_memory_`：Moves the underlying storage to CUDA shared memory. 只对 `CUDA` 端生效。移动后的 Tensor 不能进行 `resize`。
- `to_empty`：Moves the parameters and buffers to the specified device without copying storage.

## Module 的使用

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

## Optimizer

Optimizer 规定了如何进行梯度更新。**多数 Optimizer 也有状态，保存检查点时记得一并保存。**

### 属性

- `params`：该优化器所优化的 `torch.Tensor` 或者 `dict`。
- `defaults` – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).

### 方法

| 方法                                                                                                                                                      | 描述                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [`Optimizer.add_param_group`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.add_param_group.html#torch.optim.Optimizer.add_param_group) | Add a param group to the [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) s param_groups. |
| [`Optimizer.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict) | Loads the optimizer state.                                                                                               |
| [`Optimizer.state_dict`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict)                | Returns the state of the optimizer as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict).                  |
| [`Optimizer.step`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step)                                  | 一步更新                                                                                                                 |
| [`Optimizer.zero_grad`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad)                   | 清空更新的 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) 的梯度                            |

### 实现

| 算法                                                                                                         | 描述                                                                                                               |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| [`Adadelta`](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta)       | Implements Adadelta algorithm.                                                                                     |
| [`Adagrad`](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad)          | Implements Adagrad algorithm.                                                                                      |
| [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)                   | Implements Adam algorithm.                                                                                         |
| [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)                | Implements AdamW algorithm.                                                                                        |
| [`SparseAdam`](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam) | Implements lazy version of Adam algorithm suitable for sparse tensors.                                             |
| [`Adamax`](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax)             | Implements Adamax algorithm (a variant of Adam based on infinity norm).                                            |
| [`ASGD`](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD)                   | Implements Averaged Stochastic Gradient Descent.                                                                   |
| [`LBFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS)                | Implements L-BFGS algorithm, heavily inspired by [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html). |
| [`NAdam`](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)                | Implements NAdam algorithm.                                                                                        |
| [`RAdam`](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam)                | Implements RAdam algorithm.                                                                                        |
| [`RMSprop`](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)          | Implements RMSprop algorithm.                                                                                      |
| [`Rprop`](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop)                | Implements the resilient backpropagation algorithm.                                                                |
| [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)                      | Implements stochastic gradient descent (optionally with momentum).                                                 |

### 学习率调整

`torch.optim.lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs.

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9) # 将 optimizer 传进 lr_scheduler 里

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

还有同时使用多个 `scheduler` 的

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```

记住：一定要在 `optimizer.step()` 后边调用 `scheduler.step()`？？？？

| 规则                                                                                                                                                                                                   | 描述                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)                                                          | Sets the learning rate of each parameter group to the initial lr times a given function.                                                                                                                                                                                                 |
| [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)                                  | Multiply the learning rate of each parameter group by the factor given in the specified function.                                                                                                                                                                                        |
| [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)                                                                | Decays the learning rate of each parameter group by gamma every step_size epochs.                                                                                                                                                                                                        |
| [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)                                                 | Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.                                                                                                                                                                        |
| [`lr_scheduler.ConstantLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR)                                                    | Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.                                                                                                                                      |
| [`lr_scheduler.LinearLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR)                                                          | Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.                                                                                                                |
| [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)                                           | Decays the learning rate of each parameter group by gamma every epoch.                                                                                                                                                                                                                   |
| [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)                               | Set the learning rate of each parameter group using a cosine annealing schedule, where \eta*{max}*η**ma**x* is set to the initial lr and T*{cur}_T**c**u\*\*r_ is the number of epochs since the last restart in SGDR:                                                                   |
| [`lr_scheduler.ChainedScheduler`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler)                                  | Chains list of learning rate schedulers.                                                                                                                                                                                                                                                 |
| [`lr_scheduler.SequentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR)                                              | Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.                                                          |
| [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)                               | Reduce learning rate when a metric has stopped improving.                                                                                                                                                                                                                                |
| [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)                                                          | Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).                                                                                                                                                                                         |
| [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)                                                    | Sets the learning rate of each parameter group according to the 1cycle learning rate policy.                                                                                                                                                                                             |
| [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) | Set the learning rate of each parameter group using a cosine annealing schedule, where \eta*{max}*η**ma**x* is set to the initial lr, T*{cur}_T**c**u\*\*r_ is the number of epochs since the last restart and T\_{i}_T\*\*i_ is the number of epochs between two warm restarts in SGDR: |

## Stochastic Weight Averaging

见论文：https://arxiv.org/abs/1803.05407

```python
loader, optimizer, model, loss_fn = ...
swa_model = torch.optim.swa_utils.AveragedModel(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_start = 160
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
for epoch in range(300):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()
      if epoch > swa_start:
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()
# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data
preds = swa_model(test_input)
```
