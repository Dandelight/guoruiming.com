# Python & PyTorch 暗坑集合

## `for-in` 返回的是引用，可以 in-place 修改

可以看到，`NumPy` 中返回的是一个 `copy`，而 `PyTorch` 中返回的是一个引用。

```python
>>> import numpy as np
>>> a = np.arange(1, 10)
>>> a
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> for i in a: i+=1
...
>>> a
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> import torch
>>> b = torch.arange(1, 10)
>>> b
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> for i in b: i+=1
...
>>> b
tensor([ 2,  3,  4,  5,  6,  7,  8,  9, 10])
>>>
```

## `randint`：在 Python 标准库和 `NumPy`、`PyTorch` 中表现不一致

在 `Python` 标准库中，`randint(a, b)` 返回的是 `[a, b]` 的整数（闭区间），而在 `NumPy` 和 `PyTorch` 中，返回的是 `[a, b)` 的整数（左开右闭）。

## PyTorch `kl_div` 的第一个参数居然需要是对数

> To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space. The argument target may also be provided in the log-space if `log_target = True`.

也就是说，`torch.kl_div` 的 `inputs` 参数希望已经是 `log` 之后的值，而 `target` 是原来的 `target`。

其他损失函数也各有自己的假设

- `softmax` 需要注意的是，输入 `input` 要求为 `logit`（模型最后一层输出，不经过 `softmax`），`target`为该样本对应的类别, `torch.long` 类型（`int64`）。
- `torch.nn.functional.negative_log_likelihood`、 `torch.nn.BECLoss` 和 `kl_div` 一样，也是希望 `inputs` 是对数函数的输出结果。
- `BCEWithLogitsLoss` 和 `softmax` 一样，输入是 `logits`，剩余部分与 `BCELoss` 一致、
- `MultiLabelSoftMarginLosws` 和 `BCEWithLogitsLoss` 的效果是一样的

这里的一切麻烦都是因为 `torch` 需要保证数值稳定性……但我还是希望像个数学家一样，只写公式，数值稳定性让程序员们搞定吧。

<https://zhuanlan.zhihu.com/p/267787260>
