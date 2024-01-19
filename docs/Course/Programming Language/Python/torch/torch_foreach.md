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
