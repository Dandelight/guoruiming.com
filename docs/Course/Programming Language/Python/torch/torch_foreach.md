# PyTorch `foreach` 中的暗坑

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
