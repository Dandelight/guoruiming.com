# Misc

## `torch.stack` 和 `torch.cat`

两个函数非常类似，原型都是

```python
(tensors, dim=0, *, out=None)
```

，不同点是：

* `stack` 会插入一维；相应地要求其他维度都相等
* `cat | concat | concatenate` 不会插入一维；要求除 `dim` 外的所有维度都相等。
