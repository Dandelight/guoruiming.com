# Autograd 以及反向传播

要谈 `autograd`，就要先从创建 `tensor` 时的 `requires_grad` 关键词参数说起。

## 标量多元微分

```python
import torch
```

```python
t = torch.tensor([[1,2,3],[4,5,6]])
```

```python
a = torch.tensor(1.0)
b = torch.tensor(2.0)
```

```python
print(a.grad) # None
```

```python
c = a + b
try:
    c.backward()
except RuntimeError as e:
    print(e)
# element 0 of tensors does not require grad and does not have a grad_fn
```

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
```

```python
print(a.grad, b.grad)
# None None
```

```python
c = a + b
print(c)
# tensor(3., grad_fn=<AddBackward0>)
```

```python
c.backward()
# 也可以写作 torch.autograd.backward(c)
```

```python
c, a, b
# (tensor(3., grad_fn=<AddBackward0>),
# tensor(1., requires_grad=True),
# tensor(2., requires_grad=True))
```

```python
a.grad, b.grad
# (tensor(1.), tensor(1.))
```

```python
a = torch.tensor(42.0, requires_grad=True)
b = torch.tensor(99.0, requires_grad=True)
c = a * b
c
# tensor(4158., grad_fn=<MulBackward0>)
c.backward()
a.grad, b.grad
# (tensor(99.), tensor(42.))
print(a.grad_fn) # None
print(a.is_leaf) # True
print(c.grad_fn, c.is_leaf) # <MulBackward0 object at 0x000001E264B841F0> False
```

### 一个更加复杂的例子

$$
f = a \times b + \frac{c \times d^2}{e}
$$

```python
a = torch.tensor(2., requires_grad=True)
b = torch.tensor(4., requires_grad=True)
c = torch.tensor(6., requires_grad=True)
d = torch.tensor(8., requires_grad=True)
e = torch.tensor(10., requires_grad=True)
```

```python
print(a.grad) # None
```

```python
g = a * b
h = c*d**2/e
f = g + h
```

```python
print(g.is_leaf, g.grad_fn)
# False <MulBackward0 object at 0x000001E264C1F7F0>
```

```python
f.backward()
```

```python
for el in [a, b, c, d, e]:
    print(el.grad)
# tensor(4.)
# tensor(2.)
# tensor(6.4000)
# tensor(9.6000)
# tensor(-3.8400)
```

```python
a.grad = None
```

```python
g.grad, h.grad
```

```
d:\Users\Min\Anaconda3\envs\torch\lib\site-packages\torch\_tensor.py:1083: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at  C:\cb\pytorch_1000000000000\work\build\aten\src\ATen/core/TensorBody.h:482.)
      return self._grad

(None, None)
```

## 矩阵的多元微分

```python
a = torch.tensor([[2.,4.,6.],[2.,3.,4.],[9.,7.,8.]], requires_grad=True)
b = torch.tensor([[4.,9.,2.],[7.,1.,1.],[6.,7.,3.]], requires_grad=True)
c = torch.tensor([[6.,1.,3.],[2.,3.,2.],[4.,8.,6.]], requires_grad=True)
d = torch.tensor([[2.,2.,4.],[4.,4.,6.],[3.,2.,8.]], requires_grad=True)
e = torch.tensor([[1.,3.,6.],[7.,1.,4.],[1.,4.,6.]], requires_grad=True)
```

```python
g = a @ b
h = c@d**2/e
f = g + h
```

```python
grad_tensor = torch.ones_like(g)
g.backward(grad_tensor)
print(g, a.grad, b.grad)
```

    tensor([[ 72.,  64.,  26.],
            [ 53.,  49.,  19.],
            [133., 144.,  49.]], grad_fn=<MmBackward0>) tensor([[15.,  9., 16.],
            [15.,  9., 16.],
            [15.,  9., 16.]]) tensor([[13., 13., 13.],
            [14., 14., 14.],
            [18., 18., 18.]])

```python
g = a @ b
h = c@d**2/e
f = g + h

grad_tensor = torch.ones_like(g)
grad_tensor[1][1] = 2
g.backward(grad_tensor)
print(g, a.grad, b.grad)
```

    tensor([[ 72.,  64.,  26.],
            [ 53.,  49.,  19.],
            [133., 144.,  49.]], grad_fn=<MmBackward0>) tensor([[30., 18., 32.],
            [39., 19., 39.],
            [30., 18., 32.]]) tensor([[26., 28., 26.],
            [28., 31., 28.],
            [36., 40., 36.]])

```python
f
```

    tensor([[139.0000,  81.3333,  80.0000],
            [ 63.5714, 113.0000,  86.0000],
            [331.0000, 186.0000, 171.6667]], grad_fn=<AddBackward0>)

```python
g = a @ b
h = c@d**2/e
f = g + h

f.backward(torch.ones_like(f))
for el in [a, b, c, d, e]:
    print(el.grad)
```

    tensor([[45., 27., 48.],
            [54., 28., 55.],
            [45., 27., 48.]])
    tensor([[39., 41., 39.],
            [42., 45., 42.],
            [54., 58., 54.]])
    tensor([[ 8.0000, 27.3333, 21.0000],
            [ 8.5714, 27.2857, 21.2857],
            [ 7.6667, 26.0000, 20.6667]])
    tensor([[41.1429, 20.0000, 17.3333],
            [75.4286, 42.6667, 27.0000],
            [55.7143, 18.0000, 32.0000]])
    tensor([[ -67.0000,   -5.7778,   -9.0000],
            [  -1.5102,  -64.0000,  -16.7500],
            [-198.0000,  -10.5000,  -20.4444]])

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
```

```python
y = x @ x
print(y)
y.backward()
print(x.grad)
```

    tensor(14., grad_fn=<DotBackward0>)
    tensor([2., 4., 6.])

```python
y = x.T @ x
print(y)
y.backward()
print(x.grad)
```

    tensor(14., grad_fn=<DotBackward0>)
    tensor([ 4.,  8., 12.])


    C:\Users\Min\AppData\Local\Temp\ipykernel_15588\3668660662.py:1: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matricesor `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at  C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\TensorShape.cpp:2985.)
      y = x.T @ x

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

loss.backward()
```
