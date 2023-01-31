# TorchScript

~~什么，学个 PyTorch 还得学一门新语言？？~~

好消息是，TorchScript 是 Python 的一个子集。

更好的消息是，TorchScript 是强类型的。

可以直接从 `PyTorch` 中构建 `torchscript`，构建方式有 `script` 和 `trace`；两种方法的返回值都是 `ScriptModule`。

```python
import torch

@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r


def bar(x, y, z):
    return foo(x, y) + z

traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3)))
```

```python
import torch

def foo(x, y):
    return 2 * x + y

traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

@torch.jit.script
def bar(x):
    return traced_foo(x, x)
```

```python
import torch
import torchvision

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                      torch.rand(1, 3, 224, 224))

    def forward(self, input):
        return self.resnet(input - self.means)

my_script_module = torch.jit.script(MyScriptModule())
```

可以通过调用 `.code` 方法查看代码来进行调试：

```python
@torch.jit.script
def foo(len):
    # type: (int) -> torch.Tensor
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv

print(foo.code)
```

可以调用 `.graph` 方法检查计算图

```python
@torch.jit.script
def foo(len):
    # type: (int) -> torch.Tensor
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv

print(foo.graph)
```

但一个比较大的问题是 `trace` 现阶段难以跟踪**具有分支的代码**，建议将分支中相似部分尽可能提取出来，对 `common block` 进行 JIT。

## PyTorch FX

`torch.fx` 是**与 TorchScript 独立的** 对 Module 进行运行时转换的工具包，可以对模块、图、代码进行详细的编辑。暂时用不到，用到了再学。
