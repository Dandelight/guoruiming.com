# 自定义数据集

---

`MindSpore 1.5.0`，`Python 3.7.5`，`CUDA 11.1`，使用`docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:1.5.0`镜像。

---

最近有关于多模态学习的项目，使用`MindSpore`进行开发，首先需要自定义数据。

`MindSpore`中数据的定义通过[dataset API](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore.dataset.html)定义，其中有一系列内置的数据集解决方案，在本文撰写(2021.12.29)的时候支持了`CelebA`、`Cifar10`、`Cifar100`、`COCO`、`ImageFolder`、`Mnist`、`VOC`（以上为计算机视觉）、`CLUE`（文本）、`Graph`（图神经网络），还有`CSV`、`Text`等文本格式，以及`Manifest`、`TFRecord`和`MindRecord`等预处理过的结构化格式。

但是，问题是，我们现在需要在一个全新的数据集上很快地跑出结果，这就需要**自定义数据集**。

最简单的自定义数据集方式是[`GeneratorDataset`](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)。顾名思义，其实现了`Python`中`generator`的`iteratable`接口，可以作为迭代的对象。

```python
class mindspore.dataset.GeneratorDataset(
    source,
    column_names=None,
    column_types=None,
    schema=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    python_multiprocessing=True,
    max_rowsize=6
)
```

我们目前最关注的是`source`和`column_names`两个参数，`source`是一个可迭代对象，应该是一个`(generator | iteratable | random_accessable)`对象。概括来说，**这个对象应该该每次访问时返回一个`tuple`，`tuple`中的每个元素都必须是`np.array`对象或`tuple`或`list`。**`tuple`可嵌套，但最终对象一定要是`np.array`。具体来说，有四种方法定义这个对象

```python
import numpy as np

# 1) genarator callable: 单列数据，np.array，包装在tuple里
def generator_multidimensional():
    for i in range(64):
        yield (np.array([[i, i + 1], [i + 2, i + 3]]),)

dataset = ds.GeneratorDataset(source=generator_multidimensional, column_names=["multi_dimensional_data"])

# 2) generator callable: 多列数据，np.array，包装在tuple
def generator_multi_column():
    for i in range(64):
        yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])

dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["col1", "col2"])

# 3) 可迭代对象(实现了__next__、__iter__、__len__方法)
class MyIterable:
    def __init__(self):
        self._index = 0
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)

dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])

# 4) 可随机访问对象(实现了__getitem__和__len__方法)
class MyAccessible:
    def __init__(self):
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

dataset = ds.GeneratorDataset(source=MyAccessible(), column_names=["data", "label"])

# 5) Python原生对象(可以是list、tuple或generator)，但内部**必须是**tuple包装的一或多个np.array
dataset = ds.GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])

```

其中，第四种方法理解较为简单，在各框架中通用性高，较为推荐。

使用方法：

```python
#%% In[1]
import mindspore as ms
from mindspore.dataset import GeneratorDataset
import numpy as np

class RandomScalarDataset():
    def __init__(self, size=4):
        self._data = np.random.sample((size, 2))
        self._label = np.random.sample((size, 1)).astype(np.uint64)

    def __getitem__(self, index):
        # return self._data[index], self._label[index]
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

if __name__ == '__main__':
    np.random.seed(42)
    dataset = RandomScalarDataset()
    generator_dataset = GeneratorDataset(source=dataset, column_names=['scalar', 'target'])
    it = generator_dataset.create_tuple_iterator()
    for i in it: print(i)

    it = generator_dataset.create_dict_iterator()
    for i in it: print(i)
```

预期输出：

```python
[Tensor(shape=[2], dtype=Float64, value= [ 5.80836122e-02,  8.66176146e-01]), Tensor(shape=[1], dtype=UInt64, value= [0])]
[Tensor(shape=[2], dtype=Float64, value= [ 3.74540119e-01,  9.50714306e-01]), Tensor(shape=[1], dtype=UInt64, value= [0])]
[Tensor(shape=[2], dtype=Float64, value= [ 7.31993942e-01,  5.98658484e-01]), Tensor(shape=[1], dtype=UInt64, value= [0])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.56018640e-01,  1.55994520e-01]), Tensor(shape=[1], dtype=UInt64, value= [0])]
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 1.56018640e-01,  1.55994520e-01]), 'target': Tensor(shape=[1], dtype=UInt64, value= [0])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 7.31993942e-01,  5.98658484e-01]), 'target': Tensor(shape=[1], dtype=UInt64, value= [0])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 3.74540119e-01,  9.50714306e-01]), 'target': Tensor(shape=[1], dtype=UInt64, value= [0])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 5.80836122e-02,  8.66176146e-01]), 'target': Tensor(shape=[1], dtype=UInt64, value= [0])}
```

测试 2（tuple 嵌套）

```python
import mindspore as ms
from mindspore.dataset import GeneratorDataset
import numpy as np

class RandomScalarDataset():
    def __init__(self, size=4):
        self._data = (np.random.sample((size, 2)))
        self._label = np.random.randint(0, 42, (size, 1)).astype(np.uint64)

    def __getitem__(self, index):
        # return self._data[index], self._label[index]
        return self._data[index], (self._label[index], self._label[index])

    def __len__(self):
        return len(self._data)

if __name__ == '__main__':
    np.random.seed(42)
    dataset = RandomScalarDataset()
    generator_dataset = GeneratorDataset(source=dataset, column_names=['scalar', 'target'])
    it = generator_dataset.create_tuple_iterator()
    for i in it: print(i)

    it = generator_dataset.create_dict_iterator()
    for i in it: print(i)
```

输出

```python
[Tensor(shape=[2], dtype=Float64, value= [ 3.74540119e-01,  9.50714306e-01]), Tensor(shape=[2, 1], dtype=UInt64, value=
[[35],
 [35]])]
[Tensor(shape=[2], dtype=Float64, value= [ 7.31993942e-01,  5.98658484e-01]), Tensor(shape=[2, 1], dtype=UInt64, value=
[[39],
 [39]])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.56018640e-01,  1.55994520e-01]), Tensor(shape=[2, 1], dtype=UInt64, value=
[[23],
 [23]])]
[Tensor(shape=[2], dtype=Float64, value= [ 5.80836122e-02,  8.66176146e-01]), Tensor(shape=[2, 1], dtype=UInt64, value=
[[2],
 [2]])]
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 5.80836122e-02,  8.66176146e-01]), 'target': Tensor(shape=[2, 1], dtype=UInt64, value=
[[2],
 [2]])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 3.74540119e-01,  9.50714306e-01]), 'target': Tensor(shape=[2, 1], dtype=UInt64, value=
[[35],
 [35]])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 1.56018640e-01,  1.55994520e-01]), 'target': Tensor(shape=[2, 1], dtype=UInt64, value=
[[23],
 [23]])}
{'scalar': Tensor(shape=[2], dtype=Float64, value= [ 7.31993942e-01,  5.98658484e-01]), 'target': Tensor(shape=[2, 1], dtype=UInt64, value=
[[39],
 [39]])}
```
