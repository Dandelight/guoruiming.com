## 环境

`Windows` 10，`Python` 3.10.0，`NumPy` 1.21.5

设置`numpy`随机数种子：

```python
import numpy as np
np.random.seed(42)
```

## 问题导入

矩阵乘法是这样定义的：

$$
M_{ij} = \sum_k A_{ik}B_{kj}  = A_{ik}B_{kj}
$$

假设$A, B$都是$3\times 3$矩阵，将上述矩阵乘法展开将是这个样子：

$$
\left[\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right]\left[\begin{array}{lll}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{array}\right]=\left[\begin{array}{lll}
c_{11} & c_{12} & c_{13} \\
c_{21} & c_{22} & c_{23} \\
c_{31} & c_{32} & c_{33}
\end{array}\right]
$$

用`Python`循环实现如下：

```python
A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.empty((3, 2))

for i in range(3):
    for j in range(2):
        for k in range(5):
            M[i， j] += A[i, k] * B[k, j]
```

我们将存储下标所用的变量名指代循环，比如上述代码中，最外层循环称为$i$循环，最内层循环成为$k$循环。

上式可**分解**为**三个步骤**：

1. 对$i, j,k$进行三重循环，每重循环取出一个矩阵的一维
2. 将参与$k$循环的两个向量（`A[i, :]`和`B[:, j]`）进行逐元素相乘，得到一个新向量
3. 对$k$循环的结果进行求和

由此我们可以得到上述运算的一个抽象：

```python
"ik, kj -> ij"
```

对照文章最开始的公式，是否已经理解？

这就是在众多`Python`科学计算库中常用的`einsum`函数，上式使用`einsum`实现如下

```python
M = np.einsum("ik, kj -> ij", A, B)
```

## 具体原理

已知的前提条件：

1. 通过$N$个下标可以唯一确定$N$维矩阵中的一个元素（不考虑越界问题）。
1. 任何一个矩阵运算都可以使用`for`循环实现

在矩阵之间的运算中，下标可以**分为两类**：

1. **自由标(Free index)**，也就是在输出端出现的下标
1. **哑标(Summation index)**，在输入端出现但输出端没有出现的下标

这两种分类不重不漏地包含了所有参与运算的下标，但不管怎样，**相同下标代指的一轴维度必须相同**。

在上一节的例子中，$i,j$是自由标，$k$是哑标。

让我们来看一个简单的例子：

```python
a = np.random.rand(5)
b = np.random.rand(3)
outer_prod = np.einsum("i, j -> ij", a, b)
```

参照之前，我们将`einsum`展开为循环：

```python
c = np.zeros((5, 3))
# 显而易见i循环和j循环的顺序对调，在数学上是等价的，也就是i循环和j循环无关
for i in range(5):
    for j in range(3):
        c[i, j] = a[i] * b[j]
```

在这个例子中，$i, j$都是自由标，所以它们都被保留了下来；没有哑标。

## 算法规则

### 1. 在不同输入里重复出现的下标意味着这个维度需要相乘，相应地，其维度必须相等

还是拿过来刚刚的例子

```python
A = np.random.rand(3, 5)
B = np.random.rand(5, 2)

M = np.einsum("ik, kj -> ij", A, B)
```

在该代码中，$k$所在的维度被逐元素相乘。

### 2. 在输出中省略的下标，即哑标，所在的维度会被求和

依然参照以上代码，输出中没有$k$，所以$k$所在的维度被求和。

如果输出中存在$k$，那么$k$所在维度不会被求和。

```python
M = np.einsum("ik, kj -> ijk", A, B)
```

展开为循环将是这样：

```python
A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.empty((3, 2, 5))

for i in range(3):
    for j in range(2):
        for k in range(5):
            M[i, j, k] = A[i, k] * B[k, j]
```

### 3. 自由标可在输出中以任意顺序出现，但只能出现一次

输出顺序决定了轴的排列顺序，比如：

```python
M = np.einsum("ik, kj -> kij", A, B)
```

聪明的你应该已经会写出数学上等价的循环形式了吧？

```python
A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.empty((5, 3, 2))

for i in range(3):
    for j in range(2):
        for k in range(5):
            M[k, i, j] = A[i, k] * B[k, j]
```

或者采用来自百度飞桨的表述[^paddle.einsum]：

> `einsum` 求和过程理论上等价于如下四步，但实现中实际执行的步骤会有差异。
>
> - 第一步，维度对齐：将所有标记按字母序排序，按照标记顺序将输入张量逐一转置、补齐维度，使得处理后的所有张量其维度标记保持一致
> - 第二步，广播乘积：以维度下标为索引进行广播点乘
> - 第三步，维度规约：将哑标对应的维度分量求和消除
> - 第四步，转置输出：若存在输出标记，则按标记进行转置，否则按广播维度+字母序自由标的顺序转置，返回转之后的张量作为输出

## einsum 有多强？

如下的张量操作或运算均可视为 Einstein 求和的特例

- 单操作数
  - 迹：`trace`：`einsum("ii -> i", x)`
  - 转置：`transpose`：`einsum("ij -> ji", x)`
  - 求和：`sum`：`einsum("ij -> i", x)`
- 双操作数
  - 内积：`dot`
  - 外积：`outer`
  - 广播成绩：`mul`
  - 矩阵乘：`matmul`
  - 批量矩阵乘：`bmm`
- 多操作数
  - 广播乘积：`mul`
  - 多矩阵乘：`A.matmul(B).matmul(C)`

```python
# 单操作数
x = np.random.rand(2, 3)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
## 迹
np.einsum("ii -> i", m)
# 如果看不懂就展开成循环

## 转置
np.einsum("ij -> ji", x)

## 求和
np.einsum("ij -> i", x) # 对行求和
np.einsum("ij -> j", x) #对列求和
np.einsum("ij -> ", x) # 全部求和

# 双操作数
a = np.random.rand(5)
b = np.random.rand(3)
c = np.random.rand(5)
## 内积
np.einsum("i, i ->", a, c)

## 外积
np.einsum("i, j -> ij", a, b) # (5, 3)

## 矩阵乘
y = np.random.rand(3, 5)
np.einsum("ik, kj -> ij", x, y)

# 多操作数
z = np.random.rand(5, 2)
np.einsum("ij, jk, kl -> il", x, y, z)

# 广播写法
A = numpy.random.rand(2, 3, 2)
B = numpy.random.rand(2, 2, 3)
np.einsum('...jk, ...kl->...jl', A,B) # （2, 3, 3)
```

可以前往[https://ajcr.net/Basic-guide-to-einsum/](https://ajcr.net/Basic-guide-to-einsum/)查看更多高级用法。

## `numpy.einsum`隐式写法[^numpy.einsum]

上文中字符串中出现箭头符号`->`为显式写法，没有`->`为隐式写法，即隐式写法不指定输出。

没有指定输出但是会有输出，`numpy`会按以下方法推断输出：

1. 输出中，下标按字母表顺序排列
2. 在两个及以上输入中出现的下标会被求和

即，对于`np.einsum("ik, kj ->", x, y)`，等价于`np.einsum("ik, kj -> ij", x, y)`。

再来一个例子[^ajcr]：

```python
M = np.random.rand(2, 3, 4)
np.einsum("kij", M) # (3, 4, 2)
```

不过还是希望大家记住[_The Zen of Python_](https://www.python.org/dev/peps/pep-0020/)中那句话：

```python
Explicit is better than implicit.
```

## 广播写法

正如刚刚例子中最后一个：

```python
A = numpy.random.rand(2, 3, 2)
B = numpy.random.rand(2, 2, 3)
np.einsum('...jk, ...kl->...jl', A,B) # （2, 3, 3)
```

`...`指代任意多个轴，这在机器学习中处理`batch`时尤为有效。

## 更多

`torch`、`tensorflow`、`paddle`等包含`einsum`函数，并**支持反向传播**，`MindSpore`的支持也在进行之中。

如果哪位大佬引用了我的文章，麻烦在文末加一个链接，谢谢！

[^paddle.einsum]: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/einsum_cn.html#einsum
[^numpy.einsum]: https://numpy.org/devdocs/reference/generated/numpy.einsum.html
[^ajcr]: https://ajcr.net/Basic-guide-to-einsum/
[^rockt]: https://rockt.github.io/2018/04/30/einsum
[^video]: https://www.youtube.com/watch?v=pkVwUVEHmfI
