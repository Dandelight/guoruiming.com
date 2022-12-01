# 特征值分解与奇异值分解

一个非 $0$ 的 $N$ 维向量 $\mathbf{v}$ 是矩阵 $A$ 的一个特征值，当且仅当对于某 $\lambda$，

$$
\begin{equation}
\label{eq:ed}
A \mathbf{v} = \lambda \mathbf{v}
\end{equation}
$$

几何上，$A$ 的特征向向量 $\mathbf{v}$ 在被 $A$ 左乘之后，只是伸长（或缩短） $\lambda$ 倍。

## 特征多项式

对于 $\ref{eq:ed}$，我们通过移项可以得到

$$
\begin{align}
A \mathbf{v} - \lambda \mathbf{v} &= 0 \\
(\lambda I - A) \mathbf{v} &= 0
\end{align}
$$

因为 $\mathbf{v}$ 非 $0$，故 $(\lambda I - A)$ 必为奇异。采用反证法，若 $(\lambda I - A)$ 非奇异，则其可逆，则

$$
\mathbf{v} = (\lambda I- A)' \cdot 0.
$$

故求解 $\lambda$ 即为求解特征多项式

$$
\det (\lambda I - A) = 0,
$$

该多项式必有 $\operatorname{rank}(A)$ 个特征值（考虑复数域的情况下）。

但值得注意的是，埃尔米特矩阵（_Hermitian matrix_），which means

$$
\begin{equation}
A^{\mathrm{H}} \equiv \overline{\mathrm{A}^{\mathrm{T}}}=\mathrm{A} \text {, }
\end{equation}
$$

也就是对称。其所有特征值都是实数，可对角化，并且有正交的特征值。

任意一个矩阵是否有负数特征值是一个数学难题，没有简单的充分必要条件可以判断；但是有**一些充分条件**，如

1. 矩阵是负定的，所有特征值都是负的
2. 矩阵不全是 $0$ 且半负定，至少有一个负特征值
3. 实矩阵，有奇数维度，且特征值是负数的矩阵至少有一个负特征值
4. 对角阵，对角上有负数（对角阵的对角元素就是特征值）
5. 非零，实数，对称，非半正定，一定有负特征值
6. 非零， 实数，对称，迹为负数，一定有负特征值

## 奇异值分解

### 奇异值

对于任意矩阵 $A$，其特征值为 $A^*A$ 的特征值的平方根。可以通过

$$
\det(A^T A - \lambda I) = 0
$$

奇异值即为

$$
\sigma = \sqrt{\lambda}
$$

### 特征值分解

特征值分解的形式为

$$
A = U \Sigma V^T
$$

其中，

$$
A^T A = V\Sigma U^T U \Sigma V^T = V \Sigma^2 V^T
$$

即 $V$ 是 $A^TA$ 的特征值，同理

$$
AA^T = U\Sigma V^TV\Sigma U^T = U \Sigma^2 U^T
$$

如果 $A$ 是对称矩阵，则 $A$ 的奇异值分解等价于特征值分解。

```python
>>> import numpy as np
>>> a = np.random.rand(30, 10)
>>> eigval, eigvec = np.linalg.eig(a.T @ a)
>>> eigval # eigval 的结果不一定是有序的
array([78.81808649,  4.69320039,  2.78423322,  3.05038617,  2.98891652,
        0.77356156,  1.81948639,  1.17430578,  1.51506092,  1.38275923])
>>> s = np.linalg.svd(a, compute_uv=False)
>>> s # SVD 的结果是有序的
array([8.87795509, 2.16637956, 1.74653548, 1.72884832, 1.66860217,
       1.34888339, 1.23087811, 1.17590783, 1.0836539 , 0.87952348])
>>> np.sqrt(eigval)
array([8.87795509, 2.16637956, 1.66860217, 1.74653548, 1.72884832,
       0.87952348, 1.34888339, 1.0836539 , 1.23087811, 1.17590783])
>>> np.allclose(np.sort(np.sqrt(eigval)), np.sort(s)) # 相等
True
```
