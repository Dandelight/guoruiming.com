# 常量和矩阵转换技巧

在此整理一下常量和矩阵转换的技巧，以备查阅。

## 二次型

$$
\sum_i^n \sum_j^n x_i x_j \mathbf{Q}_{i, j}=\mathbf{x}^T \mathbf{Q} \mathbf{x}
$$

## 协方差矩阵

$\mathbf{C}_{ij}$ 是 $\mathbf{X}$ 的第 $i$ 个样本和第 $j$ 个样本的协方差（假设 $\mathbf{X}$ 已去过均值）

$$
\mathbf{C}_{i, j}=\frac{1}{N} \sum_k^n \mathbf{X}_{i, k} \mathbf{X}_{j, k}
$$

我们可以发现

$$
\left[\mathbf{x}_k \mathbf{x}_k^T\right]_{i, j}=\mathbf{X}_{i, k} \mathbf{X}_{j, k}
$$

同时 $\mathbf{C}_{i,j} = \frac{1}{N}\sum_k^n \mathbf{X}_{i,k}\mathbf{X}_{j,k}$ 还可以看作是 $\mathbf{X}$ 的第 $i$ 个样本和第 $j$ 个样本的内积，所以

$$
\mathbf{C}=\frac{1}{N} \mathbf{X} \mathbf{X}^T
$$

## Reference

[1] <https://zhuanlan.zhihu.com/p/411057937>
