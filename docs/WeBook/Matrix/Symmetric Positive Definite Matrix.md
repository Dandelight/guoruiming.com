# 对称正定矩阵

本文译自：<https://nhigham.com/2020/07/21/what-is-a-symmetric-positive-definite-matrix/>

$n\times n$ 的矩阵 $A$ 为实对称矩阵，当且仅当 $A = A^T$，且

$$
x^T A X > 0 \quad \forall\ x \neq 0.
$$

我们可以发现

$$
\begin{aligned}\
a_{ii} &> 0                   &\forall\ i, \\
a_{ij} &< \sqrt{a_{ii}a_{jj}} &\forall\ i \neq j.
\end{aligned}
$$

当然，这组不等式不是正定对称矩阵的充分条件。

对于一个对称阵，其为正定矩阵的充分条件是对角元素为正且主对角占优，即 $a_{ii} > \sum_{j \neq i} |a_{ij}|\ \forall \ i$。
