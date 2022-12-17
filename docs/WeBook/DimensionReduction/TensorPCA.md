上回书说到 [[2DPCA]]，将 PCA 扩展到了二维。依照类似的思路，我们可以将 PCA 扩展到任意高维，即 TensorPCA。这也将作为我们介绍 Graph Embedding 的基础。

定义相同维度的两个张量 $\mathbf{A} \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_n}$ 与 $\mathbf{B} \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_n}$，定义张量内积

$$
\langle \mathbf{A}, \mathbf{B} \rangle = \sum_{i_1=1, \ldots, i_n=1}^{i_1=m_1, \ldots, i_n=m_n} \mathbf{A}_{i_1, \ldots, i_n} \mathbf{B}_{i_1, \ldots, i_n},
$$

及张量范数

$$
\| \mathbf{A} \| = \sqrt{\langle \mathbf{A}, \mathbf{A} \rangle},
$$

定义 $\mathbf{A}$ 与 $\mathbf{B}$ 之间的距离为

$$
\| \mathbf{A} - \mathbf{B} \|.
$$

在二阶情况下，该范数也被成为 Frobenius 范数，记作 $\| \mathbf{A} \|_F$。

张量 $\mathbf{A}$ 与矩阵 $U \in \mathbb{R}^{m_k \times m_k}$ 的 $k$-mode product 定义为

$$
\mathbf{B} = \mathbf{A} \times_k U,
$$

where $\mathbf{B}_{i_1, \ldots, i_k, j, i_{k+1}, \ldots, i_n} = \sum_{i=1}^{m_k} A_{i_1, \ldots, i_k, i, i_{k+1}, \ldots, i_n} \times U_{ij}, j = 1, \ldots, m'_k$。

设我们的输入样本为

$$
\left\{\mathbf{X}_{\mathbf{i}} \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_n}, i=1,2, \ldots, N\right\}
$$

类比于 PCA，我们假设降维结果是一个低维度张量。为了便于叙述，假设将张量降到一维。在该情况下，

$$
y_i = \mathbf{X}_i \times_1 w^1 \times_2 w^2 \cdots \times_n w^n.
$$

目标函数即可表示为

$$
\begin{aligned}
\left(w^1, \ldots, w^n\right)^*= & \underset{f\left(w^1, \ldots, w^n\right)=d}{\arg \min } \sum_{i \neq j} \| \mathbf{X}_{\mathbf{i}} \times_1 w^1 \times_2 w^2 \ldots \times_n w^n
-\mathbf{X}_{\mathbf{j}} \times_1 w^1 \times_2 w^2 \ldots \times_n w^n \|^2 W_{i j}
\end{aligned}
$$

其中，如果 $B$ 是通过限制大小（scale normalization），则

$$
f(w^1, \ldots, w^n) = \sum_{i=1}^n \| \mathbf{X}_i \times_1 w^1 \times_2 w^2 \cdots \times w^n \|^2 B_{ii},
$$

若 $B$ 从惩罚图中产生，即

$$
B = L^p = D^p - W^p
$$

则

$$
\begin{aligned}
f\left(w^1, \ldots, w^n\right)= & \sum_{i \neq j} \| \mathbf{X}_{\mathbf{i}} \times_1 w^1 \times_2 w^2 \ldots \times_n w^n-\mathbf{X}_{\mathbf{j}} \times_1 w^1 \times_2 w^2 \ldots \times_n w^n \|^2 W_{i j}^p
\end{aligned}
$$

比较难顶的是，该目标函数通常没有闭式解。但是，我们可以选择一个 $i$，优化 $w_i$，而优化其他矩阵。~~当然，这个东西估计也只具有理论意义。~~
