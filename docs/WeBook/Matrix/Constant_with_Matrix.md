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

# 向量求导

[2]

## 一阶

$$
\frac{\partial a^T x}{\partial x} = \frac{\partial x^T a}{\partial x} = a
$$

## 二阶

$$
\frac{\partial x^T B x}{\partial x} = (B + B^T) x
$$

## 练习

证明

$$
\begin{gathered}
\underset{a \in \mathbb{R}^m}{\min} \frac{1}{2} \| x - D\alpha \|_2^2 + \lambda \| \alpha \|_2^2, \quad x \in \mathbb{R}^n, D\in\mathbb{R}^{n\times m} \\
\Rightarrow \alpha= (DD^T + \lambda I)^{-1}D^T x
\end{gathered}
$$

# 矩阵的迹

## 定义

$$
\operatorname{Tr}(A) = \sum_{i=1}^n a_{ii}, \quad A = (a_{ij}) \in \mathbb{R}^{n\times n}.
$$

## 性质

$$
\begin{gathered}
\text{for } A, B, C\in \mathbb{R}^{n\times n}, a \in \mathbb{R}\\
\|A\|_F^2 = \sum_{i=1}^n\sum_{j=1}^n a_{ij}^2 = \operatorname{Tr}(A^T A), \\
\operatorname{Tr}(A) = \operatorname{Tr}(A^T), \\
\operatorname{Tr}(A+B) = \operatorname{Tr}(B+A), \\
\operatorname{Tr}(aA) = a\operatorname{Tr}(A), \\
\operatorname{Tr}(AB) = \operatorname{Tr}(BA), \\
\operatorname{Tr}(ABC) = \operatorname{Tr}(BCA) = \operatorname{Tr}(CAB).
\end{gathered}
$$

## 迹的导数

### First order

$$
\begin{gathered}
\frac{\partial}{\partial X} \operatorname{Tr}(XA) = A^T \\
\operatorname{Tr}(X^T A) = A
\end{gathered}
$$

### Second order

$$
\begin{gathered}
\frac{\partial}{\partial X}\operatorname{Tr}(X^T X A) = XA^T + XA \\
\frac{\partial}{\partial X}\operatorname{Tr}(X^T B X) = B^T X + B X \\
\end{gathered}
$$

### 练习

$$
\begin{gathered}
\underset{A \in \mathbf{R}^{k\times m}}{\min}\| X - D A\|_F^2 + \lambda \| A \|_F^2, \quad X \in \mathbb{R}^{n\times m}, D \in \mathbb{n\times k} \\
\Rightarrow A = (D^TD+\lambda I)^{-1}D^TX
\end{gathered}
$$

## Reference

[1] <https://zhuanlan.zhihu.com/p/411057937>
[2] CSE 902: Selected Topics in Recognition by Machine, Anil Jain, Michigan State University.
