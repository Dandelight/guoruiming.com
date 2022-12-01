# 特征值分解与奇异值分解

一个非 $0$ 的 $N$ 维向量 $\mathbf{v}$ 是矩阵 $A$ 的一个特征值，当且仅当对于某 $\lambda$，

$$
\begin{equation}
\label{eq:ed}
A \mathbf{v} = \lambda \mathbf{v}
\end{equation}
$$ {#eq:ed}

几何上，$A$ 的特征向向量 $\mathbf{v}$ 在被 $A$ 左乘之后，只是伸长（或缩短） $\lambda$ 倍。
## 特征多项式 {#sec}

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

但值得注意的是，埃尔米特矩阵（*Hermitian matrix*），which means
$$

\begin{equation}
A^{\mathrm{H}} \equiv \overline{\mathrm{A}^{\mathrm{T}}}=\mathrm{A} \text {, }
\end{equation}

$$
所有特征值都是实数，并且有更为高效的数值解法。
$$
