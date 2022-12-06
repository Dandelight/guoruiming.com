2DPCA 是针对图像特征提取提出的数据降维方法。相比 PCA，2DPCA 用于处理矩阵而不是向量。

## 算法

我们的目标是将一个矩阵（可以是一张输入图像） $\mathbf{A} \in \mathbb{R}^{m\times n}$ 投影到一个单位向量 $\mathbf{X}\in \mathbb{R}^{n\times 1}$ 上，得到一个投影向量 $\mathbf{Y} \in \mathbb{R}^{m\times 1}$。写成线性变换的形式为

$$
\mathbf{Y} = \mathbf{AX}.
$$

为了求解 $\mathbf{X}$，作者引入了 total scatter 的概念。投影后的样本可以被投影后 feature vector 的协方差矩阵的迹来表达。所以，作者采用了以下标准：

$$
J(\mathbf{X}) = \operatorname{tr}(\mathbf{S}_x),
$$

其中 $\mathbf{S}_x$ 表示训练样本的投影点的协方差矩阵。可以展开为

$$
\begin{aligned}
\mathbf{S}_x &=E(\mathbf{Y}-E \mathbf{Y})(\mathbf{Y}-E \mathbf{Y})^T \\
&=E[\mathbf{A X}-E(\mathbf{A X})][\mathbf{A} \mathbf{X}-E(\mathbf{A X})]^T \\
& =E[(\mathbf{A}-E \mathbf{A}) \mathbf{X}][(\mathbf{A}-E \mathbf{A}) \mathbf{X}]^T,
\end{aligned}
$$

其中 $E(\cdot)$ 表示期望（平均值）。于是，

$$
\operatorname{tr}\left(\mathbf{S}_x\right)=\mathbf{X}^T\left[E(\mathbf{A}-E \mathbf{A})^T(\mathbf{A}-E \mathbf{A})\right] \mathbf{X}.
$$

我们定义

$$
\mathbf{G}_t = E[(\mathbf A - E\mathbf A)^T (\mathbf A - E \mathbf A)],
$$

并称 $\mathbf{G}_t$ 为**图像方差矩阵（image covariance matrix）**。现在假设我们共有 $M$ 个训练样本，第 $j$ 个样本用 $\mathbf{A}_j(j=1, 2, \ldots, M)$ 表示，所有训练样本的均值用 $\overline{\mathbf{A}}$ 表示。故 $\mathbf{G}$ 可写为

$$
\mathbf{G}_t=\frac{1}{M} \sum_{j=1}^M\left(\mathbf{A}_j-\overline{\mathbf{A}}\right)^T\left(\mathbf{A}_j-\overline{\mathbf{A}}\right)
$$

可得

$$
J(\mathbf{X}) = \mathbf{X}^T \mathbf{G}_t \mathbf{X},
$$

其中 $\mathbf{X}$ 为单位向量。该准则称为 **generalized total scatter criterion**。最大化该准则的 $\mathbf{X}$ 被称为 **optimal projection axis**。简而言之，最优的投影轴 $\mathbf{X}_\mathit{opt}$ 是最大化 $J(\mathbf{X})$ 的单位向量，数学形式为

$$
\begin{gathered}
\arg \max_{\{\mathbf{X}_1, \ldots, \mathbf{X}_n\}} J(\mathbf(X)) \\
\text{s. t. } \mathbf{X}_i^T \mathbf{X}_j = 0, i\neq j, i, j = 1, \ldots, d
\end{gathered}
$$

可知最优的 $\mathbf{X}_1, \ldots, \mathbf{X}_d$ 为 $\mathbf{G}_t$ 的最大 $d$ 个特征值对应的特征向量。

## 特征提取

我们得到最优投影向量 $\mathbf{X}_1, \ldots, \mathbf{X}_n$ 之后，可以使用其进行图像的特征提取。对于一个图像样本 $\mathbf{A}$，设

$$
\mathbf{Y}_k = \mathbf{AX}_k, \quad k = 1, 2, \ldots, d.
$$

如是，我们得到了一组投影后的 feature vector $\mathbf{Y}_1, \ldots, \mathbf{Y}_d$，称之为图像 $\mathbf{A}$ 的**主成分（principal component）**。PCA 的主成分是标量，而 2DPCA 的主成分是向量。主成分可组成一个 $m\times d$ 的矩阵

$$
\mathbf{B}_i = [\mathbf{Y}_1^{(i)}, \mathbf{Y}_2^{(i)}, \ldots, \mathbf{Y}_d^{(i)}],
$$

称其为图像样本 $\mathbf{A}$ 的**特征矩阵（feature matrix）** 或 **特征图像（feature image)**。

## 分类

在 2DPCA 中，使用同一套 feature vector，每一张图片都能得到一个特征矩阵，及降维结果。作者称可使用最近邻法进行分类。设 $\mathbf{B}_i = [\mathbf{Y}_1^{(i)}, \mathbf{Y}_2^{(i)}, \ldots, \mathbf{Y}_d^{(i)}]$，$\mathbf{B}_j = [\mathbf{Y}_1^{(j)}, \mathbf{Y}_2^{(j)}, \ldots, \mathbf{Y}_d^{(j)}]$ 分别是第 $i$ 和第 $j$ 张图片的特征矩阵，则二者的距离定义为

$$
d\left(\mathbf{B}_i, \mathbf{B}_j\right)=\sum_{k=1}^d\left\|\mathbf{Y}_k^{(i)}-\mathbf{Y}_k^{(j)}\right\|_2
$$

即各 feature vector 的欧氏距离之和。

## 图像重建

在 Eigenfaces 算法中，我们可以通过特征向量的线性组合重构人脸。类似的，2DPCA 算法也可以用来重构人脸。设 $\mathbf{V}=\left[\mathbf{Y}_1, \cdots, \mathbf{Y}_d\right]$，$\mathbf{U}=\left[\mathbf{X}_1, \cdots, \mathbf{X}_d\right]$，则

$$
\mathbf{V} = \mathbf{AU}
$$

因为 $\mathbf{U}$ 的列向量正交，则我们可以重构图像样本 $\mathbf{A}$

$$
\tilde{\mathbf{A}}=\mathbf{V} \mathbf{U}^T=\sum_{k=1}^d \mathbf{Y}_k \mathbf{X}_k^T
$$

设 $\tilde{\mathbf{A}} = \mathbf{Y}_k \mathbf{X}_k^T \in \mathbb{R}^{m\times n}(k = 1, 2, \ldots, d)$，表示$\mathbf{A}$ 的一张子图像。$\mathbf{A}$ 可视为前 $d$ 张子图像的和。当 $d = n$，有 $\tilde{\mathbf{A}} = \mathbf{A}$。否则，当 $d<n$，$\tilde{\mathbf{A}}$ 是 $\mathbf{A}$ 的近似。

## Reference

Jian Yang, D. Zhang, A. F. Frangi and Jing-yu Yang, "Two-dimensional PCA: a new approach to appearance-based face representation and recognition," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 26, no. 1, pp. 131-137, Jan. 2004, doi: 10.1109/TPAMI.2004.1261097. <https://ieeexplore.ieee.org/document/1261097>

```bibtex
@article{1261097,
  author  = {Jian Yang and Zhang, D. and Frangi, A.F. and Jing-yu Yang},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title   = {Two-dimensional PCA: a new approach to appearance-based face representation and recognition},
  year    = {2004},
  volume  = {26},
  number  = {1},
  pages   = {131-137},
  doi     = {10.1109/TPAMI.2004.1261097}
}

```
