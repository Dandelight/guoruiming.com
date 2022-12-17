Linear Discriminant Analysis 又称 Fisher Discriminant Analysis 是一种有监督的降维方法,其主要思想是**最小化类内方差，最大化类间方差**。

LDA 旨在寻找一个最优的投影矩阵，将高维数据投影到稠密的低维空间中，借助 Fisher 判别准则最小化投影数据的类内离散度，同时最大化投影数据的类间离散度。

$$
\mathbf{W}=\underset{\mathbf{W}^T \mathbf{W}=\mathbf{I}}{\arg \max } \frac{\operatorname{tr}\left(\mathbf{W}^T \mathbf{S}_b \mathbf{W}\right)}{\operatorname{tr}\left(\mathbf{W}^T \mathbf{S}_w \mathbf{W}\right)}
$$

其中，

$$
\mathbf{S}_w=\sum_{i=1}^c \sum_{j=1}^{n_i}\left(\mathbf{x}_j^i-\overline{\mathbf{x}}_i\right)\left(\mathbf{x}_j^i-\overline{\mathbf{x}}_i\right)^T
$$

为函数的类内离散度矩阵（within-class scatter matrix），

$$
\mathbf{S}_b=\sum_{i=1}^c n_i\left(\overline{\mathbf{x}}_i-\overline{\mathbf{x}}\right)\left(\overline{\mathbf{x}}_i-\overline{\mathbf{x}}\right)^T
$$

为函数的类间离散度矩阵（between-class scatter matrix），其中 $n_i(i=1, \ldots, c)$ 表示第 $i$ 类样本的数量，$\overline{\mathbf{x}}\in\mathbb{R}^{d\times 1}$ 表示所有训练样本的均值，$\overline{\mathbf{x}}_i\in\mathbb{R}^{d\times 1}$ 表示第 $i$ 类样本的均值，$\mathbf{x}^i_j\in\mathbb{R}^{d\times 1}$ 表示第 $i$ 类样本集合中的第 $j$ 个样本。可以观察到，

$$
\mathbf{S}_t = \mathbf{S}_w + \mathbf{S}_b,
$$

即总方差等于类内方差加类间方差。

目标函数是一个迹的比值形式的优化问题，通常是非凸的，常用的方法是将其转化为一个更为简单但不精确的比值的迹的形式。对应形式如下

$$
\begin{aligned}
\mathbf{W} & =\underset{\mathbf{W}^T \mathbf{W}=\mathbf{I}}{\arg \max } \operatorname{tr}\left(\left(\mathbf{W}^T \mathbf{S}_w \mathbf{W}\right)^{-1}\left(\mathbf{W}^T \mathbf{S}_b \mathbf{W}\right)\right) \\
& =\underset{\mathbf{W}^T \mathbf{W}=\mathbf{I}}{\arg \max } \frac{\left|\mathbf{W}^T \mathbf{S}_b \mathbf{W}\right|}{\left|\mathbf{W}^T \mathbf{S}_w \mathbf{W}\right|},
\end{aligned}
$$

其中，因为 $\mathbf{w}$ 仅指方向，其大小并没有实际意义，所以约束其为单位向量；$\mathbf{w}$ 两两正交，故有 $\mathbf{W}^T\mathbf{W}= \mathbf{I}$。

假设 $\mathbf{S}_w$ 为满秩矩阵，故 $\mathbf{S}_w^{-1}$ 存在。设 $\mathbf{S}_w^{1/2}$ 为对称、正定矩阵，满足 $\mathbf{S}_w = \mathbf{S}_w^{1/2}\mathbf{S}_w^{1/2}$，设其逆为 $\mathbf{S}_w^{-1/2}$。设

$$
\mathbf{z} = \mathbf{S}_w^{1/2}\mathbf{w}
$$

则准则函数变为

$$
\frac{\mathbf{v}^{\mathrm{T}} \mathbf{S}_b \mathbf{v}}{\mathbf{v}^{\mathrm{T}} \mathbf{S}_w \mathbf{v}}=\frac{\mathbf{z}^{\mathrm{T}} \mathbf{S}_w^{-\frac{1}{2}} \mathbf{S}_b \mathbf{S}_w^{-\frac{1}{2}} \mathbf{z}}{\mathbf{z}^{\mathrm{T}} \mathbf{z}}
$$

因为

$$
\begin{aligned}
\mathbf{v}^T \mathbf{v} &= 1 \\
{\left(\mathbf{S}_w^{-1/2} \mathbf{z}\right)}^T {\mathbf{S}_w^{-1/2} \mathbf{z}} &= 1 \\
\mathbf{z}^T \mathbf{z} &= 1
\end{aligned}
$$

即我们发现，$\mathbf{z}$ 也为单位向量。通过拉格朗日乘子法，我们可以得到

$$
\mathbf{S}_w^{-\frac{1}{2}} \mathbf{S}_b \mathbf{S}_w^{-\frac{1}{2}} \mathbf{z}_1=\lambda \mathbf{z}
$$

左乘 $\mathbf{S}_w^{-1/2}$ 可得

$$
\mathbf{S}_w^{-1} \mathbf{S}_b\left(\mathbf{S}_w^{-\frac{1}{2}} \mathbf{z}\right)=\lambda \left(\mathbf{S}_w^{-\frac{1}{2}} \mathbf{z}\right), \quad \Rightarrow \quad \mathbf{S}_w^{-1} \mathbf{S}_b \mathbf{v} = \lambda \mathbf{v}
$$

由此，我们将原问题转化为了一个特征值分解问题，最优 $\mathbf{v}$ 即为 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 的特征值。

运用拉格朗日乘子法，我们可以得到

$$
\phi(\mathbf{W}, \lambda)=\mathbf{W}^T S_B \mathbf{W}-\lambda\left(\mathbf{W}^T S_W \mathbf{W}-1\right)
$$

令 $\frac{\partial \phi}{\partial \mathbf{W}} = 1$ 可得

$$
\begin{aligned}
\frac{\partial \phi}{\partial \mathbf{w}}=2 S_B \mathbf{w}-\lambda 2 S_W \mathbf{w} & =0 \\
S_B \mathbf{w} & =\lambda S_W \mathbf{w}
\end{aligned}
$$

该优化问题可以通过广义特征值分解求解[^liao]。

## 瑞利商（Rayleigh quotient）与广义瑞利商（genralized Rayleigh quotient）

瑞利商指这样的函数

$$
R(A, x) = \frac{x^H A x}{x^Hx}
$$

其中 $x$ 为非零向量，$A\in \mathbb{R}^{n\times n}$ 为 Hermitian 矩阵，即 $A^H = A$。瑞丽商 $R(A, x)$ 的重要性质是，其最大值等于 $A$ 最大的特征值，最小值等于 $A$ 最小的特征值，也就是，

$$
\lambda_\mathit{min} \le \frac{x^H A x}{x^Hx} \le \lambda_\mathit{max}
$$

当向量 $x$ 正交时，$x^Hx=1$，此时瑞利商退化为

$$
R(A, x) = x^H A x
$$

该形式在 PCA 和 Laplacian Eigenmaps 等算法中均有出现。

广义瑞利商指

$$
R(A, B, x) = \frac{x^H A x}{x^H B x}
$$

其中 $A, B \in \mathbb{R}^{n\times n}$ 为 Hermitian 矩阵，$B$ 为正定矩阵。我们可以将其转化为瑞利商的形式。令 $x = B^{-1/2}x'$，则分母转化为

$$
x^H B x=x^{\prime H}\left(B^{-1 / 2}\right)^H B B^{-1 / 2} x^{\prime}=x^{\prime H} B^{-1 / 2} B B^{-1 / 2} x^{\prime}=x^{\prime H} x^{\prime}
$$

而分子转化为

$$
x^H A x=x^{\prime H} B^{-1 / 2} A B^{-1 / 2} x^{\prime}
$$

此时 $R(A, B, x)$ 变为

$$
R(A, B, x') = \frac{x'^H B^{-1/2} A B^{-1/2} x'}{x'^H x'}
$$

则我们知道 $R(A, B, x')$ 的最大特征值为矩阵 $B^{-1/2} A B^{-1/2}$ 的最大特征值，或者说，是 $B^{-1}A$ 的最大特征值，而最小值为 $B^{-1}A$ 的最小特征值。

## 二类 LDA

对于有 $c$ 个类别的数据集，LDA 最多将其分为 $c-1$ 类。对于第 $j$ 类样本，定义其均值

$$
\mu_j=\frac{1}{N_j} \sum_{x \in X_j} x(j=0,1)
$$

方差

$$
\Sigma_j=\sum_{x \in X_j}\left(x-\mu_j\right)\left(x-\mu_j\right)^T \quad(j=0,1)
$$

我们需要让不同类别均值距离 $\left\| w^T \mu_0 - w^T \mu_1 \right\|_2^2$ 尽可能大，同时同一类别中的协方差 $w^T \sum_0 w$ 和 $w^T \sum_1 w$ 尽可能小。==虽然这段是我最看不懂的==，但综上所述，我们的优化目标为：

$$
\underbrace{\arg \max }_w J(w)=\frac{\left\|w^T \mu_0-w^T \mu_1\right\|_2^2}{w^T \Sigma_0 w+w^T \Sigma_1 w}=\frac{w^T\left(\mu_0-\mu_1\right)\left(\mu_0-\mu_1\right)^T w}{w^T\left(\Sigma_0+\Sigma_1\right) w}
$$

也有写作乘积之和的

$$
\underbrace{\arg \max }_W J(W)=\frac{\prod_{\operatorname{diag}} W^T S_b W}{\prod_{\operatorname{diag}} W^T S_w W}
$$

其中 $\prod_{diag}$ 为矩阵对角线元素之积。

$$
J(W)=\frac{\prod_{i=1}^d w_i^T S_b w_i}{\prod_{i=1}^d w_i^T S_w w_i}=\prod_{i=1}^d \frac{w_i^T S_b w_i}{w_i^T S_w w_i}
$$

一段 `MATLAB` 实现[^dae]

```matlab
% FLD classifies an input sample into either class 1 or class 2.
%
%   [output_class w] = myfld(input_sample, class1_samples, class2_samples)
%   classifies an input sample into either class 1 or class 2,
%   from samples of class 1 (class1_samples) and samples of
%   class 2 (class2_samples).
%
% Input parameters:
% input_sample = an input sample
%   - The number of dimensions of the input sample is N.
%
% class1_samples = a NC1xN matrix
%   - class1_samples contains all samples taken from class 1.
%   - The number of samples is NC1.
%   - The number of dimensions of each sample is N.
%
% class2_samples = a NC2xN matrix
%   - class2_samples contains all samples taken from class 2.
%   - The number of samples is NC2.
%   - NC1 and NC2 do not need to be the same.
%   - The number of dimensions of each sample is N.
%
% Output parameters:
% output_class = the class to which input_sample belongs.
%   - output_class should have the value either 1 or 2.
%
% w = weight vector.
%   - The vector length must be one.
%
function [output_class w] = myfld(input_sample, class1_samples, class2_samples)
[m1, n1] = size(class1_samples);
[m2, n2] = size(class2_samples);
mu1 = sum(class1_samples) / size(class1_samples, 1);
mu2 = sum(class2_samples) / size(class2_samples, 1);
s1 = 0;
s2 = 0;

for i = 1:m1
    s1 = s1 + (class1_samples(i,:) - mu1)' * (class1_samples(i,:) - mu1);
end
for i = 1:m2
    s2 = s2 + (class2_samples(i,:) - mu2)' * (class2_samples(i,:) - mu2);
end
sw = s1 + s2;
w = inv(sw)' * (mu1 - mu2)';
w1 = w(1) / (w(1) * w(1) + w(2) * w(2))^0.5;
w2 = w(2) / (w(1) * w(1) + w(2) * w(2))^0.5;
w = [w1; w2];
separationPoint = (mu1 + mu2) * w * 0.5;
output_class = (input_sample * w < separationPoint) + 1;
```

一段 `Python` 实现[^ljpzzz]

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
from sklearn.datasets.samples_generator import make_classification
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,
                           n_clusters_per_class=1,class_sep =0.5, random_state =10)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print pca.explained_variance_ratio_
print pca.explained_variance_
```

PCA 和 LDA 的相同点有：

1. 两者均可以对数据进行降维。
2. 两者在降维时均使用了矩阵特征分解的思想。
3. 两者都假设数据符合高斯分布。

不同点有：

1. LDA 是有监督的降维方法，而 PCA 是无监督的降维方法
2. LDA 降维最多降到类别数 k-1 的维数，而 PCA 没有这个限制。
3. LDA 除了可以用于降维，还可以用于分类。
4. LDA 选择分类性能最好的投影方向，而 PCA 选择样本点投影具有最大方差的方向。

LDA 算法的主要优点有：

1. 在降维过程中可以使用类别的先验知识经验，而像 PCA 这样的无监督学习则无法使用类别先验知识。
2. LDA 在样本分类信息依赖均值而不是方差的时候，比 PCA 之类的算法较优。

LDA 算法的主要缺点有：

1. LDA 不适合对非高斯分布样本进行降维，PCA 也有这个问题。
2. LDA 降维最多降到类别数 k-1 的维数，如果我们降维的维度大于 k-1，则不能使用 LDA。当然目前有一些 LDA 的进化版算法可以绕过这个问题。
3. LDA 在样本分类信息依赖方差而不是均值的时候，降维效果不好。
4. LDA 可能过度拟合数据。[^pinard]

[^liao]: 廖爽丽. 维数约简关键技术研究[D].西安电子科技大学,2021.DOI:10.27389/d.cnki.gxadu.2021.000214.
[^dae]: https://cloud.tencent.com/developer/article/1835856
[^ljpzzz]: https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/lda.ipynb
[^pinard]: 线性判别分析 LDA 原理总结. 刘建平. https://www.cnblogs.com/pinard/p/6244265.html
