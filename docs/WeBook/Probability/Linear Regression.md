# 线性回归

线性回归是最简单的有监督学习模型之一。有监督学习中，**训练样本**由成对的**特征（feature）**和**目标（target）**构成，由 $\{(x^{(i)}, y^{(i)})\}$ 表示，其中上标 $(i)$ 表示第 $i$ 个样本。我们的任务是找到一个函数 $h: \mathcal{X} \mapsto \mathcal{Y}$ 使得 $h(x)$ 可以尽可能接近 $y$。如果 $y$ 是连续的，该任务称为**分类任务（Classification）**；如果 $y$ 是离散的，该任务称为**回归任务（Regrerssion）**。

在线性回归中，我们假设 $y$ 可以表示为 $x$ 的一个线性函数，即

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_d x_d
$$

其中，$\theta_i$ 被称为参数或权重。为了简化表达，我们设定 $x_0 = 1$（**intercept term**），以上公式化为

$$
h_\theta(x) = \sum_{i=0}^d \theta_i x_i = \boldsymbol{\theta}^T \boldsymbol{x}
$$

其中 $\boldsymbol{\theta}$ 和 $\boldsymbol{x}$ 为列向量。下面我们从概率论的角度解决该问题。

问题是，通过观测得到的 $x^{(i)}$ 和 $y^{(i)}$ 可能不是严格的线性关系，否则用两点确定一条直线的原理就解决了。我们假设 $y$ 没有被 $x$ 完美地表现出来——存在误差项 $\epsilon$ 使得

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}.
$$

误差项描述了由未被模型捕获的效果，或者随机噪声。我们进一步假设 $\epsilon$ 是**独立同分布（Independently and identically distributed）**于均值为 $0$、方差为 $\sigma^2$ 的高斯分布 $\mathcal{N}(0, \sigma^2)$，即，$\epsilon^{(i)}$ 的密度由

$$
p\left( \epsilon^{(i)} \right) = \frac{1}{\sqrt{2\pi}\sigma}
\exp\left( - \frac{\left( \epsilon^{(i)} \right)^2}{2\sigma ^2} \right)
$$

给出。

> 在概率论与统计学中，**独立同分布**（英语：Independent and identically distributed，或称独立同分配，缩写为 iid、 i.i.d.、IID）是指一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立。
>
> - 假设有两个变量 $X$ 和 $Y$
> - $X$ 和 $Y$ 同分布，当且仅当 $P[x \ge X] = P[x\ge Y],\ \forall x \in \mathcal{I}$
> - $X$ 和 $Y$ 独立，当且仅当 $(P[y \geq Y]=P[y \geq Y \mid x \geq X]) \wedge (P[x \geq X]=P[x \geq X \mid y \geq Y])\ \forall x, y \in \mathbb{I}$
>
> 一组随机变量独立同分布并不意味着它们的样本空间中每个事件发生概率都相同。例如，投掷非均匀骰子得到的结果序列是独立同分布的，但掷出每个面朝上的概率并不相同。
> <https://zh.wikipedia.org/wiki/独立同分布>

当 $\sigma=1$ 时，该函数形态为

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 100)
sigma = 1
y = np.exp(- x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) * sigma
plt.plot(x, y)
plt.savefig("gaussian.svg")
```

![[gaussian.svg]]
这说明了

$$
p\left(y^{(i)} \mid x^{(i)} ; \boldsymbol{\theta}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\left(y^{(i)}-\boldsymbol{\theta}^T x^{(i)}\right)^2}{2 \sigma^2}\right),
$$

其中，$p\left(y^{(i)} \mid x^{(i)} ; \theta\right)$ 表示 $y^{(i)}$ 的分布在 $\theta$ 做参数的情况下由 $x^{(i)}$ 给出。注意不是 $p\left(y^{(i)} \mid x^{(i)}, \theta\right)$，因为 $\theta$ 不是随机变量。该分布亦可记作 $y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}\left(\theta^T x^{(i)}, \sigma^2\right)$。

现将等式矩阵化。设矩阵 $X = [x^{(1)}, x^{(2)}, \ldots, x^{(n)}]^T \in \mathbb{R}^{n\times d}$（以数据为行向量）和 $\boldsymbol{\theta}$，$y^{(i)}$ 的分布是什么样的？该量由 $p(\vec{y} \mid X; \theta)$ 给出。该量通常被看作对于固定的 $\theta$，$\vec{y}$（或 $X$）的函数。但现在，我们将其视为 $\boldsymbol{\theta}$ 的函数，即**似然函数（likelihood function）**。

$$
L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} \mid X; \theta).
$$

同时因为 $\epsilon$ 的独立性假设，给定 $x^{(i)}$ 后 $y^{(i)}$ 也是独立的，则

$$
\begin{aligned}
L(\theta) & =\prod_{i=1}^n p\left(y^{(i)} \mid x^{(i)} ; \theta\right) \\
& =\prod_{i=1}^n \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2 \sigma^2}\right) .
\end{aligned}
$$

该概率模型将 $y^{(i)}$ 与 $x^{(i)}$ 联系起来。现在的问题是，如何选取 $\theta$？**极大似然**准则指出，我们应最大化数据的概率，即选择 $\theta$ 来最大化 $L(\theta)$。

虽然话是这么说，但直接最大化一个指数函数不是很容易。我们可以最大化任意一个 $L(\theta)$ 的单调递增函数，于是有了**对数似然函数（log likelihood function）**

$$
\begin{aligned}
\ell(\theta) & =\log L(\theta) \\
& =\log \prod_{i=1}^n \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2 \sigma^2}\right) \\
& =\sum_{i=1}^n \log \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2 \sigma^2}\right) \\
& =\underbrace{n \log \frac{1}{\sqrt{2 \pi} \sigma}}_\text{常数}-\frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^n\left(y^{(i)}-\theta^T x^{(i)}\right)^2 .
\end{aligned}
$$

于是最大化 $\ell(\theta)$ 等价于最小化

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^n\left( y^{(i)} - \theta^T x^{(i)} \right).
$$

该函数有一个漂亮的闭式解，对 $J(\theta)$ 求导可得

$$
\begin{aligned}
\nabla_\theta J(\theta) & =\nabla_\theta \frac{1}{2}(X \theta-\vec{y})^T(X \theta-\vec{y}) \\
& =\frac{1}{2} \nabla_\theta\left((X \theta)^T X \theta-(X \theta)^T \vec{y}-\vec{y}^T(X \theta)+\vec{y}^T \vec{y}\right) \\
& =\frac{1}{2} \nabla_\theta\left(\theta^T\left(X^T X\right) \theta-\vec{y}^T(X \theta)-\vec{y}^T(X \theta)\right) \\
& =\frac{1}{2} \nabla_\theta\left(\theta^T\left(X^T X\right) \theta-2\left(X^T \vec{y}\right)^T \theta\right) \\
& =\frac{1}{2}\left(2 X^T X \theta-2 X^T \vec{y}\right) \\
& =X^T X \theta-X^T \vec{y}
\end{aligned}
$$

解为（假定 $X^TX$ 可逆）

$$
\theta (X^T X)^{-1}X^T\vec{y}.
$$

这是我们高中阶段就已经熟悉的**最小二乘法**，但更深刻地看待，其意义是**在误差服从高斯分布的情况下，最小化误差**。

### 正则化线性回归

过拟合（overfitting）问题指一个模型的效果在某个特定的数据集上表现很好，但泛化能力差的现象。过拟合困扰着大多数机器学习模型，尤其是模型复杂（参数量大）的时候。我们可以减少参数，也可以使用一个参数的函数（如 $\ell_2$ 范数）来控制模型的复杂程度。典型的正则化方法是在目标函数上加上一项 $R(\theta)$。

在之前，我们将 $\theta$ 视为未知但确定的参数，但我们也可以认为 $\theta$ 是未知的随机变量。在此设定下，我们认为 $\theta$ 服从某先验分布 $p(\theta)$。该先验分布通常来源于我们的先验知识。给定数据集 $S=\{\left( x^{(i)}, y^{(i)} \right)\}_{i=1}^N$，我们需要根据现有观测找到最可能的 $\theta$，该问题用数学表示为

$$
\hat{\theta}_\mathit{MAP} = \arg \underset{\theta}{\max}\ p(\theta \mid S),
$$

其中 $p(\theta \mid S)$ 是后验分布

CS229 Lecture Notes, Andrew Ng. <https://cs229.stanford.edu/>
