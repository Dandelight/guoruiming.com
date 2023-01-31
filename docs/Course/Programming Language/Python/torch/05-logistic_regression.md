首先我们考虑二分类任务，其输出的标记为 $y \in \{0, 1\}$，所以要将实数值 $x$ 映射到 $y$ 的定义域上。最理想的是 Heaviside 函数即单位阶跃函数，但是 Heaviside 函数的缺点是**不可微**，也就没办法使用梯度下降法进行优化求解。

> 连续不一定可导，可导一定连续

所以我们找到一个类似 Heaviside 函数，同时可微的函数，如下

$$
y= H(x) = \sigma(W;x) = \frac{1}{1 + e^{-Wx}} \\
$$

该函数因为函数形态类似英文字母 S 故被形象地称为 sigmoid 函数。

若预测结果大于零，判断为正例，若小于零判断为负例，若为零则任意判断。

上式可变形为

$$
\ln\frac{y}{1-y} = Wx
$$

将 $y$ 视作样本 $x$ 作为正例的可能性，$1-y$ 为样本为范例的可能性，$\frac{y}{1-y}$ 表示了 $x$ 作为正例的相对可能性，称为 **logit**。由此看出，Logistic Regression 是在用一个线性模型 $Wx$ 拟合一个 logit。

于是我们可以通过**极大似然法**估计 $W$

$$
\ell(W)=\sum_{i=1}^m \underbrace{{\color{blue}y_i} \log(\sigma(W;x_i))}_{\text{case for } y = 1} + \underbrace{ {\color{red}(1-y_i)}(\log(1-\sigma(W;x_i))}_{\text{case for }y=0} \\
$$

经过推导可以得到

$$
\ell(W) =\sum_{i=1}^m \left( -y_iWx_i + \ln \left( 1 + \mathrm{e}^{Wx_i}\right) \right)
$$

即令每个样本属于其真实类别的概率越大越好，那么~~求不动了 autograd 救救孩子~~

$$
W := W - \alpha\frac{\partial}{\partial W} \mathrm{cost}(W)
$$

## Multinomial Logistic Regression
