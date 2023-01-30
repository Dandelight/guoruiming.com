KL Divergence 在稀疏编码、变分自动编码器（Variational Autoencoder, VAE）中作为损失函数以保证结果的稀疏性。

以下推导来自 <https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/>。我们只关心 $p$ 作为变量的情况。首先我们将 KL 散度改写为信息熵的形式：

$$
\begin{align*}
\textbf{KL}(p || q_\theta)
&= \sum_d p(d) \log \left( \frac{p(d)}{q(d)} \right) \\
&= \sum_d p(d) \left( \log p(d) - \log q(d) \right) \\
&= \underbrace{\sum_d q(d) \log q(d)}_{-\text{entropy}} - \underbrace{\sum_d q(d) \log p(d)}_{\text{cross-entropy}} \\
\end{align*}
$$

可以发现减号前一部分不包含 $p$，所以我们只需要多后一部分（交叉熵）求导。我们将其也改写一下

$$
\begin{align*}
\sum_d p(d) \log q(d)
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \log \left( \bar{q}(d)/Z_q \right) \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \left( \log \bar{q}(d) - \log Z_q \right) \\
&= \left(\frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d)\right) - \left(\frac{1}{Z_p} \sum_d \bar{p}(d) \log Z_q\right) \\
&= \left(\frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d)\right) - \left( \log Z_q \right) \left( \frac{1}{Z_p} \sum_d \bar{p}(d)\right) \\
&= \left(\frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d)\right) - \log Z_q
\end{align*}
$$

当 $q$ 是指数函数时，对其求梯度比较直观

$$
\begin{align*}
\nabla \left[ \frac{1}{Z_p} \sum_d \bar{p}(d) \log \bar{q}(d) - \log Z_q \right]
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \nabla \left[ \log \bar{q}(d) \right] - \nabla \log Z_q \\
&= \frac{1}{Z_p} \sum_d \bar{p}(d) \phi_q(d) - \mathbb{E}_q \left[ \phi_q \right] \\
&= \mathbb{E}_p \left[ \phi_q \right] - \mathbb{E}_q \left[ \phi_q \right]
\end{align*}
$$

回到

$$
q = \frac{{\mathrm{e}}^{-{\left(i-j\right)}^2}}{{\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}}
$$

$$
\frac{\partial q}{\partial i} = \frac{{\mathrm{e}}^{-{\left(i-j\right)}^2}\,\left({\mathrm{e}}^{-{\left(i-j\right)}^2}\,\left(2\,i-2\,j\right)+{\mathrm{e}}^{-{\left(i-k\right)}^2}\,\left(2\,i-2\,k\right)\right)}{{\left({\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}\right)}^2}-\frac{{\mathrm{e}}^{-{\left(i-j\right)}^2}\,\left(2\,i-2\,j\right)}{{\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}}
$$

$$
\frac{\partial q}{\partial j} = \frac{{\mathrm{e}}^{-{\left(i-j\right)}^2}\,\left(2\,i-2\,j\right)}{{\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}}-\frac{{\mathrm{e}}^{-2\,{\left(i-j\right)}^2}\,\left(2\,i-2\,j\right)}{{\left({\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}\right)}^2}
$$

$$
\frac{\partial q}{\partial k} = -\frac{{\mathrm{e}}^{-{\left(i-j\right)}^2}\,{\mathrm{e}}^{-{\left(i-k\right)}^2}\,\left(2\,i-2\,k\right)}{{\left({\mathrm{e}}^{-{\left(i-j\right)}^2}+{\mathrm{e}}^{-{\left(i-k\right)}^2}\right)}^2}
$$
