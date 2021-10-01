# UNet 3+

浙大的论文，非要用英文写，想要看懂还得翻译回来，难受。

## 先导知识

### `Up-Sampling`

There are three major ways of upsampling: bilinear interpolation, transposed convolutionand unpooling.

#### Bilinear Interpolation

假设我们想得到未知数$f$在点$P=(x, y)$处的值，已知函数$f$在$Q_{11}=(x_1, y_1)$, $Q_{12}=(x_1, y_2)$, $Q_{21}=(x_2, y_1)$, $Q_{22}=(x_2, y_2)$四个点的值，首先在$x$方向进行差值，得到
$$
f(R_1) \approx \frac{x_2-x}{x_2-x_1}f(Q_{11})+\frac{x-x_1}{x_2-x_1}f(Q_{21})\quad \text{where}\quad R=(x, y_1)
$$

$$
f(R_2) \approx \frac{x_2-x}{x_2-x_1}f(Q_{12})+\frac{x-x_1}{x_2-x_1}f(Q_{22})\quad \text{where}\quad R=(x, y_2)
$$

然后在$y$方向进行线性插值
$$
f(P) \approx \frac{y_2-y}{y_2-y_1}f(R_1) + \frac{y-y_1}{y_2-y_1}f(R_2)
$$

#### Deconvolution

Deconvolution又称Transposed Convolution，

### `Down-Sampling`

