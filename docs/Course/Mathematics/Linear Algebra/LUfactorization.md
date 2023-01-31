# LU 分解

## 高斯消元法

考虑方程组

$$
\begin{aligned}
x + y &= 3 \\
3x + 4y &= 2
\end{aligned}
$$

![image-20220111173046874](image-20220111173046874.png)

这两个函数画出来是这样

我们首先描述最简单的高斯消元法。对于线性方程组系统，可以通过三种**初等变换**操作生成等价的系统。这三种变换分别是：

1. 两个方程彼此交换位置
2. 在一个方程上加上另一个方程的倍数
3. 对一个方程乘上一个非零的常数

对于上述方程，如何让计算机求解呢？

先来让我们考虑一个三个方程、三个变量的方程组：

$$
\begin{aligned}
x + 2y -z &= 3 \\
2x + y - 2z &= 3 \\
-3x + u + z &= -6
\end{aligned}
$$

矩阵形式如下：

$$
\left[ {\begin{array}{c|c}
\begin{matrix}
1 & 2 & -1 \\
2 & 1 & -2 \\
-3 & 1 & 1
\end{matrix}&
\begin{matrix}
3 \\ 3 \\ -6
\end{matrix}
\end{array}} \right]
$$

需要两步消去第一列：

$$
\left[ {\begin{array}{c|c}
\begin{matrix}
1 & 2 & -1 \\
0 & -3 & 0 \\
0 & 7 & -2
\end{matrix}&
\begin{matrix}
3 \\ -3 \\ 3
\end{matrix}
\end{array}} \right]
$$

还要一步消去第二列

$$
\left[ {\begin{array}{c|c}
\begin{matrix}
1 & 2 & -1 \\
0 & -3 & 0 \\
0 & 0 & -2
\end{matrix}&
\begin{matrix}
3 \\ -3 \\ -4
\end{matrix}
\end{array}} \right]
$$

返回方程组为

$$
\begin{aligned}
x &= 3 - 2y + z\\
-3y &= -3\\
-2z &= -4
\end{aligned}
$$

之后，从最后一个方程组开始进行**回代**。

话不多说，先上代码。

> Talk is not cheap.

```matlab
% 下三角化
for j = 1 : n-1
	if abs(a(j, j)) < eps; error("zero pivot encountered"); end
	for i = j+1 : n
		mult = a(i, j) / a(j, j);
		for k = j+1 : n
			a(i, k) = a(i, k) - mult * a(j, k);
		end
		b(i) = b(i) - mult*b(j);
	end
end
% 回代
for i = n : -1 : 1
	for j = i+1 : n
		b(i) = b(i) - a(i, j)*x(j);
    end
    x(i) = b(i) / a(i, i);
end
```

## LU 分解

将矩阵$A$分解为一个上三角矩阵$U$和一个下三角矩阵$L$，使得$L\times U = A$。这样，就可以在一定程度上降低计算的复杂度。

一旦知道$L$和$U$，需要求解的问题$Ax=b$便可表示为$LUx=b$，定义辅助向量$c=Ux$，则回代是一个两步的过程：

1. 对于方程$Lc=b$，求解$c$
2. 对于方程$Ux=c$，求解$x$

高斯消元的时间复杂度为$\Theta(n^3)$，而 LU 分解的时间复杂度为$\Theta(\frac23n^3 + 2kn^2)$，其中$k$是解的维度。
