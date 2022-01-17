$\operatorname{IoU}$即交并比，又名 Jaccard Index，在语义分割中一直广为使用。

$$
\operatorname{IoU} = \frac{target \cup prediction}{target \cap prediction}
$$

$\mathrm{IoU}$一般基于类计算，也有基于图片计算[^csdn87901365]，一定要看清数据集的评价标准。

对于$\mathrm{mIoU}$，就是基于类计算的$\mathrm{IoU}$对每一类的值进行累加后取平均。

具体到计算上，设 Ground Truth 为$A_{GT} \in \mathbb{R}^{2}$，预测结果$A_p \in \mathbb{R}^2$，则对于$A_{GT}$中的每一个像素的真实值值$A_{GT}[x, y]$（代表一个类），与预测值作混淆矩阵（confusion matrix）。我们考虑一个简单的情况：

$$
A_{GT} = \begin{bmatrix}
 1 & 1 \\ 2 & 2
\end{bmatrix},\quad
A_p = \begin{bmatrix}
2 & 1 \\ 2 & 2
\end{bmatrix}
$$

则其混淆矩阵$M$（横向为 GT，纵向为 P）

$$
M = \begin{bmatrix}
	1 & 0 \\ 1 & 2
\end{bmatrix}
$$

> 或许画个正字会好一些？

那么可知，$\mathrm{IoU}_1 = \frac{1}{2} = 0.5$，$\mathrm{IoU}_2=\frac{2}{3} \approx 0.667$，$\mathrm{mIoU}$为二者的平均。

从向量的角度理解，$\mathrm{IoU}_i$即为主对角线上第$i$个元素除以（第$i$行所有值的和），即$\mathrm{IoU}_i = \frac{M[i, i]}{M[i, :]}$。进而，$\mathrm{mIoU} = \frac{\sum_{i=1}^n{\mathrm{IoU_i}}}{n}$。

[^csdn87901365]: https://blog.csdn.net/lingzhou33/article/details/87901365
