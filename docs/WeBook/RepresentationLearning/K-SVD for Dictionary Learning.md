来自维基百科的解释：

> In applied mathematics, K-SVD is a dictionary learning algorithm for creating a dictionary for sparse representations, via a singular value decomposition approach. K-SVD is a generalization of the k-means clustering method, and it works by iteratively alternating between sparse coding the input data based on the current dictionary, and updating the atoms in the dictionary to better fit the data. It is structurally related to the expectation maximization (EM) algorithm.[1][2] K-SVD can be found widely in use in applications such as image processing, audio processing, biology, and document analysis.

## 从字典学习的角度看 K-means

在此之前，我们已经了解过 K-means [[Clustering]]。给定输入数据 $\{x_i\}_{i=1}^n$，为了将其聚类为 $c$ 类，我们采用类内均值作为类的代表（原型）——若第 $i$ 个点距离第 $j$ 类的代表最近，则将 $i$ 归为第 $j$ 类，目标函数可写为

$$
\min _{D, X}\left\{\|Y-D X\|_F^2\right\} \quad \text { subject to } \forall i, x_i=e_k \text { for some } k
$$

等价于

$$
\min _{D, X}\left\{\|Y-D X\|_F^2\right\} \quad \text { subject to } \forall i, \|x_i\|_0 = 1
$$

也就是说，K-Means 将 $c$ 个均值向量看作字典 $D$，将 $x_i$ 看作对第 $i$ 个元素的分类。也就是说，K-Means 只使用字典中的一个原子（一列）进行分类。K-SVD 将该限制放松到“可以由 $T_0$ 个原子线性重构”，故其目标函数为：

$$
\min _{D, X}\left\{\|Y-D X\|_F^2\right\} \quad \text { subject to } \forall i, \|x_i\|_0 = T_0
$$

或者变换一下

$$
\underset{D,X}{\min} \sum_i \| x_i \|_0 \quad \text{subject to }\forall i, \| Y-DX \|_F^2 \le \epsilon.
$$

但可惜直接优化该函数是 NP-hard 的，但我们可以使用 approximation pursuit method 对其进行近似。
