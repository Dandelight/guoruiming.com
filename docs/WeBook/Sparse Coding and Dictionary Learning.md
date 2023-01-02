## Sparse coding

Sparse coding，即稀疏编码，是一种无监督学习方法，其通过寻找一组“超完备”基向量来表达样本数据。相比而言，经典的无监督学习算法主成分分析（Principal Component Analysis, PCA）可以看做寻找一组“完备”基向量——对于同一输入数据，只有一组线性重构，即在训练样本上求得空间的正交基向量 $x_1, \ldots, x_n$ 后，$y$ 可以被**唯一最优表出**，如

$$
y = \sum_{i=1}^n\alpha_i x_i.
$$

但对于稀疏编码，假设我们学得的一组基是 $\{\phi_i\}_{i=1}^k (k > n)$ ，我们的目标同样是线性表出 $y$

$$
y = \sum_{i=1}^k \beta_i \phi_i
$$

超完备的优势在于我们有更多的基向量来捕获数据的内在特征，但是，对同一个 $y$，我们可能有很多种表出方式，$[\beta_1, \ldots, \beta_k]$ 是不唯一的。所以我们增加一条**稀疏性**要求（==这是为什么呢？==）：最优的 $\beta_i$ 只有很少的几个元素大于 $0$，其余元素非常接近 $0$。

Sparser $\beta$s are

- Easy to interpret
- Efficient to store

重构的目标信号还是 $y$，但重构使用的信号少了，所以单个信号的强度会更高，即一旦一个值非零就会很大。

在

Orthogonal Matching Pursuit

https://angms.science/doc/RM/OMP.pdf
