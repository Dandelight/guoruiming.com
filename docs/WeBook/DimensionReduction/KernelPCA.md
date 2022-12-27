注意：本文内容全文由 GitHub Copilot 合成，不保证其中信息真实准确。

# Kernel PCA

Kernel PCA is a dimensionality reduction technique that uses the kernel trick to project the data into a higher dimensional space where it is linearly separable. This is useful when the data is not linearly separable in the original space, but is linearly separable in the higher dimensional space.

## Classical PCA

The objective function of classical PCA is

$$
\begin{align}
\min_{\mathbf{W}} \frac{1}{n} \sum_{i=1}^n \left\| \mathbf{x}_i - \mathbf{W} \mathbf{W}^T \mathbf{x}_i \right\|^2
\end{align}
$$

where $\mathbf{W}$ is the projection matrix. The solution to this problem is

$$
\begin{align}
\mathbf{W} = \mathbf{U}_k
\end{align}
$$

where $\mathbf{U}_k$ is the first $k$ eigenvectors of the covariance matrix $\mathbf{C} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$.

## Kernel PCA

The objective function of kernel PCA is

$$
\begin{align}
\min_{\mathbf{W}} \frac{1}{n} \sum_{i=1}^n \left\| \mathbf{z}_i - \mathbf{W} \mathbf{W}^T \mathbf{z}_i \right\|^2
\end{align}
$$

where $\mathbf{W}$ is the projection matrix and $\mathbf{z}_i = \phi(\mathbf{x}_i)$ is the feature vector of the data $\mathbf{x}_i$ in the higher dimensional space. The solution to this problem is

In this objective we use the so-called **kernel trick**. The kernel trick is a method to compute the inner product of two vectors in a higher dimensional space without actually computing the vectors in the higher dimensional space.
In fact, we introduce a mapping $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^D$ and compute the inner product in the higher dimensional space.

### Covariance Matrix with Kernel

The covariance matrix in the higher dimensional space is

$$
\begin{align}
\mathbf{C} = \frac{1}{n} \sum_{i=1}^n \mathbf{z}_i \mathbf{z}_i^T
\end{align}
$$

where $\mathbf{z}_i = \phi(\mathbf{x}_i)$.

Any eigenvector can be represented as a weight sum of the data points $\phi(\mathbf{x}_i)$

Assume the $N$ $d$-dimensional data points are $\mathbf{x}_1, \ldots, \mathbf{x}_N$. Assume we project it into a $D$-dimensional space using the mapping $\phi$. Then the inner product of $\mathbf{z}_i$ and $\mathbf{z}_j$ is

$$
\begin{align}
\mathbf{z}_i^T \mathbf{z}_j = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j) = \sum_{d=1}^D \phi_d(\mathbf{x}_i) \phi_d(\mathbf{x}_j)
\end{align}
$$

where $\phi_d(\mathbf{x})$ is the $d$-th component of $\phi(\mathbf{x})$.

The kernel trick is to replace the inner product with the kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$.

The kernel trick is defined as

$$
\begin{align}
\langle \mathbf{z}_i, \mathbf{z}_j \rangle = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j) = K(\mathbf{x}_i, \mathbf{x}_j),
\end{align}
$$

where $\phi_l$ is the $l$th basis function of the feature vector $\mathbf{z}_i$. The kernel trick is used to compute the inner product of two vectors in the higher dimensional space without actually computing the vectors in the higher dimensional space.

## Kernel Functions

The kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ is a function that maps two data points $\mathbf{x}_i$ and $\mathbf{x}_j$ to a scalar. The kernel function is symmetric, i.e. $K(\mathbf{x}_i, \mathbf{x}_j) = K(\mathbf{x}_j, \mathbf{x}_i)$.

The kernel function is positive semi-definite, i.e. $\forall \mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^d, K(\mathbf{x}_i, \mathbf{x}_j) \geq 0$.

The kernel function is homogeneous, i.e. $\forall \mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^d, \forall \alpha \in \mathbb{R}, K(\alpha \mathbf{x}_i, \mathbf{x}_j) = \alpha K(\mathbf{x}_i, \mathbf{x}_j)$.

## Common Kernel Functions

### Linear Kernel

The linear kernel is

$$
\begin{align}
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
\end{align}
$$

### Polynomial Kernel

The polynomial kernel is

$$
\begin{align}
K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d
\end{align}
$$

where $c$ is the bias and $d$ is the degree of the polynomial.

### Gaussian Kernel

The Gaussian kernel is

$$
\begin{align}
K(\mathbf{x}_i, \mathbf{x}_j) = \exp \left( -\frac{\| \mathbf{x}_i - \mathbf{x}_j \|^2}{2 \sigma^2} \right)
\end{align}
$$

where $\sigma$ is the standard deviation.

## Code

```python
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA, KernelPCA


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ----------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    -------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return X_pc


def main():
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)
    plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.show()
```

简单理解就是，核方法就是在距离矩阵上做文章，比如普通 LDE 的求解的等式为

$$
X (D' - W') X^T \mathbf{v} = \lambda X (D - W) X^T \mathbf{v}
$$

Kernel LDE 求解的等式为

$$
K (D' - W') K \mathbf{\alpha} = \lambda K (D-W) K \mathbf{\alpha}
$$

其中 $K$ 为 kernel matrix，$K_{ij}=\mathrm{k}(\mathbf{x}_i, \mathbf{x}_j)$，因核函数 $\mathrm{k}(\cdot)$ 的对称性，$K$ 也为对称矩阵。
