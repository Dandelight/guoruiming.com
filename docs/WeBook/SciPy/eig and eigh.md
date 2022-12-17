在 `scipy.linalg` 中，有两个用于求特征值的函数：`eig` 和 `eigh`。虽然二者长相几乎一样，但

- `eigh` 按特征值**从小到大**的顺序
- `eig` 按特征值**从大到小**的顺序

返回的特征向量都是 `v[:, i]` 表示第 `i` 个特征向量。

```python
class SubspaceLearningMixin():
    """ Mixin Class for Linear Subspace Learning Algorithms.
    Features in the implementation of ``__getitem__`` method.    """    components: np.ndarray
    def __init__(self):
        self.mean_ = 0.0
    @abstractmethod
    def fit(self, X, y):
        pass
    def transform(self, X, n_components=None):
        if n_components: pc = self[n_components]
        else: pc=self.components_
        X = X - self.mean_
        return X @ pc.T
    def __getitem__(self, item):
        """Get principal components."""
        if isinstance(item, slice):
            return self.components_[item]
        else:
            return self.components_[:item]
class MyPCA(SubspaceLearningMixin):
    def __init__(self):
        super().__init__()
    def fit(self, X, y=None, method="svd"):
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_
        if method == 'svd':
            U, S, Vt = scipy.linalg.svd(X, full_matrices=False)
            U, Vt = svd_flip(U, Vt)
            self.variances_ = S # 从大到小
            self.components_ = Vt
        elif method == 'eigen':
            eigval, eigvec = scipy.linalg.eig((X.T @ X) / (n_samples - 1))
            self.components_ = eigvec.real.T
            self.variances_ = eigval.real
        elif method == 'eigen_hermitian':
            eigval, eigvec = scipy.linalg.eigh((X.T @ X) / (n_samples - 1))
            self.components_ = eigvec.T[::-1]
            self.variances_ = eigval[::-1]
```
