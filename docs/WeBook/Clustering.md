## K-Means

```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def normalize(data: np.ndarray):
    mean_ = data.mean(0)
    std_ = data.std(0)
    return (data - mean_) / std_


def get_labels(data, centroids):
    # (n_samples, n_classes)
    dist = euclidean_distances(data, centroids)
    labels = np.argmin(dist, axis=1)
    return labels  # (n_samples,)


def get_centroids(data, labels, n_classes):
    n_features = data.shape[1]
    new_centroids = np.empty((n_classes, n_features))
    for i in range(n_classes):
        cls = data[labels == i]  # (n_c_samples, n_features)
        if cls.size != 0:
            new_centroids[i] = np.mean(cls, axis=0)
        else:
            new_centroids[i] = 0
    return new_centroids


def k_means(data: np.ndarray, n_classes):
    n_samples, n_features = data.shape
    data = normalize(data)
    centroids = np.random.rand(n_classes, n_features)
    n_iteration_ = 0

    while True:
        old_centroids = centroids
        n_iteration_ += 1
        labels = get_labels(data, centroids)
        centroids = get_centroids(data, labels, n_classes)
        if np.allclose(centroids, old_centroids):
            print(f"Converged after {n_iteration_} iterations.")
            break
    return labels
```
