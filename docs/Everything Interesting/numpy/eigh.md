# `numpy.linalg.eigh`

`numpy` provides a function `eigh` to compute the eigenvalues and eigenvectors of a `Hermitian` matrix. The function `eigh` is a wrapper around the `LAPACK` function `dsyevd` which computes the eigenvalues and eigenvectors of a real symmetric matrix. The function `eigh` is a wrapper around the `LAPACK` function `zheevd` which computes the eigenvalues and eigenvectors of a complex Hermitian matrix.

The function `eigh` returns the eigenvalues in ascending order, while the corresponding eigenvectors are in the columns of the same order.

The function `eigh` is a wrapper around the `LAPACK` function `dsyevd` which computes the eigenvalues and eigenvectors of a real symmetric matrix.

The function `eigh` is a wrapper around the `LAPACK` function `zheevd` which computes the eigenvalues and eigenvectors of a complex Hermitian matrix.

Hermitian matrix is a square matrix that is equal to its own conjugate transpose. A complex matrix is Hermitian if it equals its own complex conjugate transpose.

## Why `eigh` is more efficient than `eig`?

The function `eig` computes the eigenvalues and eigenvectors of a general square matrix. The function `eigh` computes the eigenvalues and eigenvectors of a Hermitian matrix. The function `eigh` is more efficient than the function `eig` because the function `eigh` only needs to compute the upper or lower triangular part of the matrix.

## Example

```python
import numpy as np

A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
w, v = np.linalg.eigh(A)
print(w)
print(v)
```

The function `eigh` returns the eigenvalues in ascending order, while the corresponding eigenvectors are in the columns of the same order.

## Syntax

```python
numpy.linalg.eigh(a, UPLO='L')
```

## Parameters

- `a` : (..., M, M) array_like
  - Hermitian input matrix.
- `UPLO` : {'L', 'U'}, optional
  - Whether the calculation is done with the lower or upper triangular part of `a`. Default is to use the lower triangular part.

## Returns

- `w` : (..., M) ndarray
  - The eigenvalues, each repeated according to its multiplicity. They are not necessarily ordered. The eigenvalues are not necessarily ordered. The resulting array will be of complex type if the eigenvalues are complex.
- `v` : (..., M, M) ndarray
  - The normalized (unit “length”) eigenvectors, such that the column `v[:, i]` is the eigenvector corresponding to the eigenvalue `w[i]`. Will return a 2-D array of float or complex numbers, depending on the type of `a`.
