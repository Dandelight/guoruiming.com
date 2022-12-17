# Properties of the Covariance Matrix

source: https://www.robots.ox.ac.uk/~davidc/pubs/tt2015_dac1.pdf

The covariance matrix of a random vector $\mathbf{X} \in \mathbf{R}^{n}$ with mean vector $\mathbf{m}_{x}$ is defined via:

$$
\mathbf{C}_{x}=E\left[(\mathbf{X}-\mathbf{m})(\mathbf{X}-\mathbf{m})^{T}\right] .
$$

The $(i, j)^{\text {th }}$ element of this covariance matrix $\mathbf{C}_{x}$ is given by

$$
C_{i j}=E\left[\left(X_{i}-m_{i}\right)\left(X_{j}-m_{j}\right)\right]=\sigma_{i j} .
$$

The diagonal entries of this covariance matrix $\mathbf{C}_{x}$ are the variances of the components of the random vector $\mathbf{X}$, i.e.,

$$
C_{i i}=E\left[\left(X_{i}-m_{i}\right)^{2}\right]=\sigma_{i}^{2} .
$$

Since the diagonal entries are all positive the trace of this covariance matrix is positive, i.e.,

$$
\operatorname{Trace}\left(\mathbf{C}_{x}\right)=\sum_{i=1}^{n} C_{i i}>0 .
$$

This covariance matrix $\mathbf{C}_{x}$ is symmetric, i.e., $\mathbf{C}_{x}=\mathbf{C}_{x}^{T}$ because :

$$
C_{i j}=\sigma_{i j}=\sigma_{j i}=C_{j i} .
$$

The covariance matrix $\mathbf{C}_{x}$ is positive semidefinite, i.e., for $\mathbf{a} \in \mathbf{R}^{n}$ :

$$
\begin{aligned}
E\left\{\left[(\mathbf{X}-\mathbf{m})^{T} \mathbf{a}\right]^{2}\right\} & =E\left\{\left[(\mathbf{X}-\mathbf{m})^{T} \mathbf{a}\right]^{T}\left[(\mathbf{X}-\mathbf{m})^{T} \mathbf{a}\right]\right\} \geq 0 \\
E\left[\mathbf{a}^{T}(\mathbf{X}-\mathbf{m})(\mathbf{X}-\mathbf{m})^{T} \mathbf{a}\right] & \geq 0, \quad \mathbf{a} \in \mathbf{R}^{n} \\
\mathbf{a}^{T} \mathbf{C}_{x} \mathbf{a} & \geq 0, \quad \mathbf{a} \in \mathbf{R}^{n} .
\end{aligned}
$$

Since the covariance matrix $\mathbf{C}_{x}$ is symmetric, i.e., self-adjoint with the usual inner product its eigenvalues are all real and positive and the eigenvectors that belong to distinct eigenvalues are orthogonal, i.e.,

$$
\mathbf{C}_{x}=\mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{T}=\sum_{i=1}^{n} \lambda_{i} \vec{v}_{i} \vec{v}_{i}^{T} .
$$

As a consequence, the determinant of the covariance matrix is positive, i.e.,

$$
\operatorname{Det}\left(\mathbf{C}_{X}\right)=\prod_{i=1}^{n} \lambda_{i} \geq 0 .
$$

The eigenvectors of the covariance matrix transform the random vector into statistically uncorrelated random variables, i.e., into a random vector with a diagonal covariance matrix. The Rayleigh coefficient of the covariance matrix is bounded above and below by the maximum and minimum eigenvalue :

$$
\lambda_{\min } \leq \frac{\mathbf{a}^{T} \mathbf{C}_{x} \mathbf{a}}{\mathbf{a}^{T} \mathbf{a}}, \quad \mathbf{a} \in \mathbf{R} \leq \lambda_{\max } .
$$
