The answer is that, for any square matrix $A$, the trace is the sum of eigenvalues, that is,

$$
\operatorname{tr}(A) = \sum \lambda_i,
$$

and determinant is the product of eigenvalues, i.e.,

$$
\det(A) = \prod \lambda_i.
$$

So maximizing the trace or determinant corresponding the eigenvalues of $A$.

<u>I think that using the Forbenius norm as the objective function is **incorrect**</u>. To make sure that we understand thoroughly, we go through the basics of covariance matrix. Consider the data matrix $\mathbf{X}$, with each column as the observation of the same data, and each row as a measurement type. The covariance matrix is defined as

$$
\mathbf{C}_\mathbf{X} = \frac{1}{n} \mathbf{XX}^T.
$$

The element $c_{ij}$ of $\mathbf{C}$ measures the covariance of the $i$-th and the $j$-th measurement types, and when $i=j$ we call it variance of measurement type $i$. Principal component analysis (PCA) aims at **maximize the variance **maximizes** $\mathbf{C_X}$'s diagonal value, that is, the variance, while **minimizes\*\* the off-diagonal value. Since $\mathbf{C_X}$ is diagonalizable, Its objective function is formulated as

$$
\begin{gathered}
\max_\mathbf{W} \operatorname{tr}(\mathbf{W}^T\mathbf{C_X} \mathbf{W}) \\
\text{s.t. } \mathbf{W}^T \mathbf{W} = I.
\end{gathered}
$$

**or**

$$
\begin{gathered}
\max_\mathbf{W} \det(\mathbf{W}^T\mathbf{C_X} \mathbf{W}) \\
\text{s.t. } \mathbf{W}^T \mathbf{W} = I.
\end{gathered}
$$

Back to the Fisher discriminant analysis is to **maximize the between-class variance** and **minimize the within-class variance**, thus its objective function is formulated as

$$
\begin{gathered}
\max_\mathbf{W} \frac{\operatorname{tr}(\mathbf{W}^T\mathbf{S}_b \mathbf{W})}{\operatorname{tr}(\mathbf{W}^T\mathbf{S}_w \mathbf{W})} \\
\text{s.t. } \mathbf{W}^T \mathbf{W} = I.
\end{gathered}
$$

**or**

$$
\begin{gathered}
\max_\mathbf{W} \frac{\det(\mathbf{W}^T\mathbf{S}_b \mathbf{W})}{\det(\mathbf{W}^T\mathbf{S}_w \mathbf{W})} \\
\text{s.t. } \mathbf{W}^T \mathbf{W} = I.
\end{gathered}
$$

Since a Forbenius norm is

$$
\|\mathrm{A}\|_F \equiv \sqrt{\sum_{i=1}^m \sum_{j=1}^n\left|a_{i j}\right|^2},
$$

it takes all elements in $A$ into account. It doesn't make sense.
