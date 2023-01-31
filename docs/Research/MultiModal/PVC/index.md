_Partially View-aligned Clustering_ is a method to address the **Partially View-aligned Problem**.

## Notations

|            Notation            | Definition | Correspondence                                                                   |
| :----------------------------: | :--------: | -------------------------------------------------------------------------------- |
|              $c$               |  Constant  | Number of classes                                                                |
|              $m$               |  Constant  | The number of Views                                                              |
|              $v$               |   Index    | The $v$-th view                                                                  |
|              $V$               |   Matrix   | A View matrix                                                                    |
|      $\mathbf{x}_i^{(v)}$      |   Vector   | The $i$-th datapoint of the $v$-th view                                          |
|       $\mathbf{X}^{(v)}$       |  Dataset   | The set of observations in the $v$-th view                                       |
| $\{\mathbf{X}^{(v)}\}_{v=1}^m$ |  Dataset   | The Given dataset                                                                |
|    $\{\mathbf{A}\}_{v=1}^m$    |  Dataset   | Aligned part of $\{X\}$                                                          |
|    $\{\mathbf{U}\}_{v=1}^m$    |  Dataset   | Unaligned part of $\{X\}$                                                        |
|          $\mathbf{P}$          |   Matrix   | The permutation matrix that has exactly one entry of $1$ in each row and column. |

Partially view-aligned problem (PVP) is the problem that multiple views may be unaligned. A typical example of PVP is the street surveillance problem. In this setting, we have multiple cameras from different directions, which correspond to multiple views. An object of interest may appear in view $v_1$ at time $t_1$ and position $p_1$, but in view $v_2$ it may appear at $t_2 \neq t_1$ and $p_2 \neq p_1$. Solving this problem is daunting for the following three reasons:

1. There's no label.
2. The Munkres algorithm (or Hungarian algorithm) for bipartite matching is not differentiable.
3. It is expected to jointly learn common representation and perform alignment.

The authors expect to establish the correspondence of unaligned data with the help of the ground-truth aligned data.

## Problem Formulation

Given a dataset $\{\mathbf{X}\}_{v=1}^m$, MVC aims to separate all datapoints into one of the $c$ clusters. There's only a part of $X$ is with correspondence in advance. Of course we can simply use the Munkres on the aligned data, it has two drawbacks:

1. We cannot utilize the partially-aligned data.
2. The Hungarian algorithm is non-differentiable, so it cannot be optimized with the neural network.

So, it is expected to develop ==a differentiable alignment algorithm== so that the representation learning and data alignment can be jointly optimized. This will utilize the correspondence at the set level instead at point level. We seek to find the correspondence that

$$
\mathbf{X}^{(1)} \sim \mathbf{PX}^{(2)},
$$

where $\sim$ is a relational operator which denote that $X^{(2)}$ and $X^{(1)}$ are **aligned** by $\mathbf{P}$.

The proposed Partially View-aligned Clustering (PVC) model have two modules -- one to learn the cross-view representation by utilizing the predicted correspondence, and the other is to perform the data alignment in the latent space learned by the neural network.

### Cross-view Representation Learning

First, we use the aligned data $\{\mathbf{A^{(v)}}\}_{v=1}^m$ to train a neural network to ?

The objective function is

$$
\mathcal{L}=\mathcal{L}_1+\lambda \sum_{v \neq u} \mathcal{L}_2^{(u v)},
\label{eq:loss}
$$

where the $\mathcal{L}_1$ encourages the learning of common representations across views by

$$
\mathcal{L}_1=\sum_{v=1}^m \underbrace{\left\|\mathbf{A}^{(v)}-g^{(v)}\left(f^{(v)}\left(\mathbf{A}^{(v)}\right)\right)\right\|_2^2}_{\text {within-view reconstruction }}+\sum_{v \neq u} \underbrace{\left\|f^{(v)}\left(\mathbf{A}^{(v)}\right)-\mathbf{P}_{u v} f^{(u)}\left(\hat{\mathbf{A}}^{(u)}\right)\right\|_2^2}_{\text {cross-view consistency }},
$$

and $\mathcal{L}_2$ minimizes the distance between the model prediction and the ground-truth by

$$
\mathcal{L}_2^{(u v)}=\left\|\mathbf{P}^{(u v)}-\mathbf{P}_{g t}^{(u v)}\right\|_2^2,
$$

### Differentiable alignment

The optimization of $\mathbf{P}$ becomes the integer linear programming problem which aims at achieving the best matching of bi-graph. Formally,

$$
\begin{aligned}
\underset{\mathbf{P}}{\arg \min } & \operatorname{Tr}\left(\mathbf{D P}^{\top}\right) \\
\text { s.t. } & P_{i j} \in\{0,1\}, \forall(i, j) \\
& \mathbf{P} \mathbf{1}=\mathbf{1} \\
& \mathbf{P}^{\top} \mathbf{1}=\mathbf{1},
\end{aligned}
$$

Where $\operatorname{Tr}()$ denotes the matrix trace and $\mathbf{D}\in \mathbb{R}^{n\times n}$ is the distance matrix in which $D_{ij}$ denotes the distance of assigning $i$ to $j$. We define $\mathbf{D}$ as the pairwise distance between $\mathbf{A}_i^{(1)}$ and $\hat{\mathbf{A}}_J^2$, i.e.,

$$
D_{i j}=\left\|f^{(1)}\left(\mathbf{A}_i^{(1)}\right)-f^{(2)}\left(\hat{\mathbf{A}}_j^{(2)}\right)\right\|_2^2
\label{eq:distance}
$$

The optimization problem is NP-complete and non-differentiable, so it is impossible to optimize together with a neural network. So we relax the constraint of binary matrix to real-valued matrix, and the objective function could be written as

$$
\begin{aligned}
\arg \min & \operatorname{Tr}\left(\mathbf{D P}^{\top}\right) \\
\text { s.t. } & P_{i j} \geq 0, \forall(i, j) \\
& \mathbf{P} \mathbf{1}=\mathbf{1} \\
& \mathbf{P}^{\top} \mathbf{1}=\mathbf{1},
\end{aligned}
$$

Now the loss can be solved by gradient descent, and the optimized set is the intersection of the three closed convex sets from the constraints above.

To solve this problem, we proceed by iteratively projecting $\mathbf{P}$ onto three spaces.

$$
\begin{aligned}
& \Psi_1(\mathbf{P})=\operatorname{ReLU}(\mathbf{P}) \\
& \Psi_2(\mathbf{P})=\mathbf{P}-\frac{1}{n}(\mathbf{P} \mathbf{1}-\mathbf{1}) \mathbf{1}^{\top} \\
& \Psi_3(\mathbf{P})=\mathbf{P}-\frac{1}{n} \mathbf{1}\left(\mathbf{1}^{\top} \mathbf{P}-\mathbf{1}^{\top}\right)
\end{aligned}
\label{eq:dykstra}
$$

The resulting formula is differentiable and pluggable into any neural network.

## Implementation

We assume we only have two modalities for simplicity in explanation. We can extend to more modalities if necessary. Iteratively do the following

1. **Representation Learning**: Pass _unaligned_ data through network, i.e., $\{f^{(1)}, f^{(2)}, g^{(1)}, g^{(2)}\}$, yielding the hidden representations $\mathbf{Z}^{(1)} = f^{(1)} \left( \mathbf{A}^{(1)}\right)$ and $\mathbf{Z}^{(2)} = f^{(2)} \left( \mathbf{A}^{(2)}\right)$. Then we calculate the distance matrix $\mathbf{D}$ in the latent space by $(\ref{eq:distance})$.
2. **View Aligning**: Use $\mathbf{D}$ to calculate $\mathbf{P}$ as in $(\ref{eq:dykstra})$.
3. **Back-propagation**: Compute the loss as in $(\ref{eq:loss})$ and run back-propagation to update network parameters.

Code as provided by the author in [XLearning-SCU/2020-NIPS-PVC](https://github.com/XLearning-SCU/2020-NIPS-PVC):

```python
import math
import torch
import torch.nn as nn

class PVC(nn.Module):
    def __init__(self, arch_list):
        super(PVC, self).__init__()
        self.view_size = len(arch_list)
        self.enc_list = nn.ModuleList()
        self.dec_list = nn.ModuleList()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        # network
        for view in range(self.view_size):
            enc, dec = self.single_ae(arch_list[view])
            self.enc_list.append(enc)
            self.dec_list.append(dec)
        self.dim = arch_list[0][0]

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.dim)

    def single_ae(self, arch):
        # encoder
        enc = nn.ModuleList()
        for i in range(len(arch)):
            if i < len(arch)-1:
                enc.append(nn.Linear(arch[i], arch[i+1]))
            else:
                break

        # decoder
        arch.reverse()
        dec = nn.ModuleList()
        for i in range(len(arch)):
            if i < len(arch)-1:
                dec.append(nn.Linear(arch[i], arch[i+1]))
            else:
                break

        return enc, dec

    def forward(self, inputs_list):
        encoded_list = []
        decoded_list = []

        for view in range(self.view_size):
            # encoded
            encoded = inputs_list[view]
            for i, layer in enumerate(self.enc_list[view]):
                if i < len(self.enc_list[view]) - 1:
                    encoded = self.relu(layer(encoded))
                else: # the last layer
                    encoded = layer(encoded)
            encoded_list.append(encoded)

            # decoded
            decoded = encoded
            for i, layer in enumerate(self.dec_list[view]):
                if i < len(self.dec_list[view]) - 1:
                    decoded = self.relu(layer(decoded))
                else: # the last layer
                    decoded = layer(decoded)
            decoded_list.append(decoded)

        return encoded_list, decoded_list

```
