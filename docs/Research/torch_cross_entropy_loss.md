# `CrossEntropyLoss`

Cross Entropy 的公式为

$$
H(p, q) = - \sum_x p(x) \log q(x)
$$

其中 $p$ 为真实标签值，$q$ 为网络预测值。因为 $p$ 经常是 one-hot label，而 $q$ 通常是通过 `softmax` 函数得到的 `logits`，所以上式可化为

Given an input $x$ of size $(N, C)$, where $N$ is the batch size and $C$ is the number of classes, and a target class index $y$ of size $(N,)$, the `CrossEntropyLoss` is calculated as:

$$
\text{{CrossEntropyLoss}}(x, y) = -\frac{1}{N}\sum_{i=1}^{N}\log\left(\frac{{\exp(x_{i,y_i})}}{{\sum_{j=1}^{C} \exp(x_{i,j})}}\right)
$$

Here, $x_{i,y_i}$ represents the logit value of the target class $y_i$ for the $i$-th sample in the batch, and $x_{i,j}$ represents the logit value for class $j$ for the $i$-th sample. The sum in the denominator is taken over all classes. The logarithm and exponential operations ensure that the values are processed in a logarithmic scale and that the probabilities are properly normalized.

The negative sign at the beginning of the formula indicates that the loss is minimized, which means the model is encouraged to assign high probabilities to the correct class and lower probabilities to the incorrect classes.

The `CrossEntropyLoss` function is often used in conjunction with a softmax activation function applied to the logits to obtain class probabilities before computing the loss. This formulation ensures that the loss takes into account the probabilities and the true class index, making it suitable for training models in multi-class classification tasks.

The `torch.nn.CrossEntropyLoss` function in PyTorch is equivalent to the combination of `torch.nn.LogSoftmax` and `torch.nn.NLLLoss`. Let's break down each component individually:

1. `torch.nn.LogSoftmax`: This function applies the logarithm operation to the softmax function output. The softmax function normalizes the input values into a probability distribution, while the logarithm operation transforms the resulting probabilities into logarithmic scale. The formula for `LogSoftmax` can be expressed as:

   $$
   \text{{LogSoftmax}}(x_i) = \log\left(\frac{{\exp(x_i)}}{{\sum\limits_{j=1}^{\text{{num\_classes}}} \exp(x_j)}}\right)
   $$

   Here, $x_i$ represents the input logits for class $i$, and $\exp$ denotes the exponential function.

2. `torch.nn.NLLLoss` (Negative Log Likelihood Loss): This loss function measures the negative log likelihood of the predicted class probability distribution compared to the true distribution. It expects the input to be in logarithmic scale, which is why we use `LogSoftmax` before applying `NLLLoss`. The formula for `NLLLoss` can be expressed as:

   $$
   \text{{NLLLoss}}(x, y) = -x[y]
   $$

   Here, $x$ represents the input logits, and $y$ represents the target class index.

Now, let's derive how the combination of `LogSoftmax` and `NLLLoss` is equivalent to `CrossEntropyLoss`:

Given an input $x$ and its corresponding target class index $y$, the `CrossEntropyLoss` function can be defined as follows:

$$
\text{{CrossEntropyLoss}}(x, y) = \text{{NLLLoss}}(\text{{LogSoftmax}}(x), y)
$$

Substituting the definition of `NLLLoss` and `LogSoftmax`, we have:

$$
\begin{aligned}
\text{{CrossEntropyLoss}}(x, y) &= -\text{{LogSoftmax}}(x)[y] \\
                      &= -\log\left(\frac{{\exp(x[y])}}{{\sum\limits_{j=1}^{\text{{num\_classes}}} \exp(x[j])}}\right) \\
                      &= -\left(x[y] - \log\left(\sum\limits_{j=1}^{\text{{num\_classes}}} \exp(x[j])\right)\right) \\
                      &= \log\left(\sum\limits_{j=1}^{\text{{num\_classes}}} \exp(x[j])\right) - x[y] \\
\end{aligned}
$$

The final expression above is equivalent to the `CrossEntropyLoss` formula. The first term, $\log\left(\sum\limits_{j=1}^{\text{{num\_classes}}} \exp(x[j])\right)$, is the logarithm of the sum of exponential values of all logits, which acts as a normalization factor. The second term, $x[y]$, represents the logit value corresponding to the target class. Thus, minimizing `CrossEntropyLoss` encourages the model to assign a high probability to the correct class and lower probabilities to the incorrect classes.

In summary, by combining `LogSoftmax` and `NLLLoss`, the `CrossEntropyLoss` function provides a convenient and efficient way to compute the loss for multi-class classification problems while ensuring proper handling of logarithmic scaling and probability distributions.

`torch` 的 `CrossEntropyLoss` 还有一个可选的 `label_smoothing` 参数，记为 $\varepsilon$。这是因为 one-hot label 可能让网络过于自信，影响了预测。

$$
q^{\prime}(k)=(1-\epsilon) q(k|x) +\frac{\epsilon}{K}
$$

其中 $K$ 是种类数。
