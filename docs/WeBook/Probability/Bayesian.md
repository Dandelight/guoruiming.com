# 贝叶斯理论

科学理论通常以可观测的科学变量的概率分布的形式表达。概率分布与未知参数 $\theta$ 有关。在贝叶斯范式中，对于参数 $\theta$，通过我们的现有知识，可以写出其概率分布，称为**先验概率分布（prior distribution）**

$$
p(\theta).
$$

当我们得到新的数据 $\mathbf{x}$ 时，我们希望观察由 $\theta$ 表述的模型与 $\mathbf{x}$ 的契合程度，即似然分布或似然估计（likelihood）。似然估计与概率不同之处在于，概率指未来时间的可能性，而似然指已知结果而估计过去时间（原因）发生的可能性[^mathworld]。该似然估计，即给定 $\theta$ 求 $\mathbf{x}$ 的概率，记作

$$
p(\mathbf{x} | \theta),
$$

其中 $\mid$ 读作 _given_。

这些知识汇总起来，得到**后验分布（posterior distribution）**。贝叶斯定理提出了如何通过先验分布和似然估计得到后验分布方法，即

$$
p(\theta|x) = \frac{p(\theta) p (x|\theta)}{\int_\Theta p(\theta)p(x|\theta)\,\mathrm{d}\theta},
$$

其中 $\int_\Theta p(\theta)p(x|\theta)\,\mathrm{d}\theta$ 被称为**证据率（Evidence）**。

总结来说，后验分布是集中了总体、样本和先验中有关 $\theta$ 的一切信息，又排除了一切与 $\theta$ 无关的信息之后得到的结果。

- 先验分布 Prior：$p(因)$
- 似然分布 Likelihood：$p(果|因)$
- 后验概率 Posterior：$p(因|果)$
- 证据率 Evidence：$p(果)$

换个角度考虑，先验概率是采样前 $p(\theta)$ 人们对 $\theta$ 的认知，后验概率 $p(\theta|x)$ 可以看作采样后对先验概率 $p(\theta)$ 进行的一种调整。

而上式可以表示为

$$
\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}.
$$

因为这个结论实在是太重要，我们再换个形式展示

$$
p(因|果) = \frac{p(果|因) \times p(因)}{p(果)}.
$$

至于贝叶斯估计，挖个坑吧，TODO，啥时候想起来再填。

<https://bayesian.org/what-is-bayesian-analysis/>
[^mathworld]: <https://mathworld.wolfram.com/Likelihood.html>
