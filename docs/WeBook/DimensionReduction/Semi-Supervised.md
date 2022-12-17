- Induction is reasoning from observed training cases to general rules, which are then applied to the test cases.
- Transduction is reasoning from observed, specific (training) cases to specific (test) cases. [^sanjay]

![](https://pica.zhimg.com/80/v2-7f17bddd039854a3440554b66501d2c3_720w.webp?source=1940ef5c)

现在有这个问题，已知 ABC 的类别，求问号的类别，

inductive learning 就是只根据现有的 ABC，用比如 kNN 距离算法来预测，在来一个新的数据的时候，还是只根据 5 个 ABC 来预测。

transductive learning 直接以某种算法观察出数据的分布，这里呈现三个 cluster，就根据 cluster 判定，不会建立一个预测的模型，如果一个新的数据加进来 就必须重新算一遍整个算法，新加的数据也会导致旧的已预测问号的结果改变。[^techonly]

ref:
[^sanjay]: https://www.quora.com/What-is-the-difference-between-inductive-and-transductive-learning
[^techonly]: 如何理解 inductive learning 与 transductive learning? - TechOnly 的回答 - 知乎 https://www.zhihu.com/question/68275921/answer/480709225

https://zhuanlan.zhihu.com/p/455808338
