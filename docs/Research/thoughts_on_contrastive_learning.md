# Thoughts on Contrastive Learning

Contrastive Learning（对比学习）自 SimCLR 提出以来受到广泛应用，也产生了多种变体。SimCLR 中，取 $N$ 个样本，对每个样本进行两次增广得到 $2N$ 个样本，通过拉进同一图像的增广、推远不同图像的增广进行训练。其损失函数为

$$
\ell(i, j)=-\log \frac{\exp \left(s_{i, j} / \tau\right)}{\sum_{k=1}^{2 N} \mathbb{1}_{[k \neq i]} \exp \left(s_{i, k} / \tau\right)}
$$

$$
\mathcal{L}=\frac{1}{2 N} \sum_{k=1}^N[\ell(2 k-1,2 k)+\ell(2 k, 2 k-1)]
$$

虽然原论文一句度量学习都没提，但它毕竟没投稿，如果投了我觉得肯定要引用度量学习的相关文章。

但还有一种写法 `InfoNCE` loss:

$$
\mathcal{L}_{\text {InfoNCE }}=-\log \frac{\exp \left(\operatorname{sim}\left(z^a, z^p\right) / \tau\right)}{\exp \left(\operatorname{sim}\left(z^a, z^p\right) / \tau\right)+\sum_{n \in N} \exp \left(\operatorname{sim}\left(z^a, z^n\right) / \tau\right.}
$$

其中 $z^a, z^p, z^n$ 分别为锚样本、正样本和负样本。在 SimCLR 中，$\operatorname{sim}$ 被定义为 cosine similarity。

为啥突然想起总结对比学习是因为 _Learning to Retrieve Prompts for In-Context Learning_ 提出的 Efficient Prompt Retriever (EPR) 方法，它在 retrieve 部分的方法来源于 Facebook Research 的 Dense Passage Retrieval (DPR)。

设数据集为 $\mathcal{D}$。在训练阶段，对于每对训练样本 $(x, y)$，使用一个无监督的检索方法 $\bar{\mathcal{E}} = R_u((x, y), \mathcal{D})$。得到的 $\overline{\mathcal{E}}=\left\{\bar{e}_1, \cdots, \overline{e_L}\right\}$，使用一个打分网络对每个样本进行打分，如

$$
s\left(\bar{e}_l\right)=\operatorname{Prob}_{\hat{g}}\left(y \mid \bar{e}_l, x\right),
$$

DPR 的正样本可以使用 QA 对中的 Answer，或者从一个段落集合（比如维基百科段落）通过 Answer 找到；DPR 的负样本可以通过采样，也可以通过 in-batch negatives（其实就是 SimCLR）；EPR 的取法比较复杂，首先得到打分前 $k$ 高和后 $k$ 低的样本集合 $\mathcal{E}_{\mathrm{pos}}$ 和 $\mathcal{E}_{\mathrm{neg}}$。利用这些样本，训练两个网络输入编码器 $E_X(\cdot)$ 和提示编码器 $E_P(\cdot)$。训练样本表示为

$$
\left\langle x_i, e_i^{+}, e_{i, 1}^{-}, \ldots e_{i, 2 B-1}^{-}\right\rangle
$$

的形式，其中 $e_i^+$ 为正样本，$e_i^-$ 为负样本。正样本从 $\mathcal{E}_\mathrm{pos}^{(i)}$ 中抽取，负样本由

- 一个从 $\mathcal{E}_\mathrm{neg}^{(i)}$ 中抽取的 hard negative
- $B-1$ 个从同 mini-batch 中其它样本的 $\mathcal{E}_\mathrm{pos}$ 中抽取（每个集合抽一个）
- $B-1$ 个从同 mini-batch 中其它样本的 $\mathcal{E}_\mathrm{neg}$ 中抽取（每个集合抽一个）

。设相似性函数

$$
\operatorname{sim}(x, e)=E_X(x)^{\top} E_P(e)
$$

其中 $E_X$ 和 $E_P$ 是两个不同的 `encoder`。计算对比学习 loss

$$
L\left(x_i, e_i^{+}, e_{i, 1}^{-}, \ldots e_{i, 2 B-1}^{-}\right) = -\log \frac{e^{\operatorname{sim}\left(x_i, e_i^{+}\right)}}{e^{\operatorname{sim}\left(x_i, e_i^{+}\right)}+\sum_{j=1}^{2 B-1} e^{\operatorname{sim}\left(x_i, e_{i, j}^{-}\right)}}
$$
