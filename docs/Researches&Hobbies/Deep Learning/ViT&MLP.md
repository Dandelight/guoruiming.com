最近做了一些关于 Vision Transformer 和 MLP 的调研，记录如下

# An Image Is Worth A Thousand Words

书接上回，Transformer 横空出世，一举成为 NLP 领域的最佳模型，不仅在各项任务上有了突破性进展， 还催生了 BERT、GPT 等大模型。就像有人想把 CNN 搬到文本上一样，有很多工作尝试将自注意力机制与卷积融合起来。~~但为什么这么具有开创性意义又直截了当的工作早没有人做呢？因为他们没有 Google 这样庞大的算力。~~作者实验性地将 Transformer 结构作尽可能小的改动，将图像分成$16\times 16$的小块，经过可学习的`Linear Projection`（其实就是`nn.Linear`），就成为了 Multihead Self-Attention 的输入 Token。使用 JFT-300M 数据集在 TPU 上训练 2.5k 核日之后在 ImageNet 上达到了$88.55\pm0.04 \%$的结果，~~成功地证明了训练 ViT 需要消耗极大的算力。~~

## 著名代码库

[huggingface/transformers](https://github.com/huggingface/transformers)新兴 AI 企业:hugs:huggingface 的 transformer 库，不仅收录了用于 NLP 的 transformer，还收录了多种 ViT 甚至 ConvNeXt，[详细列表](https://github.com/huggingface/transformers#model-architectures)。`\cite{wolf-etal-2020-transformers}`

[liuruiyang98/Jittor-MLP](https://github.com/liuruiyang98/Jittor-MLP)多种 MLP 算法的`Pytorch`和/或`Jittor`实现，内附一篇综述`\cite{liu2021we}`。

```bibtex
@article{DosovitskiyAlexey2020AIiW,
abstract = {While the Transformer architecture has become the de-facto standard for
natural language processing tasks, its applications to computer vision remain
limited. In vision, attention is either applied in conjunction with
convolutional networks, or used to replace certain components of convolutional
networks while keeping their overall structure in place. We show that this
reliance on CNNs is not necessary and a pure transformer applied directly to
sequences of image patches can perform very well on image classification tasks.
When pre-trained on large amounts of data and transferred to multiple mid-sized
or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision
Transformer (ViT) attains excellent results compared to state-of-the-art
convolutional networks while requiring substantially fewer computational
resources to train.},
year = {2020},
title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
copyright = {http://arxiv.org/licenses/nonexclusive-distrib/1.0},
language = {eng},
author = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
}
```

```bibtex
@article{liu2021we,
  title={Are we ready for a new paradigm shift? A Survey on Visual Deep MLP},
  author={Liu, Ruiyang and Li, Yinghui and Liang, Dun and Tao, Linmi and Hu, Shimin and Zheng, Hai-Tao},
  journal={arXiv preprint arXiv:2111.04060},
  year={2021}
}
```

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
