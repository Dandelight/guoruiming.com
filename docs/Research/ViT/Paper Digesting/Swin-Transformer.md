# Swin-Transformer

> 各位小伙伴们，大家下午好。下周四（2021 年 11 月 26 日）我将进行研读论文的题目是《Res2Net: A New Multi-Scale Backbone Architecture》和《Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows》，论文 PDF 已上传至群文件，请感兴趣的小伙伴自行查阅。

[http://scuvis.org/2021autumnpaper-1202/](http://scuvis.org/2021autumnpaper-1202/)

```tex
@InProceedings{Liu_2021_ICCV,
    author    = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    title     = {Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10012-10022}
}
```

Transformer 是在自然语言处理领域大获成功的 RNN 模型，并且在计算机视觉任务中也表现出优异的成绩。但是，相较于文本，视觉实体的尺度变化更大、分辨率更高等特点，都给 Transformer 的在计算机视觉上的应用带来了挑战。为了解决此问题，作者提出了一种层级化的 Transformer 网络，并使用 Shifted Windows 方法进行 representation 的计算。Shifted Windows（简称 Swin）方法将自注意力的计算限制在不重叠的局部窗口中，通过 shifted window 方法实现跨窗口信息连接。网络的层级化结构使得网络能够适应多种尺度的视觉目标，并且 Swin 的计算只有线性时间复杂度。以 Swin Transformer 作为骨干网络的模型在多个计算机视觉任务中取得了 SOTA 的成绩。

## 背景引入

研究动机

> to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision.

相较于 CV 领域各种维度、各种残差连接、各种计算层（研究结构、研究单层（depth wise convolution [70] and deformable convolution [18, 84].））的卷积神经网络百花齐放，NLP 领域的 Transformer 以其强大的建模数据中长依赖的能力一统江湖。因为 Transformer 在 NLP 里太成功了，所以很多研究者在尝试把 Transformer 应用在视觉领域。在 Swin-Transformer 之前，将 Transformer 迁移到计算机视觉领域取得了一定成功（如 ViT），但在 CV 任务上完全发挥 Transformer 的能力有很大挑战，主要有两点：

1. 在 NLP 中，文字的 token 即为实体的最小元素；而在 CV 中，一个实体可以由几个到几万个像素表达。
2. 在基于 Transformer 的模型中，token 的大小处于固定不变的尺度，但在 CV 中不是如此，甚至在场景分割等任务中需要预测每一个像素点所属类别，Transformer 计算量过大而不可行。

### Transformer

要理解文中的 Swin Transformer，需要先了解 Transformer 和 Vision Transformer。而要了解 Transformer，需要先了解 Attention 和 Self-attention。

#### Attention

Attention机制是一种在深度学习领域中，特别是在自然语言处理（NLP）任务中广泛应用的模型组件。它源于人类注意力机制的概念，即我们在处理信息时会聚焦于最关键的部分，而忽略不太相关的信息。在机器学习模型中引入Attention机制，使得模型能够更加灵活和高效地处理输入数据，尤其是长序列数据，如文本或者时间序列。

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
是 Transformer 中使用到的 Attention，这里涉及到三个主要的向量：查询（Query, $Q$）、键（Key, $K$）和值（Value, $V$）。这个公式描述了如何通过查询向量来加权值向量，从而获得上下文相关的输出。具体解释如下：

- **查询（Query, $Q$）**：代表当前要处理的信息或者说是“问题”，模型试图基于这个查询从其他信息中找出最相关部分。
- **键（Key, $K$）**：是来自输入序列的表示，每个键与输入序列中的一个位置或元素相对应，用于衡量该位置信息与查询的相关性。
- **值（Value, $V$）**：同样对应于输入序列中的每个位置，包含了实际需要关注的信息内容。一旦确定了哪些键（即输入序列中的哪些部分）与查询最为相关，就根据这些键对应的值来计算输出。

公式中的$\frac{QK^T}{\sqrt{d_k}}$部分是用来计算查询向量$Q$与所有键向量$K$之间的相似度得分矩阵，其中$d_k$是键向量的维度，除以$\sqrt{d_k}$是为了缩放得分，避免在维度较大时softmax函数的梯度消失问题。接着，对这个得分矩阵应用softmax函数，得到每个键相对于查询的权重分布。最后，将这个权重分布应用于值向量$V$，通过加权求和的方式生成最终的输出，这个输出综合了输入序列中各部分的信息，但更侧重于与查询最相关的部分。

#### Multi-head Self-Attention

![ModalNet-20](media/Swin-Transformer/ModalNet-20.png)



#### Transformer Architecture

![ModalNet-21](media/Swin-Transformer/ModalNet-21.png)

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. _Advances in Neural Information Processing Systems_, _2017-December_.

#### Self-attention based

卷积神经网络的本质是特征提取，但卷积并不是最好的特征提取器（其实没有最好），一定程度上因为卷积核大小固定，难以提取图像的多尺度信息

在 2019 年微软亚洲研究研究院的 LR-Net，在 26 层的 LR-Net 超过 26 层的 ResNet 3 个百分点

![fig2_local_relation_layer](media/Swin-Transformer/fig2_local_relation_layer-16378441549171.png)

![fig5_appearance](media/Swin-Transformer/fig5_appearance.png)

> H. Hu, Z. Zhang, Z. Xie and S. Lin, "Local Relation Networks for Image Recognition," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3463-3472, doi: 10.1109/ICCV.2019.00356.

滑动窗口访问内存的随机性导致计算性能并不好

#### Transformers to complement CNNs

### ViT & ResNe(X)t

![model_scheme_00](media/Swin-Transformer/model_scheme_00.png)

> Is, M., For, R., & At, E. (2021). An image is worth 16x16 words. _The International Conference on Learning Representations_.

Vision Transformer 将图像分割成小 patch，将每个 patch 当成 Transformer encoder 中的输入序列，达到了极佳的速度-精度平衡。ViT 通常需要很大的数据集（JFT-300M）才能达到比较好的效果。

但是，ViT 并不适合用作 general purpose backbone，因为图像都是稠密的，当输入图像分辨率很高时，每个 patch 得到的信息将会变少，同时在$O(n^2)$（$n$为图像大小）复杂度下算力消耗大。

虽然 Swin Transformer 不一定是最能刷榜的，但是是最有思想的。

> Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. In _Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017_ (Vol. 2017-January, pp. 5987–5995). Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/CVPR.2017.634

### To overcome…

hierachical feature maps => conveniently leverage advanced techniques for dense predicion such as FPN or U-Net

shifted windows

linear complexity to image size <= The numbers of patches in each window is fixed (previous: quadratic)

## Method

Swin Transformer Tiny 结构如(a)，选取$4\times4$大小的 patch（3 个通道，一共 48 个像素），Linear Embedding（是个全连接层吗？看代码）将 patch 投影成一个$C$维向量。Stage 1 的 Swin Transformer Block 维护一定数量的$\left(\frac H4 \times \frac W4\right)$的 token。

为了得到层级式的表达，token 的数目在每两个 Swin Transformer Block 之间会经过 Patch Merging 减到原来的$1/4$，也就是$2\times 2$的 patch 会被融合成一个$4C$大小的向量，然后经过一个全连接（linear layer），输出维度为$2C$的向量。Stage 2 的 Swin Transformer Block 的分辨率就是$\left(\frac H8 \times \frac W8\right)$；Stage 3 和 4 与 2 相同，所以他们的输出分别是$\left(\frac H{16} \times \frac W{16}\right)$和$\left(\frac H{32} \times \frac W{32}\right)$。该网络可以和 ResNet、VGGNet 产生相同分辨率的 feature map，可以方便地替换之而成为新的 backbone。

介绍了 Swin Transformer 之后，我们介绍其中最重要的，Swin Transformer Block。STB 将传统 Attention 机制中的多头自注意力模块换成了 Shifted Window 模块。在(b)中，STB 包括一个 LayerNorm 块，一个基于多头自注意力机制的 Swin 块，又一个 LayerNorm 块，最后一个两层的、中间一个 GELU 激活函数的多层感知机。

![HiT-arch-v2_00](media/Swin-Transformer/HiT-arch-v2_00.png)

### key design

理解这个<i>shift<b>ed</b></i>，注意用的是 shifted 而不是 sliding 或 shifting，是现在完成时而不是进行时，所以在神经网络训练过程中是不会出现 shift 这个操作的，这个我最开始也没想明白。

all _query_ patches within a window share the same _key_ set1 ,

![image-20211125174813927](media/Swin-Transformer/image-20211125174813927.png)

### 计算复杂度问题

对于将图像用于 Transformer 来说，由于 Transformer 处理的是序列。如果要处理一张$H\times W$的图像，需要将图像当成序列输入，将图像切分为$h \times w$的 patch，MSA 的时间复杂度可以表示为

$$
\Omega(\textrm{MSA}) = 4hwC^2+2\left(hw\right)^2C
$$

而如果将图像划分为若干个 patch，每个 window 中有$M\times M$个 patch，这种 Window-MSA 的时间复杂度为

$$
\Omega(\textrm{W-MSA}) = 4hwC^2+2M^2hwC
$$

MSA 对$hw$是平方级，而 W-MSA 对$hw$是线性级，这就很好地处理了随着图像分辨率的提高造成的计算复杂度快速上升的问题。

**提问：它对$M$不是平方级的吗？$M$不会增大吗？**

### Shifted Windows

在单个 Window-based self-attention 模块中没有跨 Window 的连接。

下一块中，作者对 window 进行了$\left(\lfloor\frac M 2\rfloor, \lfloor\frac M 2 \rfloor\right)$的位移（displace）。

所以，连续两层的 Swin transformer 的计算可以如下表示：

$$
\begin{aligned}
&\hat{\mathbf{z}}^{l}=\textrm{W-MSA}\left(\textrm{LN}\left(\mathbf{z}^{l-1}\right)\right)+\mathbf{z}^{l-1} \\
&\mathbf{z}^{l}=\operatorname{MLP}\left(\textrm{LN}\left(\hat{\mathbf{z}}^{l}\right)\right)+\hat{\mathbf{z}}^{l} \\
&\hat{\mathbf{z}}^{l+1}=\operatorname{SW-MSA}\left(\textrm{LN}\left(\mathbf{z}^{l}\right)\right)+\mathbf{z}^{l} \\
&\mathbf{z}^{l+1}=\operatorname{MLP}\left(\textrm{LN}\left(\hat{\mathbf{z}}^{l+1}\right)\right)+\hat{\mathbf{z}}^{l+1}
\end{aligned}
$$

$\hat{\mathbf z}^l$指第$l$层(S)W-MSA 输出的特征向量，$\hat{\mathbf z}^l$指第$l+1$层 MLP 输出的特征向量

但是可以看到，Shifted window 会产生更多的 window，也就是周围一圈大小小于$M\times M$的 window，故采用 cylic shifting，也就是，循环移位，也就是，一个 window 里有好多 sub-window，然后在 self-attention 计算的时候将 sub-window 使用 mask 提取出来。

### 相对偏置

在计算自注意力时，相较于原方法，在每个头上加了一个 relative position embedding 的偏置$B \in \mathbb{R}^{M^2\times M^2}$的偏置，修正后的公式如下：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{SoftMax}(\frac{QK^\top}{\sqrt{d}}+B)V
$$

$d$是*query/key* dimension

### 结构变种

上文介绍主要基于 Swin Transformer Tiny，还有 Small，Big，Large 等大小。<del>（移动端的，PC 端的，服务器端的，用来刷榜的）</del>

## 实验结果

在各数据集上的测试、与 ViT、ResNeXt 的对比、消融实验（Ablation study），略

## 总结

代码解读可参考：https://zhuanlan.zhihu.com/p/401661320

Marr Prize
