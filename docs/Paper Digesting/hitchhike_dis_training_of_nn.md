# A Hitchhiker’s Guide On Distributed Training Of Deep Neural Networks

> 小郭今天吃什么？
>
> 哈，为什么吃分布式训练呢？
>
> 模型非常大，一口吃不下！

论文地址：[arXiv](https://arxiv.org/abs/1810.11787v1) [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0743731518308712)

模型层面最近卷花了，来分布式冷静一下。

## 主要任务

分布式训练的主要任务有二：

1. 训练速度随着GPU数量的增加应该是线性下降的
2. 在高延迟的网络状况下也要有足够的容错

## Overview

分布式训练有两种主要途径：数据并行和模型并行。

数据并行中，每个计算节点都均分有数据的一个子集，在本地维护一个权值，在更新时每个局部的权值被收集起来统一更新，新的权值分发给每个节点，也就是说所有权值是相同的。

模型并行中，模型被分成多个块放在不同的计算节点上。模型并行的主要原因是模型非常大，一块GPU装不下。

## 分布式训练框架的自我修养

一个分布式训练框架，在单机训练上有两点改进：分布式训练算法和节点间交流。

### 分布式训练算法

SGD（Stochastic Gradient Descent）不仅是单机训练中很流行的算法，也是分布式训练中的主流。对SGD的讨论也很容易扩展到Adam、RMSProp等算法中。分布式SGD主要有两种：**同步SGD**和**异步SGD**。

同步SGD意在模仿单机训练，其收敛性是可以证明的。

异步SGD降低了计算节点之间的耦合。但是，解耦带来了并行性的提升，也造成了稳定性和准确性的下降。

研究更倾向于大规模同步SGD，主要方法是增大mini-batch size，降低学习率。有神人使用64K的mini-batch在四分钟之内完成了ImageNet上的训练。

### 节点间通信

主要是大数据打好的基础，包括分布式文件系统Google File System、Hadoop Distributed File System，以及高性能计算使用的[collective communication primitives](https://journals.sagepub.com/doi/10.1177/1094342005051521)等。TensorFlow和PyTorch就是用这些primitives进行节点间高效的All Reduce操作。梯度压缩和混合精度可以提高整个系统的吞吐量。



### 同步SGD怎么了

同步SGD在数学上一定收敛，但是在大型分布式系统中有以下问题：

#### 拖后腿者

80%倒数第二在2s之内到达，但是只有30%的倒数第一在2s之内到达

### Synchronization Barrier

所有节点都在等主节点算完把新的权值发给他们。

### Single Point of Faliure

单点宕机，你等不等？等，就是一台机子挂了一个系统。

### 容错

这就是消息传递机制的实现问题了，如何保证传输的权值是正确的，如果有错如何恢复

