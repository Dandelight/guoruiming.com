# faiss

参考资料：https://github.com/liqima/faiss_note

`faiss` 是一款用于向量搜索的开源软件，同时提供了 CPU 和 GPU 版本，在向量数据库领域其重要程度堪比 Pytorch 的计算核心。faiss 分为两篇讲解，一篇应用篇，

## 安装

`conda` 是安装 `faiss` 的最便捷方式，`meta` 官方的发包渠道也只有 `conda`。目前最新的版本是 `1.7.4`。（目前是什么时候可以参考 `commit` 日期）

```shell
# 安装 cpu 版本
# 更新 conda
conda update conda
# 安装 faiss-cpu
conda install faiss-cpu -c pytorch
# 测试安装是否成功，首次运行可能需要等很久
python -c "import faiss; print(faiss.__version__)"
```

## 数据准备

`faiss` 可以处理固定维度 $d$ 的向量集合，这样的集合这里用二维数组表示。一般来说，我们需要两个数组：

1. data。包含被索引的所有向量元素；

2. query。索引向量，我们需要根据索引向量的值返回 xb 中的最近邻元素。

为了对比不同索引方式的差别，在下面的例子中我们统一使用完全相同的数据，即维数 $d = 512$，data $X$ 包含 2000 个向量，每个向量符合正态分布的情况。需要注意的是，faiss 需要数组中的元素只能是 32 位浮点数格式，也就是由 `dtype = 'float32'` 的 `numpy` 数组转换过来。

## 精确索引

使用 `faiss` 是围绕 `index` 对象进行的。`index` 中包含被索引的数据库向量，在索引时可以选择不同方式的预处理来提高索引的效率，表现维不同的索引类型。在精确搜索时选择最简单的 `IndexFlatL2` 索引类型。

`IndexFlatL2` 类型遍历计算查询向量与被查询向量的 L2 精确距离，不需要 `train` 操作（大部分 `index` 类型都需要 train 操作）。

在构建 index 时要提供相关参数，这里是向量维数 $d$。构建完成之后可以通过 `add()` 和 `search()` 进行查询。
