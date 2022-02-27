## VinVL

VinVL 可以看做 Oscar 的互补工作。Oscar 更着重于语言模型的实现，通过（图像特征、关键目标、句子）三元组训练模型，让模型对目标有更深刻的印象；而 VinVL 模型则是对特征提取部分的改进（虽然也没有太大改进，主要 1. 采用了 ResNet152-C4 模型 2. 合并了多个数据集使用了巨量数据进行训练

## 其他方法

### class-agonistic non-maximal suppression
