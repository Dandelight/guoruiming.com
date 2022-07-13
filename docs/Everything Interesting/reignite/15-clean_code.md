# Clean Code

建议参考：https://github.com/chenyuntc/pytorch-best-practice

或者我的一份 fork：https://github.com/Dandelight/pytorch-best-practice

## [ignite](https://pytorch.org/ignite/index.html)

PyTorch 没有高层 API——至少是 `Keras` 那个级别的三行代码走天下。一方面，这有助于工程师集中精力于核心功能，提高竞争力；另一方面，社区却迟迟没有给出一个合适的高层框架，代码库里各种 home-made framework 满天飞，一个 `acc` 或者 `pr-curve` 都要自己写，反射机制就更费劲了。自己也不是没造过轮子，不过看到昇思和飞桨都自带高层 API 瞬间感觉手里的 Torch 不香了。

[Ignite](https://github.com/pytorch/ignite) is a **library** that provides three high-level features:

- Extremely simple engine and event system
- Out-of-the-box metrics to easily evaluate models
- Built-in handlers to compose training pipeline, save artifacts and log parameters and metrics

这里甚至有类似 Spring Initializr 的 [code generator](https://code-generator.pytorch-ignite.ai/)，通过简单的选项一键生成合适的初始代码。
