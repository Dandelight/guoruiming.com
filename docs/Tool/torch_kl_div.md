# PyTorch `kl_div` 的第一个参数居然需要是对数？

> To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space. The argument target may also be provided in the log-space if `log_target = True`.

也就是说，`torch.kl_div` 的 `inputs` 参数希望已经是 `log` 之后的值，而 `target` 是原来的 `target`。

其他损失函数也各有自己的假设

- `softmax` 需要注意的是，输入 `input` 要求为 `logit`（模型最后一层输出，不经过 `softmax`），`target`为该样本对应的类别, `torch.long` 类型（`int64`）。
- `torch.nn.functional.negative_log_likelihood`、 `torch.nn.BECLoss` 和 `kl_div` 一样，也是希望 `inputs` 是对数函数的输出结果。
- `BCEWithLogitsLoss` 和 `softmax` 一样，输入是 `logits`，剩余部分与 `BCELoss` 一致、
- `MultiLabelSoftMarginLosws` 和 `BCEWithLogitsLoss` 的效果是一样的

这里的一切麻烦都是因为 `torch` 需要保证数值稳定性……但我还是希望像个数学家一样，只写公式，数值稳定性让程序员们搞定吧。

<https://zhuanlan.zhihu.com/p/267787260>
