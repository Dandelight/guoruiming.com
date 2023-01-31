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

## [Lightning](https://lightning.ai/)

官网上慷慨地贴了一段自编码器的代码，一切尽在代码中了。

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 3))
		self.decoder = nn.Sequential(
      nn.Linear(3, 64),
      nn.ReLU(),
      nn.Linear(64, 28 * 28))

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
```

虽然真的想说，**组合胜过继承**，但看在它还挺直白好用，加上 21K Star 的份上，个人觉得还挺可以的。Lightning AI 也是一家比较出众的创业公司，让我们静观其变吧。
