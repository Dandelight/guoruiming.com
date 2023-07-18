```python
# 在训练脚本开头初始化一个新的运行项；
wandb.init()
# 跟踪超参数
wandb.config()
# 监听模型梯度
wandb.watch(model)
# 在训练循环中持续记录变化的指标；
wandb.log
# 保存运行项相关文件，如模型权值；
wandb.save
# 运行指定运行项时，恢复代码状态。
wandb.restore
```

使用方法

```python
# Flexible integration for any Python script
import wandb

# 1. Start a W&B run
wandb.init(project='gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training here
‍
# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})
```

## 保存代码

`wandb.init` 有一个 `save_code` 选项，但该选项只能保存调用了 `wandb.init` 的代码。如果要保存整个项目，可以

- 在 `wandb.init()` 之后调用  `wandb.run.log_code(".")`
- 给 `wandb.init`  传一个 `code_dir` 参数: `wandb.init(settings=wandb.Settings(code_dir="."))`
