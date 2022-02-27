# `YAML`与`ArgumentParser`的共舞

你，见过怎样的深度学习代码？

是一堆 ArgParse 看不懂的惆怅？

```python
importy argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
# 略去十个add_argument
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
cfgs = parser.parse_args()
```

还是一堆`if-elif-else`的迷茫？

```python
def get_model(name, num_classes=10, stem=False, verbose=True, **block_kwargs):
    # AlexNet
    if name in ["alexnet_dnn"]:
        model = alexnet.dnn(num_classes=num_classes, stem=stem, name=name, **block_kwargs)
    # VGG
    elif name in ["vgg_dnn_11"]:
        model = vgg.dnn_11(num_classes=num_classes, name=name, **block_kwargs)
    # 略去一百多个 elif
    # MLP Mixer
    elif name in ["mixer_l"]:
        model = mixer.large(num_classes=num_classes, name=name, **block_kwargs)
    else:
        raise NotImplementedError
    return model
```

还是用命令行之后找不到参数的痛苦？

```bash
CUDA_VISIBLE_DEVICES=0 python train.py  --log_dir logs/log_rs --batch_size 2 --dataset_root /path/to/imagenet # 略去十几个参数
```

在此文中，我们将一一解决这些问题！
