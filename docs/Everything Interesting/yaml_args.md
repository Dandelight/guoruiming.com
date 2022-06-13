# `YAML`与`ArgumentParser`的共舞

你，见过怎样的深度学习代码？

是一堆`ArgParse`看不懂的惆怅？

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
# 略去几十个add_argument
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

主要问题来自于，`Python` 的 `dict` 的内容是通过 `__getitem__` 和 `__setitem__` 获取的，而 `ArgumentParser` 的内容作为属性获取。

解决问题和很简单，转换一下：

```python
from os import get_inheritable
import yaml
import argparse

default_config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument(
    '-c',
    '--config_yaml',
    default=
    'train.yml',
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')


# YAML should override the argparser's content
def _parse_args_and_yaml(given_parser=None):
    if given_parser == None:
        given_parser = default_config_parser
    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.config_yaml:
        with open(given_configs.config_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            given_parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = given_parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def parse_args_and_yaml(arg_parser=None):
    return _parse_args_and_yaml(arg_parser)[0]


if __name__ == "__main__":
    args, args_text = _parse_args_and_yaml()
```

[^unknown]:
    ArgumentParser 和 YAML 在 Python 中的共同使用 / 用 YAML 更新 Parser
    https://blog.51cto.com/u_15127596/4233240
