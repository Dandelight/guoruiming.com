# `Hydra`：`Python` 配置管理

[`hydra`](https://hydra.cc/) 是一个由 Meta (Facebook) 开源的配置管理软件。其基于 `yaml` 语言，并支持模块化配置。它基于 [`OmegaConf`](https://omegaconf.readthedocs.io/)。具体使用请看官方文档，下面依照文档列举一下使用范例。

## 基础使用

```yaml
# conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

```python
# app.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    app()
```

```shell
$ python my_app.py
db:
  driver: mysql
  pass: secret
  user: omry
```

```shell
$ python my_app.py db.user=root db.pass=1234
db:
  driver: mysql
  user: root
  pass: 1234
```

## 组合使用

假设我们有一个项目，可以用 `MySQL` 或者 `PostgreSQL` 作为数据库。依据控制反转的思想，我们想在不影响代码的情况下在两种数据库之间自由切换。现在代码已经写好，目录结构如下

```
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   └── __init__.py
└── app.py
```

其中主配置文件 `conf/config.yaml` 如下

```yaml
defaults:
  - db: mysql
```

其中的 `defaults.db: mysql` 表示默认使用 `config/db/mysql.yaml` 作为 `db` 这一项的配置。`app.py` 和上一节中相同。正常使用 `python app.py` 的结果显然是使用 `mysql` 作为配置，但如果我们想使用 `postgresql`，只需要在命令行指定

```shell
$ python my_app.py db=postgresql db.timeout=20
db:
  driver: postgresql
  pass: drowssap
  timeout: 20
  user: postgres_user
```

## Multirun

使用 `-m|--multirun` 参数可以通过命令行控制使用不同参数运行多次，适用于持续集成等场景。命令如下：

```shell
$ python my_app.py --multirun db=mysql,postgresql
[HYDRA] Sweep output dir : multirun/2020-01-09/01-16-29
[HYDRA] Launching 2 jobs locally
[HYDRA]        #0 : db=mysql
db:
  driver: mysql
  pass: secret
  user: omry

[HYDRA]        #1 : db=postgresql
db:
  driver: postgresql
  pass: drowssap
  timeout: 10
  user: postgres_user
```

## 在深度学习研究项目中与 `Pytorch-lightning` 结合使用

[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) 提供了一个友好的，基于 `hydra` 和 `pytorch-lightning` 的项目模板（或者按 `Spring Boot` 的惯例，叫做 `starter`）。

## `YAML`与`ArgumentParser`的共舞（过时）

> 此文档已过时，个人认为 `argparse` 相比于 `hydra` 略逊一筹——从一开始我就是想把 `argparse` 换掉的。

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
